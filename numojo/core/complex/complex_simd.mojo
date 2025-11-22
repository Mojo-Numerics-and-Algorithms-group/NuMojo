# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Implement the ComplexSIMD type and its operations.

This module provides a ComplexSIMD type that represents complex numbers using SIMD
operations for efficient computation. It supports basic arithmetic operations
like addition, subtraction, multiplication, and division, as well as other
complex number operations like conjugation and absolute value.

The implementation allows for vectorized operations on complex numbers which can
significantly improve performance for numerical computations.
"""

from math import sqrt, sin, cos
from numojo.core.complex.complex_dtype import ComplexDType

alias ComplexScalar[cdtype: ComplexDType] = ComplexSIMD[cdtype, width=1]
"""ComplexScalar alias is for internal purposes (width=1 specialization)."""
alias CScalar[cdtype: ComplexDType] = ComplexSIMD[cdtype, width=1]
"""User-friendly alias for scalar complex numbers."""


@register_passable("trivial")
struct ComplexSIMD[cdtype: ComplexDType, width: Int = 1](
    ImplicitlyCopyable, Movable, Stringable, Writable
):
    """
    A SIMD-enabled complex number container (SoA layout).

    Fields:
        re: SIMD vector of real parts.
        im: SIMD vector of imaginary parts.

    The parameter `cdtype` determines the component precision (e.g. cf32, cf64).
    The parameter `width` is the SIMD lane count; when `width == 1` this acts like a scalar complex number.

    Example (scalar):
        var a = ComplexSIMD[cf32](1.0, 2.0)
        var b = ComplexSIMD[cf32](3.0, 4.0)
        print(a + b)  # (4.0 + 6.0 j)

    Example (SIMD width=2):
        var a2 = ComplexSIMD[cf32, 2](
            SIMD[cf32._dtype, 2](1.0, 1.5),
            SIMD[cf32._dtype, 2](2.0, -0.5)
        )
        print(a2)

    Convenience factories:
        ComplexSIMD[cf64].zero()
        ComplexSIMD[cf64].one()
        ComplexSIMD[cf64].i()
        ComplexSIMD[cf64].from_polar(2.0, 0.5)
    """

    alias dtype: DType = cdtype._dtype
    """Component dtype alias (underlying real/imag dtype)."""

    var re: SIMD[Self.dtype, width]
    var im: SIMD[Self.dtype, width]

    #--- Internal helper for broadcasting scalar to SIMD lanes ---
    @staticmethod
    @always_inline
    fn _broadcast(val: Scalar[Self.dtype]) -> SIMD[Self.dtype, Self.width]:
        return SIMD[Self.dtype, Self.width](val)

    # --- Constructors ---
    @always_inline
    fn __init__(out self, other: Self):
        self = other

    @always_inline
    fn __init__(
        out self,
        re: SIMD[Self.dtype, Self.width],
        im: SIMD[Self.dtype, Self.width],
    ):
        self.re = re
        self.im = im

    @always_inline
    fn __init__(out self, val: SIMD[Self.dtype, Self.width]):
        self.re = val
        self.im = val

    @always_inline
    fn __init__(out self, re: Scalar[Self.dtype], im: Scalar[Self.dtype]):
        self.re = SIMD[Self.dtype, Self.width](re)
        self.im = SIMD[Self.dtype, Self.width](im)

    # Factory constructors.
    @staticmethod
    fn zero() -> Self:
        return Self(Self._broadcast(0), Self._broadcast(0))

    @staticmethod
    fn one() -> Self:
        return Self(Self._broadcast(1), Self._broadcast(0))

    @staticmethod
    fn i() -> Self:
        return Self(Self._broadcast(0), Self._broadcast(1))

    @staticmethod
    fn from_real_imag(re: Scalar[Self.dtype], im: Scalar[Self.dtype]) -> Self:
        return Self(re, im)

    @staticmethod
    fn from_polar(r: Scalar[Self.dtype], theta: Scalar[Self.dtype]) -> Self:
        return Self(
            Self._broadcast(r * cos(theta)),
            Self._broadcast(r * sin(theta)),
        )

    # --- Arithmetic operators ---
    fn __add__(self, other: Self) -> Self:
        return Self(self.re + other.re, self.im + other.im)

    fn __iadd__(mut self, other: Self):
        self.re += other.re
        self.im += other.im

    fn __sub__(self, other: Self) -> Self:
        return Self(self.re - other.re, self.im - other.im)

    fn __isub__(mut self, other: Self):
        self.re -= other.re
        self.im -= other.im

    fn __mul__(self, other: Self) -> Self:
        # (a+bi)(c+di) = (ac - bd) + (ad + bc)i
        return Self(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    fn __imul__(mut self, other: Self):
        var new_re = self.re * other.re - self.im * other.im
        self.im = self.re * other.im + self.im * other.re
        self.re = new_re

    fn __truediv__(self, other: Self) -> Self:
        """
        Naive complex division.
        """
        var denom = other.re * other.re + other.im * other.im
        return Self(
            (self.re * other.re + self.im * other.im) / denom,
            (self.im * other.re - self.re * other.im) / denom,
        )

    fn __itruediv__(mut self, other: Self):
        var denom = other.re * other.re + other.im * other.im
        var new_re = (self.re * other.re + self.im * other.im) / denom
        self.im = (self.im * other.re - self.re * other.im) / denom
        self.re = new_re

    fn reciprocal(self) -> Self:
        """
        Returns 1 / self.

        If self == 0 (all lanes), division by zero will occur (no guard yet).
        """
        var d = self.norm()
        return Self(self.re / d, -self.im / d)

    # --- Power helpers ---
    fn elem_pow(self, other: Self) -> Self:
        """
        Component-wise power: (re^other.re, im^other.im).
        """
        return Self(self.re**other.re, self.im**other.im)

    fn elem_pow(self, exponent: Scalar[Self.dtype]) -> Self:
        """
        Component-wise scalar exponent applied separately to real and imaginary parts.
        """
        return Self(self.re**exponent, self.im**exponent)

    fn __pow__(self, n: Int) -> Self:
        """
        Integer power using exponentiation by squaring.

        For negative n: returns reciprocal(pow(self, -n)).
        """
        if n == 0:
            return Self.one()
        var base = self
        var exp = n
        var result = Self.one()
        var is_negative = exp < 0
        if is_negative:
            exp = -exp
        while exp > 0:
            if (exp & 1) == 1:
                result = result * base
            base = base * base
            exp >>= 1
        if is_negative:
            return Self.one() / result
        return result

    # --- Unary operators ---
    fn __pos__(self) -> Self:
        return self

    fn __neg__(self) -> Self:
        return Self(-self.re, -self.im)

    # --- Helpers ---
    @staticmethod
    @always_inline
    fn _abs_simd(
        x: SIMD[Self.dtype, Self.width]
    ) -> SIMD[Self.dtype, Self.width]:
        return sqrt(x * x)

    # --- Equality ---
    fn __eq__(self, other: Self) -> Bool:
        return (self.re == other.re) and (self.im == other.im)

    fn __ne__(self, other: Self) -> Bool:
        return ~(self == other)

    fn allclose(
        self,
        other: Self,
        *,
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
    ) -> Bool:
        """
        Approximate equality for scalar complexes (width == 1).

        For SIMD width > 1, each lane must satisfy the tolerance criteria.
        TODO: Optionally return a SIMD[Bool] mask instead of a single Bool.
        """
        var diff_re = Self._abs_simd(self.re - other.re)
        var diff_im = Self._abs_simd(self.im - other.im)
        var rtol_b = Self._broadcast(rtol)
        var atol_b = Self._broadcast(atol)
        var thresh_re = atol_b + rtol_b * Self._abs_simd(other.re)
        var thresh_im = atol_b + rtol_b * Self._abs_simd(other.im)
        var ok_re = diff_re <= thresh_re
        var ok_im = diff_im <= thresh_im
        return ok_re and ok_im

    # --- Representations ---
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        String formatting:
          width == 1: (re + im j)
          width > 1 : [(re0 + im0 j), (re1 + im1 j), ...]
        """
        try:
            @parameter
            if Self.width == 1:
                writer.write(String("({} + {} j)").format(self.re, self.im))
            else:
                var s = String("[")
                for i in range(0, Self.width):
                    if i > 0:
                        s += ", "
                    s += String("({} + {} j)").format(self.re[i], self.im[i])
                s += "]"
                writer.write(s)
        except e:
            writer.write("<<ComplexSIMD formatting error>>")

    fn __repr__(self) raises -> String:
        return String("ComplexSIMD[{}](re={}, im={})").format(
            String(Self.dtype), self.re, self.im
        )

    # --- Indexed access to real/imag vectors (NOT per lane complex extraction) ---
    fn __getitem__(self, idx: Int) raises -> SIMD[Self.dtype, Self.width]:
        if idx == 0:
            return self.re
        elif idx == 1:
            return self.im
        else:
            raise Error("Index out of range (0=real,1=imag)")

    fn __setitem__(
        mut self, idx: Int, value: SIMD[Self.dtype, Self.width]
    ) raises:
        if idx == 0:
            self.re = value
        elif idx == 1:
            self.im = value
        else:
            raise Error("Index out of range (0=real,1=imag)")

    fn __setitem__(mut self, idx: Int, value: Self) raises:
        if idx == 0:
            self.re = value.re
        elif idx == 1:
            self.im = value.im
        else:
            raise Error("Index out of range (0=real,1=imag)")

    fn item(self, idx: Int) raises -> SIMD[Self.dtype, Self.width]:
        return self[idx]

    fn itemset(mut self, val: Self):
        self.re = val.re
        self.im = val.im

    # --- Magnitude / norm / conjugate ---
    fn __abs__(self) -> SIMD[Self.dtype, Self.width]:
        return sqrt(self.re * self.re + self.im * self.im)

    fn norm(self) -> SIMD[Self.dtype, Self.width]:
        return self.re * self.re + self.im * self.im

    fn conj(self) -> Self:
        return Self(self.re, -self.im)

    fn real(self) -> SIMD[Self.dtype, Self.width]:
        return self.re

    fn imag(self) -> SIMD[Self.dtype, Self.width]:
        return self.im
