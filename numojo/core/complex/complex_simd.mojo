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
"""

from math import sqrt, sin, cos
from numojo.core.complex.complex_dtype import ComplexDType

alias ComplexScalar[cdtype: ComplexDType] = ComplexSIMD[cdtype, width=1]
"""ComplexScalar alias is for internal purposes (width=1 specialization)."""
alias CScalar[cdtype: ComplexDType] = ComplexSIMD[cdtype, width=1]
"""User-friendly alias for scalar complex numbers."""

alias `1j` = ComplexSIMD[_, width=1].i()
"""Constant representing the imaginary unit complex number 0 + 1j for cf32 in Python style."""

# TODO: add overloads for arithmetic functions to accept Scalar[dtype].
@register_passable("trivial")
struct ComplexSIMD[cdtype: ComplexDType = ComplexDType.float64, width: Int = 1](
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

    # --- Internal helper for broadcasting scalar to SIMD lanes ---
    @staticmethod
    @always_inline
    fn _broadcast(val: Scalar[Self.dtype]) -> SIMD[Self.dtype, Self.width]:
        return SIMD[Self.dtype, Self.width](val)

    # --- Constructors ---
    @always_inline
    fn __init__(out self, other: Self):
        """
        Copy constructor for ComplexSIMD.

        Initializes a new ComplexSIMD instance by copying the values from another instance.
        """
        self = other

    @always_inline
    fn __init__(
        out self,
        re: SIMD[Self.dtype, Self.width],
        im: SIMD[Self.dtype, Self.width],
    ):
        """
        Constructs a ComplexSIMD from SIMD vectors of real and imaginary parts.

        Args:
            re: SIMD vector containing the real components.
            im: SIMD vector containing the imaginary components.
        """
        self.re = re
        self.im = im

    @always_inline
    fn __init__(out self, val: SIMD[Self.dtype, Self.width]):
        """
        Constructs a ComplexSIMD where both real and imaginary parts are set to the same SIMD value.

        Args:
            val: SIMD vector to broadcast to both real and imaginary components.
        """
        self.re = val
        self.im = val

    # Factory constructors.
    @staticmethod
    fn zero() -> Self:
        """
        Returns a ComplexSIMD instance with all real and imaginary components set to zero.

        Example:
            ```mojo
            from numojo.prelude import *
            var comp = ComplexSIMD[cf64].zero()  # (0 + 0j)
            ```
        """
        return Self(Self._broadcast(0), Self._broadcast(0))

    @staticmethod
    fn one() -> Self:
        """
        Returns a ComplexSIMD instance representing the complex number 1 + 0j.

        Example:
            ```mojo
            from numojo.prelude import *
            var comp = ComplexSIMD[cf64].one()  # (1 + 0j)
            ```
        """
        return Self(Self._broadcast(1), Self._broadcast(0))

    @staticmethod
    fn i() -> Self:
        """
        Returns a ComplexSIMD instance representing the imaginary unit 0 + 1j.

        Example:
            ```mojo
            from numojo.prelude import *
            var comp = ComplexSIMD[cf64].i()  # (0 + 1j)
            ```
        """
        return Self(Self._broadcast(0), Self._broadcast(1))

    @staticmethod
    fn from_real_imag(re: Scalar[Self.dtype], im: Scalar[Self.dtype]) -> Self:
        """
        Constructs a ComplexSIMD instance from scalar real and imaginary values.

        Args:
            re: Scalar value for the real component.
            im: Scalar value for the imaginary component.

        Example:
            ```mojo
            from numojo.prelude import *
            var comp = ComplexSIMD[cf64].from_real_imag(2.0, 3.0)  # (2.0 + 3.0j)
            ```
        """
        return Self(re, im)

    @staticmethod
    fn from_polar(r: Scalar[Self.dtype], theta: Scalar[Self.dtype]) -> Self:
        """
        Constructs a ComplexSIMD instance from polar coordinates.

        Args:
            r: Magnitude (radius).
            theta: Angle (in radians).

        Returns:
            ComplexSIMD instance with real part r * cos(theta) and imaginary part r * sin(theta).

        Example:
            ```mojo
            from numojo.prelude import *
            var comp = ComplexSIMD[cf64].from_polar(2.0, 0.5)
            ```
        """
        return Self(
            Self._broadcast(r * cos(theta)),
            Self._broadcast(r * sin(theta)),
        )

    # --- Arithmetic operators ---
    fn __add__(self, other: Self) -> Self:
        """
        Returns the element-wise sum of two ComplexSIMD instances.

        Args:
            other: Another ComplexSIMD instance.

        Returns:
            ComplexSIMD instance where each lane is the sum of corresponding lanes.
        """
        return Self(self.re + other.re, self.im + other.im)

    fn __add__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Returns the sum of this ComplexSIMD instance and a scalar added to the real part.

        Args:
            other: Scalar value to add to the real component.

        Returns:
            ComplexSIMD instance where each lane's real part is increased by the scalar.
        """
        return Self(self.re + Self._broadcast(other), self.im)

    fn __iadd__(mut self, other: Self):
        """
        In-place addition of another ComplexSIMD instance.

        Args:
            other: Another ComplexSIMD instance.
        """
        self.re += other.re
        self.im += other.im

    fn __iadd__(mut self, other: Scalar[Self.dtype]):
        """
        In-place addition of a scalar to the real part of this ComplexSIMD instance.

        Args:
            other: Scalar value to add to the real component.
        """
        self.re += Self._broadcast(other)

    fn __radd__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Returns the sum of a scalar and this ComplexSIMD instance, adding to the real part.

        Args:
            other: Scalar value to add to the real component.

        Returns:
            ComplexSIMD instance where each lane's real part is increased by the scalar.
        """
        return Self(Self._broadcast(other) + self.re, self.im)

    fn __sub__(self, other: Self) -> Self:
        """
        Returns the element-wise difference of two ComplexSIMD instances.

        Args:
            other: Another ComplexSIMD instance.

        Returns:
            ComplexSIMD instance where each lane is the difference of corresponding lanes.
        """
        return Self(self.re - other.re, self.im - other.im)

    fn __sub__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Returns the difference of this ComplexSIMD instance and a scalar subtracted from the real part.

        Args:
            other: Scalar value to subtract from the real component.

        Returns:
            ComplexSIMD instance where each lane's real part is decreased by the scalar.
        """
        return Self(self.re - Self._broadcast(other), self.im)

    fn __isub__(mut self, other: Self):
        """
        In-place subtraction of another ComplexSIMD instance.

        Args:
            other: Another ComplexSIMD instance.
        """
        self.re -= other.re
        self.im -= other.im

    fn __isub__(mut self, other: Scalar[Self.dtype]):
        """
        In-place subtraction of a scalar from the real part of this ComplexSIMD instance.

        Args:
            other: Scalar value to subtract from the real component.
        """
        self.re -= Self._broadcast(other)

    fn __rsub__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Returns the difference of a scalar and this ComplexSIMD instance, subtracting from the real part.

        Args:
            other: Scalar value to subtract from the real component.

        Returns:
            ComplexSIMD instance where each lane's real part is (scalar - self.re).
        """
        return Self(Self._broadcast(other) - self.re, -self.im)

    fn __mul__(self, other: Self) -> Self:
        """
        Returns the element-wise product of two ComplexSIMD instances.

        Args:
            other: Another ComplexSIMD instance.

        Returns:
            ComplexSIMD instance where each lane is the product of corresponding lanes, using complex multiplication: (a+bi)(c+di) = (ac - bd) + (ad + bc)i.
        """
        return Self(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    fn __mul__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Returns the product of this ComplexSIMD instance and a scalar.

        Args:
            other: Scalar value to multiply with both real and imaginary parts.

        Returns:
            ComplexSIMD instance where each lane is scaled by the scalar.
        """
        var scalar_simd = Self._broadcast(other)
        return Self(self.re * scalar_simd, self.im * scalar_simd)

    fn __imul__(mut self, other: Self):
        """
        In-place complex multiplication with another ComplexSIMD instance.

        Args:
            other: Another ComplexSIMD instance.
        """
        var new_re = self.re * other.re - self.im * other.im
        self.im = self.re * other.im + self.im * other.re
        self.re = new_re

    fn __imul__(mut self, other: Scalar[Self.dtype]):
        """
        In-place multiplication of this ComplexSIMD instance by a scalar.

        Args:
            other: Scalar value to multiply with both real and imaginary parts.
        """
        var scalar_simd = Self._broadcast(other)
        self.re *= scalar_simd
        self.im *= scalar_simd

    fn __rmul__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Returns the product of a scalar and this ComplexSIMD instance.

        Args:
            other: Scalar value to multiply with both real and imaginary parts.

        Returns:
            ComplexSIMD instance where each lane is scaled by the scalar.
        """
        var scalar_simd = Self._broadcast(other)
        return Self(scalar_simd * self.re, scalar_simd * self.im)

    fn __truediv__(self, other: Self) -> Self:
        """
        Performs element-wise complex division of two ComplexSIMD instances.

        Args:
            other: Another ComplexSIMD instance to divide by.

        Returns:
            ComplexSIMD instance where each lane is the result of dividing the corresponding lanes:
            (a + bi) / (c + di) = [(ac + bd) / (c^2 + d^2)] + [(bc - ad) / (c^2 + d^2)]i
            where a, b are self.re, self.im and c, d are other.re, other.im.
        """
        var denom = other.re * other.re + other.im * other.im
        return Self(
            (self.re * other.re + self.im * other.im) / denom,
            (self.im * other.re - self.re * other.im) / denom,
        )

    fn __truediv__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Performs element-wise division of this ComplexSIMD instance by a scalar.

        Args:
            other: Scalar value to divide both real and imaginary parts by.

        Returns:
            ComplexSIMD instance where each lane is divided by the scalar.
        """
        var scalar_simd = Self._broadcast(other)
        return Self(self.re / scalar_simd, self.im / scalar_simd)

    fn __itruediv__(mut self, other: Self):
        """
        Performs in-place element-wise complex division of self by another ComplexSIMD instance.

        Args:
            other: Another ComplexSIMD instance to divide by.
        """
        var denom = other.re * other.re + other.im * other.im
        var new_re = (self.re * other.re + self.im * other.im) / denom
        self.im = (self.im * other.re - self.re * other.im) / denom
        self.re = new_re

    fn __itruediv__(mut self, other: Scalar[Self.dtype]):
        """
        Performs in-place element-wise division of this ComplexSIMD instance by a scalar.

        Args:
            other: Scalar value to divide both real and imaginary parts by.
        """
        var scalar_simd = Self._broadcast(other)
        self.re /= scalar_simd
        self.im /= scalar_simd

    fn __rtruediv__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Performs element-wise division of a scalar by this ComplexSIMD instance.

        Args:
            other: Scalar value to be divided by this ComplexSIMD instance.

        Returns:
            ComplexSIMD instance where each lane is the result of dividing the scalar by the corresponding lane:
            other / (a + bi) = [other * a / (a^2 + b^2)] + [-other * b / (a^2 + b^2)]i
            where a, b are self.re, self.im.
        """
        var denom = self.re * self.re + self.im * self.im
        var scalar_simd = Self._broadcast(other)
        return Self(
            (scalar_simd * self.re) / denom,
            (-scalar_simd * self.im) / denom,
        )

    fn reciprocal(self) raises -> Self:
        """
        Returns the element-wise reciprocal (1 / self) of the ComplexSIMD instance.

        Returns:
            ComplexSIMD instance representing the reciprocal of each lane:
            1 / (a + bi) = (a / (a^2 + b^2)) + (-b / (a^2 + b^2)).
        """
        var d = self.norm()
        if d == 0:
            raise Error(
                "Cannot compute reciprocal of zero norm complex number."
            )
        return Self(self.re / d, -self.im / d)

    # --- Power helpers ---
    fn elem_pow(self, other: Self) -> Self:
        """
        Raises each component of this ComplexSIMD to the power of the corresponding component in another ComplexSIMD.

        Args:
            other: Another ComplexSIMD instance.

        Returns:
            ComplexSIMD instance where each lane is (re^other.re, im^other.im).
        """
        return Self(self.re**other.re, self.im**other.im)

    fn elem_pow(self, exponent: Scalar[Self.dtype]) -> Self:
        """
        Raises each component of this ComplexSIMD to a scalar exponent.

        Args:
            exponent: Scalar exponent to apply to both real and imaginary parts.

        Returns:
            ComplexSIMD instance where each lane is (re^exponent, im^exponent).
        """
        return Self(self.re**exponent, self.im**exponent)

    fn __pow__(self, n: Int) -> Self:
        """
        Raises this ComplexSIMD to an integer.

        Args:
            n: Integer exponent.

        Returns:
            ComplexSIMD instance raised to the power n.
            For negative n, returns the reciprocal of self raised to -n.
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
        """
        Returns the positive value of this ComplexSIMD (identity operation).

        Returns:
            The same ComplexSIMD instance.
        """
        return self

    fn __neg__(self) -> Self:
        """
        Returns the negation of this ComplexSIMD.

        Returns:
            ComplexSIMD instance with both real and imaginary parts negated.
        """
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
        """
        Checks if two ComplexSIMD instances are exactly equal.

        Returns:
            True if both the real and imaginary parts are equal for all lanes, otherwise False.
        """
        return (self.re == other.re) and (self.im == other.im)

    fn __ne__(self, other: Self) -> Bool:
        """
        Checks if two ComplexSIMD instances are not equal.

        Returns:
            True if either the real or imaginary parts differ for any lane, otherwise False.
        """
        return ~(self == other)

    fn allclose(
        self,
        other: Self,
        *,
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
    ) -> Bool:
        """
        Checks if two ComplexSIMD instances are approximately equal within given tolerances.

        For each lane, compares the real and imaginary parts using the formula:
            abs(a - b) <= atol + rtol * abs(b)
        where a and b are the corresponding components of self and other.

        Args:
            other: Another ComplexSIMD instance to compare against.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            True if all lanes of both real and imaginary parts are within the specified tolerances, otherwise False.

        Note:
            For SIMD width > 1, all lanes must satisfy the tolerance criteria.
        """
        # TODO: Optionally return a SIMD[Bool] mask instead of a single Bool.
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
        Returns a string representation of the ComplexSIMD instance.

        For width == 1, the format is: (re + im j).
        For width > 1, the format is: [(re0 + im0 j), (re1 + im1 j), ...].
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
        """
        Returns a string representation of the ComplexSIMD instance for debugging. `ComplexSIMD[dtype](re=<real SIMD>, im=<imag SIMD>)`.
        """
        return String("ComplexSIMD[{}](re={}, im={})").format(
            String(Self.dtype), self.re, self.im
        )

    # --- Indexing ---
    fn __getitem__(self, idx: Int) raises -> SIMD[Self.dtype, Self.width]:
        """
        Returns the SIMD vector for the specified component.

        Args:
            idx: Index of the component (0 for real, 1 for imaginary).

        Returns:
            SIMD vector of the requested component.

        Raises:
            Error if idx is not 0 or 1.
        """
        if idx == 0:
            return self.re
        elif idx == 1:
            return self.im
        else:
            raise Error("Index out of range (0=real,1=imag)")

    fn __setitem__(
        mut self, idx: Int, value: SIMD[Self.dtype, Self.width]
    ) raises:
        """
        Sets the SIMD vector for the specified component.

        Args:
            idx: Index of the component (0 for real, 1 for imaginary).
            value: SIMD vector to assign.

        Raises:
            Error if idx is not 0 or 1.
        """
        if idx == 0:
            self.re = value
        elif idx == 1:
            self.im = value
        else:
            raise Error("Index out of range (0=real,1=imag)")

    fn __setitem__(mut self, idx: Int, value: Self) raises:
        """
        Sets the real or imaginary component from another ComplexSIMD instance.

        Args:
            idx: Index of the component (0 for real, 1 for imaginary).
            value: ComplexSIMD instance whose component will be assigned.

        Raises:
            Error if idx is not 0 or 1.
        """
        if idx == 0:
            self.re = value.re
        elif idx == 1:
            self.im = value.im
        else:
            raise Error("Index out of range (0=real,1=imag)")

    fn item(self, idx: Int) raises -> SIMD[Self.dtype, Self.width]:
        """
        Returns the SIMD vector for the specified component.

        Args:
            idx: Index of the component (0 for real, 1 for imaginary).

        Returns:
            SIMD vector of the requested component.

        Raises:
            Error if idx is not 0 or 1.
        """
        return self[idx]

    fn itemset(mut self, val: Self):
        """
        Sets both the real and imaginary components from another ComplexSIMD instance.

        Args:
            val: ComplexSIMD instance whose real and imaginary parts will be assigned to self.
        """
        self.re = val.re
        self.im = val.im

    # --- Magnitude / norm / conjugate ---
    fn __abs__(self) -> SIMD[Self.dtype, Self.width]:
        """
        Returns the magnitude (absolute value) of the complex number(s).

        Returns:
            SIMD vector containing the magnitude for each lane: sqrt(re^2 + im^2).
        """
        return sqrt(self.re * self.re + self.im * self.im)

    fn norm(self) -> SIMD[Self.dtype, Self.width]:
        """
        Returns the squared magnitude (norm) of the complex number(s).

        Returns:
            SIMD vector containing the squared magnitude for each lane: re^2 + im^2.
        """
        return self.re * self.re + self.im * self.im

    fn conj(self) -> Self:
        """
        Returns the complex conjugate of the ComplexSIMD instance.

        Returns:
            ComplexSIMD instance with the imaginary part negated: (re, -im).
        """
        return Self(self.re, -self.im)

    fn real(self) -> SIMD[Self.dtype, Self.width]:
        """
        Returns the real part(s) of the complex number(s).

        Returns:
            SIMD vector containing the real components.
        """
        return self.re

    fn imag(self) -> SIMD[Self.dtype, Self.width]:
        """
        Returns the imaginary part(s) of the complex number(s).

        Returns:
            SIMD vector containing the imaginary components.
        """
        return self.im
