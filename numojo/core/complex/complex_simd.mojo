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

comptime ComplexScalar = ComplexSIMD[_, width=1]
"""ComplexScalar alias is for internal purposes (width=1 specialization)."""

comptime CScalar = ComplexSIMD[_, width=1]
"""User-friendly alias for scalar complex numbers."""

comptime `1j` = ImaginaryUnit()
"""Constant representing the imaginary unit complex number 0 + 1j.
Enables Python like syntax for complex numbers, e.g., (3 + 4 * `1j`)."""


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

    Examples:
        ```mojo
        from numojo.prelude import *
        var a = ComplexSIMD[cf32](1.0, 2.0)
        var b = ComplexSIMD[cf32](3.0, 4.0)
        print(a + b)  # (4.0 + 6.0 j)

        # SIMD width=2:
        var a2 = ComplexSIMD[cf32, 2](
            SIMD[cf32._dtype, 2](1.0, 1.5),
            SIMD[cf32._dtype, 2](2.0, -0.5)
        )
        print(a2) # ( [1.0 2.0] + [1.5 -0.5]j )
        ```
    Convenience factories:
        ComplexSIMD[cf64].zero()
        ComplexSIMD[cf64].one()
        ComplexSIMD[cf64].i()
        ComplexSIMD[cf64].from_polar(2.0, 0.5)
    """

    comptime dtype: DType = cdtype._dtype
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

        Returns:
            ComplexSIMD instance with real part 0 and imaginary part 1 for all lanes.

        Examples:
            ```mojo
            from numojo.prelude import *

            # Create imaginary unit for different types
            var i_f64 = ComplexSIMD[cf64].i()  # (0 + 1j)
            var i_f32 = ComplexSIMD[cf32].i()  # (0 + 1j)
            print(i_f64)  # (0 + 1j)

            # Use in complex arithmetic
            var z = 3.0 + 4.0 * ComplexSIMD[cf64].i()  # 3 + 4j
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

    # FIXME: currently mojo doesn't allow overloading with both SIMD[Self.dtype, Self.width] and SIMD[*_, size=Self.width]. So keep SIMD[*_, size=Self.width] only for now. We need this method to create complex numbers with syntax like (1 + 2 * `1j`).
    fn __add__(self, other: SIMD[*_, size = Self.width]) -> Self:
        """
        Returns the sum of this ComplexSIMD instance and a SIMD vector added to the real part.

        Args:
            other: SIMD vector to add to the real component.

        Returns:
            ComplexSIMD instance where each lane's real part is increased by the corresponding lane in the SIMD vector.
        """
        return Self(self.re + other.cast[Self.dtype](), self.im)

    fn __add__(self, other: ImaginaryUnit) -> Self:
        """
        Returns the sum of this ComplexSIMD instance and the imaginary unit 1j.

        Args:
            other: Imaginary unit (1j) to add to this complex number.

        Returns:
            ComplexSIMD instance where each lane's imaginary part is increased by 1.
            If self = a + bj, then result = a + (b+1)j.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z = ComplexSIMD[cf64, 1](3.0, 2.0)  # 3 + 2j
            var result = z + `1j`  # 3 + 3j
            print(result)  # (3 + 3j)
            ```
        """
        return Self(self.re, self.im + Self._broadcast(1))

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

    fn __iadd__(mut self, other: SIMD[*_, size = Self.width]):
        """
        In-place addition of a SIMD vector to the real part of this ComplexSIMD instance.

        Args:
            other: SIMD vector to add to the real component.
        """
        self.re += other.cast[Self.dtype]()

    fn __iadd__(mut self, other: ImaginaryUnit):
        """
        In-place addition of the imaginary unit 1j to this ComplexSIMD instance.

        Args:
            other: Imaginary unit (1j) to add to this complex number.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z = ComplexSIMD[cf64, 1](3.0, 2.0)  # 3 + 2j
            z += `1j`  # Now z = 3 + 3j
            print(z)  # (3 + 3j)
            ```
        """
        self.im += Self._broadcast(1)

    fn __radd__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Returns the sum of a scalar and this ComplexSIMD instance, adding to the real part.

        Args:
            other: Scalar value to add to the real component.

        Returns:
            ComplexSIMD instance where each lane's real part is increased by the scalar.
        """
        return Self(Self._broadcast(other) + self.re, self.im)

    fn __radd__(self, other: SIMD[*_, size = Self.width]) -> Self:
        """
        Returns the sum of a SIMD vector and this ComplexSIMD instance, adding to the real part.

        Args:
            other: SIMD vector to add to the real component.

        Returns:
            ComplexSIMD instance where each lane's real part is increased by the corresponding lane in the SIMD vector.
        """
        return Self(other.cast[Self.dtype]() + self.re, self.im)

    fn __radd__(self, other: ImaginaryUnit) -> Self:
        """
        Returns the sum of the imaginary unit 1j and this ComplexSIMD instance.

        Args:
            other: Imaginary unit (1j) to add to this complex number.

        Returns:
            ComplexSIMD instance where each lane's imaginary part is increased by 1.
            If self = a + bj, then result = a + (b+1)j.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z = ComplexSIMD[cf64, 1](3.0, 2.0)  # 3 + 2j
            var result = `1j` + z  # 3 + 3j
            print(result)  # (3 + 3j)
            ```
        """
        return Self(self.re, self.im + Self._broadcast(1))

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

    fn __sub__(self, other: SIMD[*_, size = Self.width]) -> Self:
        """
        Returns the difference of this ComplexSIMD instance and a SIMD vector subtracted from the real part.

        Args:
            other: SIMD vector to subtract from the real component.

        Returns:
            ComplexSIMD instance where each lane's real part is decreased by the corresponding lane in the SIMD vector.
        """
        return Self(self.re - other.cast[Self.dtype](), self.im)

    fn __sub__(self, other: ImaginaryUnit) -> Self:
        """
        Subtracts the imaginary unit 1j from this ComplexSIMD instance.

        Args:
            other: Imaginary unit (1j) to subtract from this complex number.

        Returns:
            ComplexSIMD instance where each lane's imaginary part is decreased by 1.
            If self = a + bj, then result = a + (b-1)j.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z = ComplexSIMD[cf64, 1](3.0, 2.0)  # 3 + 2j
            var result = z - `1j`  # 3 + 1j
            print(result)  # (3 + 1j)
            ```
        """
        return Self(self.re, self.im - Self._broadcast(1))

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

    fn __isub__(mut self, other: SIMD[*_, size = Self.width]):
        """
        In-place subtraction of a SIMD vector from the real part of this ComplexSIMD instance.

        Args:
            other: SIMD vector to subtract from the real component.
        """
        self.re -= other.cast[Self.dtype]()

    fn __isub__(mut self, other: ImaginaryUnit):
        """
        In-place subtraction of the imaginary unit 1j from this ComplexSIMD instance.

        Args:
            other: Imaginary unit (1j) to subtract from this complex number.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z = ComplexSIMD[cf64, 1](3.0, 2.0)  # 3 + 2j
            z -= `1j`  # Now z = 3 + 1j
            print(z)  # (3 + 1j)
            ```
        """
        self.im -= Self._broadcast(1)

    fn __rsub__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Returns the difference of a scalar and this ComplexSIMD instance, subtracting from the real part.

        Args:
            other: Scalar value to subtract from the real component.

        Returns:
            ComplexSIMD instance where each lane's real part is (scalar - self.re).
        """
        return Self(Self._broadcast(other) - self.re, -self.im)

    fn __rsub__(self, other: SIMD[*_, size = Self.width]) -> Self:
        """
        Returns the difference of a SIMD vector and this ComplexSIMD instance, subtracting from the real part.

        Args:
            other: SIMD vector to subtract from the real component.

        Returns:
            ComplexSIMD instance where each lane's real part is (SIMD lane - self.re).
        """
        var other_casted = other.cast[Self.dtype]()
        return Self(other_casted - self.re, -self.im)

    fn __rsub__(self, other: ImaginaryUnit) -> Self:
        """
        Returns the difference of the imaginary unit 1j and this ComplexSIMD instance.

        Args:
            other: Imaginary unit (1j) from which this complex number is subtracted.

        Returns:
            ComplexSIMD instance equal to 1j - self.
            If self = a + bj, then result = -a + (1-b)j.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z = ComplexSIMD[cf64, 1](3.0, 2.0)  # 3 + 2j
            var result = `1j` - z  # -3 + (-1)j = -3 - 1j
            print(result)  # (-3 - 1j)
            ```
        """
        return Self(-self.re, Self._broadcast(1) - self.im)

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

    fn __mul__(self, other: SIMD[*_, size = Self.width]) -> Self:
        """
        Returns the product of this ComplexSIMD instance and a SIMD vector.

        Args:
            other: SIMD vector to multiply with both real and imaginary parts.

        Returns:
            ComplexSIMD instance where each lane is scaled by the corresponding lane in the SIMD vector.
        """
        var other_casted = other.cast[Self.dtype]()
        return Self(self.re * other_casted, self.im * other_casted)

    fn __mul__(self, other: ImaginaryUnit) -> Self:
        """
        Returns the product of this ComplexSIMD instance and the imaginary unit 1j.

        Args:
            other: Imaginary unit (1j) to multiply with this complex number.

        Returns:
            ComplexSIMD instance where each lane is multiplied by 1j.
            If self = a + bj, then result = (a + bj) * 1j = -b + aj.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z = ComplexSIMD[cf64, 1](3.0, 2.0)  # 3 + 2j
            var result = z * `1j`  # -2 + 3j
            print(result)  # (-2 + 3j)
            ```
        """
        return Self(-self.im, self.re)

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

    fn __imul__(mut self, other: SIMD[*_, size = Self.width]):
        """
        In-place multiplication of this ComplexSIMD instance by a SIMD vector.

        Args:
            other: SIMD vector to multiply with both real and imaginary parts.
        """
        var other_casted = other.cast[Self.dtype]()
        self.re *= other_casted
        self.im *= other_casted

    fn __imul__(mut self, other: ImaginaryUnit):
        """
        In-place multiplication of this ComplexSIMD instance by the imaginary unit 1j.

        Args:
            other: Imaginary unit (1j) to multiply with this complex number.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z = ComplexSIMD[cf64, 1](3.0, 2.0)  # 3 + 2j
            z *= `1j`  # Now z = -2 + 3j
            print(z)  # (-2 + 3j)
            ```
        """
        var new_re = -self.im
        self.im = self.re
        self.re = new_re

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

    fn __rmul__(self, other: SIMD[*_, size = Self.width]) -> Self:
        """
        Returns the product of a SIMD vector and this ComplexSIMD instance.

        Args:
            other: SIMD vector to multiply with both real and imaginary parts.

        Returns:
            ComplexSIMD instance where each lane is scaled by the corresponding lane in the SIMD vector.
        """
        var other_casted = other.cast[Self.dtype]()
        return Self(other_casted * self.re, other_casted * self.im)

    fn __rmul__(self, other: ImaginaryUnit) -> Self:
        """
        Returns the product of the imaginary unit 1j and this ComplexSIMD instance.

        Args:
            other: Imaginary unit (1j) to multiply with this complex number.

        Returns:
            ComplexSIMD instance where each lane is multiplied by 1j.
            If self = a + bj, then result = 1j * (a + bj) = -b + aj.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z = ComplexSIMD[cf64, 1](3.0, 2.0)  # 3 + 2j
            var result = `1j` * z  # -2 + 3j
            print(result)  # (-2 + 3j)
            ```
        """
        return Self(-self.im, self.re)

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

    fn __truediv__(self, other: SIMD[*_, size = Self.width]) -> Self:
        """
        Performs element-wise division of this ComplexSIMD instance by a SIMD vector.

        Args:
            other: SIMD vector to divide both real and imaginary parts by.

        Returns:
            ComplexSIMD instance where each lane is divided by the corresponding lane in the SIMD vector.
        """
        var other_casted = other.cast[Self.dtype]()
        return Self(self.re / other_casted, self.im / other_casted)

    fn __truediv__(self, other: ImaginaryUnit) -> Self:
        """
        Performs division of this ComplexSIMD instance by the imaginary unit 1j.

        Args:
            other: Imaginary unit (1j) to divide this complex number by.

        Returns:
            ComplexSIMD instance where each lane is divided by 1j.
            If self = a + bj, then result = (a + bj) / 1j = b - aj.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z = ComplexSIMD[cf64, 1](3.0, 2.0)  # 3 + 2j
            var result = z / `1j`  # 2 - 3j
            print(result)  # (2 - 3j)
            ```
        """
        return Self(self.im, -self.re)

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

    fn __itruediv__(mut self, other: SIMD[*_, size = Self.width]):
        """
        Performs in-place element-wise division of this ComplexSIMD instance by a SIMD vector.

        Args:
            other: SIMD vector to divide both real and imaginary parts by.
        """
        var other_casted = other.cast[Self.dtype]()
        self.re /= other_casted
        self.im /= other_casted

    fn __itruediv__(mut self, other: ImaginaryUnit):
        """
        Performs in-place division of this ComplexSIMD instance by the imaginary unit 1j.

        Args:
            other: Imaginary unit (1j) to divide this complex number by.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z = ComplexSIMD[cf64, 1](3.0, 2.0)  # 3 + 2j
            z /= `1j`  # Now z = 2 - 3j
            print(z)  # (2 - 3j)
            ```
        """
        var new_re = self.im
        self.im = -self.re
        self.re = new_re

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

    fn __rtruediv__(self, other: SIMD[*_, size = Self.width]) -> Self:
        """
        Performs element-wise division of a SIMD vector by this ComplexSIMD instance.

        Args:
            other: SIMD vector to be divided by this ComplexSIMD instance.

        Returns:
            ComplexSIMD instance where each lane is the result of dividing the corresponding lane in the SIMD vector by the corresponding lane in this ComplexSIMD:
            other[i] / (a + bi) = [other[i] * a / (a^2 + b^2)] + [-other[i] * b / (a^2 + b^2)]i
            where a, b are self.re, self.im.
        """
        var denom = self.re * self.re + self.im * self.im
        var other_casted = other.cast[Self.dtype]()
        return Self(
            (other_casted * self.re) / denom,
            (-other_casted * self.im) / denom,
        )

    fn __rtruediv__(self, other: ImaginaryUnit) -> Self:
        """
        Performs division of the imaginary unit 1j by this ComplexSIMD instance.

        Args:
            other: Imaginary unit (1j) to be divided by this ComplexSIMD instance.

        Returns:
            ComplexSIMD instance where each lane is the result of dividing 1j by the corresponding lane.
            If self = a + bj, then result = 1j / (a + bj) = [b / (a² + b²)] + [-a / (a² + b²)]j.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z = ComplexSIMD[cf64, 1](3.0, 4.0)  # 3 + 4j
            var result = `1j` / z  # 1j / (3 + 4j) = 0.16 - 0.12j
            print(result)  # (0.16 - 0.12j)
            ```
        """
        var denom = self.re * self.re + self.im * self.im
        return Self(
            self.im / denom,
            -self.re / denom,
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

    fn __invert__(
        self,
    ) -> Self where cdtype == ComplexDType.bool or cdtype.is_integral():
        """
        Element-wise logical NOT operation on this ComplexSIMD instance.

        Returns:
            ComplexSIMD instance where each lane is the logical NOT of the corresponding lane.
        """
        return Self(~self.re, ~self.im)

    # --- Comparison operators ---
    fn __and__(
        self, other: Self
    ) -> Self where cdtype == ComplexDType.bool or cdtype.is_integral():
        """
        Element-wise logical AND operation between two ComplexSIMD instances.

        Args:
            other: Another ComplexSIMD instance.

        Returns:
            True if both the real and imaginary parts are non-zero for all lanes, otherwise False.
        """
        return Self(self.re & other.re, self.im & other.im)

    fn __or__(
        self, other: Self
    ) -> Self where cdtype == ComplexDType.bool or cdtype.is_integral():
        """
        Element-wise logical OR operation between two ComplexSIMD instances.

        Args:
            other: Another ComplexSIMD instance.

        Returns:
            True if either the real or imaginary part is non-zero for any lane, otherwise False.
        """
        return Self(self.re | other.re, self.im | other.im)

    fn __xor__(
        self, other: Self
    ) -> Self where cdtype == ComplexDType.bool or cdtype.is_integral():
        """
        Element-wise logical XOR operation between two ComplexSIMD instances.

        Args:
            other: Another ComplexSIMD instance.

        Returns:
            True if exactly one of the real or imaginary parts is non-zero for any lane, otherwise False.
        """
        return Self(self.re ^ other.re, self.im ^ other.im)

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

    fn __eq__(self, other: ImaginaryUnit) -> Bool:
        """
        Checks if this ComplexSIMD instance is equal to the imaginary unit 1j.

        Args:
            other: Imaginary unit (1j) to compare with this ComplexSIMD instance.

        Returns:
            True if the real part is 0 and the imaginary part is 1 for all lanes, otherwise False.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z1 = ComplexSIMD[cf64, 1](0.0, 1.0)  # 0 + 1j
            var z2 = ComplexSIMD[cf64, 1](1.0, 1.0)  # 1 + 1j
            print(z1 == `1j`)  # True
            print(z2 == `1j`)  # False
            ```
        """
        return (self.re == Self._broadcast(0)) and (
            self.im == Self._broadcast(1)
        )

    fn __ne__(self, other: Self) -> Bool:
        """
        Checks if two ComplexSIMD instances are not equal.

        Returns:
            True if either the real or imaginary parts differ for any lane, otherwise False.
        """
        return ~(self == other)

    fn __ne__(self, other: ImaginaryUnit) -> Bool:
        """
        Checks if this ComplexSIMD instance is not equal to the imaginary unit 1j.

        Args:
            other: Imaginary unit (1j) to compare with this ComplexSIMD instance.

        Returns:
            True if either the real part is not 0 or the imaginary part is not 1 for any lane, otherwise False.

        Examples:
            ```mojo
            from numojo.prelude import *

            var z1 = ComplexSIMD[cf64, 1](0.0, 1.0)  # 0 + 1j
            var z2 = ComplexSIMD[cf64, 1](1.0, 1.0)  # 1 + 1j
            print(z1 != `1j`)  # False
            print(z2 != `1j`)  # True
            ```
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
    fn __getitem__(self, idx: Int) raises -> ComplexScalar[Self.cdtype]:
        """
        Returns the complex number at the specified lane index.

        Args:
            idx: SIMD lane index (0 to width-1).

        Returns:
            ComplexScalar containing the complex number at that lane index.

        Raises:
            Error if lane index is out of range for the SIMD width.

        Example:
            ```mojo
            from numojo.prelude import *
            var c_simd = ComplexSIMD[cf32, 2](SIMD[f32, 2](1, 2), SIMD[f32, 2](3, 4))
            var c0 = c_simd[0]  # 1 + 3j
            var c1 = c_simd[1]  # 2 + 4j
            ```
        """
        if idx < 0 or idx >= Self.width:
            raise Error("Lane index out of range for SIMD width")
        return ComplexScalar[Self.cdtype](self.re[idx], self.im[idx])

    fn __setitem__(
        mut self, idx: Int, value: ComplexScalar[Self.cdtype]
    ) raises:
        """
        Sets the complex scalar at the specified lane index.

        Args:
            idx: SIMD lane index (0 to width-1).
            value: ComplexScalar whose values will be assigned.

        Raises:
            Error if lane index is out of range for the SIMD width.

        Example:
            ```mojo
            from numojo.prelude import *
            var c_simd = nm.ComplexSIMD[cf32, 2](SIMD[f32, 2](1, 2), SIMD[f32, 2](3, 4)) # [(1 + 3j), (2 + 4j)]
            c_simd[0] = nm.CScalar[cf32](5, 6)
            print(c_simd) # [(1 + 3j), (2 + 4j)] becomes [(5 + 6j), (2 + 4j)]
            ```
        """
        if idx < 0 or idx >= Self.width:
            raise Error("Lane index out of range for SIMD width")
        self.re[idx] = value.re
        self.im[idx] = value.im

    fn item[name: String](self, idx: Int) raises -> Scalar[Self.dtype]:
        """
        Returns the scalar value for the specified lane index and component.

        Parameters:
            name: Name of the component ('re' or 'im').

        Args:
            idx: Lane index to retrieve.

        Returns:
            Scalar value of the specified component at the given lane index.

        Raises:
            - Error if the component name is invalid.
            - Error if lane index is out of range for the SIMD width.

        Example:
            ```mojo
            from numojo.prelude import *
            var c_simd = nm.ComplexSIMD[cf32, 2](SIMD[f32, 2](1, 2), SIMD[f32, 2](3, 4)) # [(1 + 3j), (2 + 4j)]
            var re0 = c_simd.item["re"](0)  # 1.0
            var im1 = c_simd.item["im"](1)  # 4.0
            ```
        """
        if idx < 0 or idx >= Self.width:
            raise Error("Lane index out of range for SIMD width")

        @parameter
        if name == "re":
            return self.re[idx]
        elif name == "im":
            return self.im[idx]
        else:
            raise Error("Invalid component name: {}".format(name))

    fn itemset[
        name: String
    ](mut self, idx: Int, val: Scalar[Self.dtype]) raises:
        """
        Sets the scalar value for the specified lane index and component.

        Parameters:
            name: Name of the component ('re' or 'im').

        Args:
            idx: Lane index to set.
            val: Scalar value to assign to the specified component.

        Raises:
            - Error if the component name is invalid.
            - Error if lane index is out of range for the SIMD width.

        Example:
            ```mojo
            from numojo.prelude import *
            var c_simd = nm.ComplexSIMD[cf32, 2](SIMD[f32, 2](1, 2), SIMD[f32, 2](3, 4)) # [(1 + 3j), (2 + 4j)]
            c_simd.itemset["re"](0, 5.0)  # Now first complex number is (5 + 3j)
            c_simd.itemset["im"](1, 6.0)  # Now second complex number is (2 + 6j)
            ```
        """
        if idx < 0 or idx >= Self.width:
            raise Error("Lane index out of range for SIMD width")

        @parameter
        if name == "re":
            self.re[idx] = val
        elif name == "im":
            self.im[idx] = val
        else:
            raise Error("Invalid component name: {}".format(name))

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


@register_passable("trivial")
struct ImaginaryUnit(Boolable, Stringable, Writable):
    """
    Constant representing the imaginary unit complex number 0 + 1j.

    The ImaginaryUnit struct provides a convenient way to work with the imaginary unit
    in complex arithmetic operations. It supports arithmetic operations with SIMD vectors,
    scalars, and other complex numbers, enabling Python-like syntax for complex number creation.

    Examples:
        ```mojo
        from numojo.prelude import *

        # Create complex numbers using the imaginary unit
        var z1 = 3 + 4 * `1j`  # 3 + 4j
        var z2 = `1j` * 2      # 0 + 2j
        var z3 = 5.0 + `1j`    # 5 + 1j

        # Powers of the imaginary unit
        print(`1j` ** 0)  # 1 + 0j
        print(`1j` ** 1)  # 0 + 1j
        print(`1j` ** 2)  # -1 + 0j
        print(`1j` ** 3)  # 0 - 1j

        # SIMD complex vectors
        var c4 = SIMD[f32, 4](1.0) + `1j` * SIMD[f32, 4](2.0)  # ComplexSIMD[cf32, 4]
        var c5 = SIMD[f64, 2](3.0, 4.0) + `1j`                 # ComplexSIMD[cf64, 2]
        var d = SIMD[f32, 2](1) + SIMD[f32, 2](2) * `1j`         # creates [( 1 + 2 j) (1 + 2 j)]

        # Mathematical properties
        var c6 = `1j` * `1j`                     # -1 (Scalar[f64])
        var c7 = `1j` ** 3                       # (0 - 1j) (ComplexScalar[cf64])
        var c8 = (1 + `1j`) / `1j`               # (1 - 1j) (ComplexScalar[cf64])
        ```
    """

    fn __init__(out self):
        """
        Constructor for ImaginaryUnit.

        Creates an instance representing the imaginary unit 1j.
        """
        pass

    fn conj(self) -> ComplexSIMD[cf64, 1]:
        """
        Returns the complex conjugate of the imaginary unit.

        Returns:
            ComplexSIMD representing -1j (the conjugate of 1j).

        Examples:
            ```mojo
            from numojo.prelude import *

            var conj_i = `1j`.conj()  # -1j
            print(conj_i)  # (0 - 1j)
            ```
        """
        return -self

    # --- Arithmetic operators with SIMD and Scalar types ---
    # Addition: 1j + SIMD -> ComplexSIMD
    fn __add__[
        dtype: DType, width: Int
    ](self, other: SIMD[dtype, width]) -> ComplexSIMD[
        ComplexDType(mlir_value=dtype._mlir_value), width
    ]:
        """
        Returns the sum of the imaginary unit 1j and a SIMD vector.

        Args:
            other: SIMD vector to add to the imaginary unit.

        Returns:
            ComplexSIMD where real part equals the SIMD vector and imaginary part is 1.

        Examples:
            ```mojo
            from numojo.prelude import *

            var vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
            var result = `1j` + vec  # [1+1j, 2+1j, 3+1j, 4+1j]
            ```
        """
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), width](
            other, SIMD[dtype, width](1)
        )

    # Addition: 1j + Scalar -> ComplexScalar
    fn __add__[
        dtype: DType
    ](self, other: Scalar[dtype]) -> ComplexScalar[
        ComplexDType(mlir_value=dtype._mlir_value)
    ]:
        """
        Returns the sum of the imaginary unit 1j and a scalar.

        Args:
            other: Scalar to add to the imaginary unit.

        Returns:
            ComplexScalar with real part equal to the scalar and imaginary part 1.

        Examples:
            ```mojo
            from numojo.prelude import *

            var result = `1j` + 3.5  # 3.5 + 1j
            ```
        """
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), 1](
            other, 1
        )

    fn __add__(self, other: Int) -> ComplexScalar[ComplexDType.int]:
        """
        Returns the sum of the imaginary unit 1j and an integer.

        Args:
            other: Integer to add to the imaginary unit.

        Returns:
            ComplexScalar with real part equal to the integer and imaginary part 1.

        Examples:
            ```mojo
            from numojo.prelude import *

            var result = `1j` + 5  # 5 + 1j
            ```
        """
        return ComplexSIMD[ComplexDType.int, 1](other, 1)

    # SIMD + 1j -> ComplexSIMD
    fn __radd__[
        dtype: DType, width: Int
    ](self, other: SIMD[dtype, width]) -> ComplexSIMD[
        ComplexDType(mlir_value=dtype._mlir_value), width
    ]:
        """Returns the sum of a SIMD vector and the imaginary unit."""
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), width](
            other, SIMD[dtype, width](1)
        )

    # Addition: Scalar + 1j -> ComplexScalar
    fn __radd__[
        dtype: DType
    ](self, other: Scalar[dtype]) -> ComplexScalar[
        ComplexDType(mlir_value=dtype._mlir_value)
    ]:
        """Returns the sum of a scalar and the imaginary unit."""
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), 1](
            other, 1
        )

    # Addition: Int + 1j -> ComplexScalar
    fn __radd__(self, other: Int) -> ComplexScalar[ComplexDType.int]:
        """Returns the sum of an integer and the imaginary unit."""
        return ComplexSIMD[ComplexDType.int, 1](other, 1)

    # Subtraction: 1j - SIMD -> ComplexSIMD
    fn __sub__[
        dtype: DType, width: Int
    ](self, other: SIMD[dtype, width]) -> ComplexSIMD[
        ComplexDType(mlir_value=dtype._mlir_value), width
    ]:
        """Returns the difference of the imaginary unit and a SIMD vector."""
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), width](
            -other, SIMD[dtype, width](1)
        )

    # Subtraction: 1j - Scalar -> ComplexScalar
    fn __sub__[
        dtype: DType
    ](self, other: Scalar[dtype]) -> ComplexScalar[
        ComplexDType(mlir_value=dtype._mlir_value)
    ]:
        """Returns the difference of the imaginary unit and a scalar."""
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), 1](
            -other, 1
        )

    fn __sub__(self, other: Int) -> ComplexScalar[ComplexDType.int]:
        """Returns the difference of the imaginary unit and an integer."""
        return ComplexSIMD[ComplexDType.int, 1](-other, 1)

    # Subtraction: SIMD - 1j -> ComplexSIMD
    fn __rsub__[
        dtype: DType, width: Int
    ](self, other: SIMD[dtype, width]) -> ComplexSIMD[
        ComplexDType(mlir_value=dtype._mlir_value), width
    ]:
        """Returns the difference of a SIMD vector and the imaginary unit."""
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), width](
            other, SIMD[dtype, width](-1)
        )

    # Subtraction: Scalar - 1j -> ComplexScalar
    fn __rsub__[
        dtype: DType
    ](self, other: Scalar[dtype]) -> ComplexScalar[
        ComplexDType(mlir_value=dtype._mlir_value)
    ]:
        """Returns the difference of a scalar and the imaginary unit."""
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), 1](
            other, -1
        )

    # Subtraction: Int - 1j -> ComplexScalar
    fn __rsub__(self, other: Int) -> ComplexScalar[ComplexDType.int]:
        """Returns the difference of an integer and the imaginary unit."""
        return ComplexSIMD[ComplexDType.int, 1](other, -1)

    # Multiplication: 1j * SIMD -> ComplexSIMD
    fn __mul__[
        dtype: DType, width: Int
    ](self, other: SIMD[dtype, width]) -> ComplexSIMD[
        ComplexDType(mlir_value=dtype._mlir_value), width
    ]:
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), width](
            SIMD[dtype, width](0), other
        )

    # Multiplication: 1j * Scalar -> ComplexScalar
    fn __mul__[
        dtype: DType
    ](self, other: Scalar[dtype]) -> ComplexScalar[
        ComplexDType(mlir_value=dtype._mlir_value)
    ]:
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), 1](
            0, other
        )

    # Multiplication: 1j * Int -> ComplexScalar
    fn __mul__(self, other: Int) -> ComplexScalar[ComplexDType.int]:
        return ComplexSIMD[ComplexDType.int, 1](0, other)

    fn __rmul__[
        dtype: DType,
        width: Int,
    ](self, other: SIMD[dtype, width]) -> ComplexSIMD[
        ComplexDType(mlir_value=dtype._mlir_value), width
    ]:
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), width](
            SIMD[dtype, width](0), other
        )

    fn __rmul__[
        dtype: DType
    ](self, other: Scalar[dtype]) -> ComplexScalar[
        ComplexDType(mlir_value=dtype._mlir_value)
    ]:
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), 1](
            0, other
        )

    # Multiplication: Scalar * 1j -> ComplexScalar
    fn __rmul__(self, other: Int) -> ComplexScalar[ComplexDType.int]:
        return ComplexSIMD[ComplexDType.int, 1](0, other)

    # Division: 1j / SIMD -> ComplexSIMD
    fn __truediv__[
        dtype: DType, width: Int
    ](self, other: SIMD[dtype, width]) -> ComplexSIMD[
        ComplexDType(mlir_value=dtype._mlir_value), width
    ]:
        """Returns the division of the imaginary unit by a SIMD vector."""
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), width](
            SIMD[dtype, width](0), 1 / other
        )

    # Division: 1j / Scalar -> ComplexScalar
    fn __truediv__[
        dtype: DType
    ](self, other: Scalar[dtype]) -> ComplexScalar[
        ComplexDType(mlir_value=dtype._mlir_value)
    ]:
        """Returns the division of the imaginary unit by a scalar."""
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), 1](
            0, 1 / other
        )

    # Division: SIMD / 1j -> ComplexSIMD
    fn __rtruediv__[
        dtype: DType, width: Int
    ](self, other: SIMD[dtype, width]) -> ComplexSIMD[
        ComplexDType(mlir_value=dtype._mlir_value), width
    ]:
        """
        Returns the division of a SIMD vector by the imaginary unit.

        Args:
            other: SIMD vector to be divided by the imaginary unit.

        Returns:
            ComplexSIMD representing (0 - other j).
        """
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), width](
            SIMD[dtype, width](0), -other
        )

    # Division: Scalar / 1j -> ComplexScalar
    fn __rtruediv__[
        dtype: DType
    ](self, other: Scalar[dtype]) -> ComplexScalar[
        ComplexDType(mlir_value=dtype._mlir_value)
    ]:
        """
        Returns the division of a scalar by the imaginary unit.

        Args:
            other: Scalar to be divided by the imaginary unit.

        Returns:
            ComplexScalar representing (0 - other j).
        """
        return ComplexSIMD[ComplexDType(mlir_value=dtype._mlir_value), 1](
            0, -other
        )

    # Division: Int / 1j -> ComplexScalar
    fn __rtruediv__(self, other: Int) -> ComplexScalar[ComplexDType.int]:
        """
        Returns the division of an integer by the imaginary unit.

        Args:
            other: Integer to be divided by the imaginary unit.

        Returns:
            ComplexScalar representing (0 - other j).
        """
        return ComplexSIMD[ComplexDType.int, 1](0, -other)

    # Self-operations: 1j with 1j
    fn __mul__(self, other: ImaginaryUnit) -> Scalar[DType.float64]:
        """
        Returns the product of the imaginary unit with itself: 1j * 1j = -1.

        Args:
            other: Another imaginary unit to multiply with.

        Returns:
            Scalar value -1, since (1j) * (1j) = -1.

        Examples:
            ```mojo
            from numojo.prelude import *

            var result = `1j` * `1j`  # -1
            print(result)  # -1
            ```
        """
        return -1

    fn __add__(self, other: ImaginaryUnit) -> ComplexScalar[cf64]:
        """
        Returns the sum of the imaginary unit with itself: 1j + 1j = 2j.

        Args:
            other: Another imaginary unit to add.

        Returns:
            ComplexScalar representing 2j.

        Examples:
            ```mojo
            from numojo.prelude import *

            var result = `1j` + `1j`  # 2j
            print(result)  # (0 + 2j)
            ```
        """
        return ComplexSIMD[cf64, 1](0, 2)

    fn __sub__(self, other: ImaginaryUnit) -> Scalar[DType.float64]:
        """
        Returns the difference of the imaginary unit with itself: 1j - 1j = 0.

        Args:
            other: Another imaginary unit to subtract.

        Returns:
            Scalar value 0.

        Examples:
            ```mojo
            from numojo.prelude import *

            var result = `1j` - `1j`  # 0
            print(result)  # 0
            ```
        """
        return 0

    fn __truediv__(self, other: ImaginaryUnit) -> Scalar[DType.float64]:
        """
        Returns the division of the imaginary unit by itself: 1j / 1j = 1.

        Args:
            other: Another imaginary unit to divide by.

        Returns:
            Scalar value 1.

        Examples:
            ```mojo
            from numojo.prelude import *

            var result = `1j` / `1j`  # 1
            print(result)  # 1
            ```
        """
        return 1

    fn __pow__(self, exponent: Int) -> ComplexScalar[cf64]:
        """
        Returns the imaginary unit raised to an integer power.

        The powers of 1j cycle with period 4:
        - 1j^0 = 1
        - 1j^1 = 1j
        - 1j^2 = -1
        - 1j^3 = -1j
        - 1j^4 = 1 (cycle repeats)

        Args:
            exponent: Integer exponent.

        Returns:
            ComplexScalar representing (1j) ** exponent.

        Examples:
            ```mojo
            from numojo.prelude import *

            print(`1j` ** 0)  # (1 + 0j)
            print(`1j` ** 1)  # (0 + 1j)
            print(`1j` ** 2)  # (-1 + 0j)
            print(`1j` ** 3)  # (0 - 1j)
            print(`1j` ** 4)  # (1 + 0j)
            ```
        """
        var remainder = exponent % 4
        if remainder == 0:
            return ComplexSIMD[cf64, 1](1, 0)
        elif remainder == 1:
            return ComplexSIMD[cf64, 1](0, 1)
        elif remainder == 2:
            return ComplexSIMD[cf64, 1](-1, 0)
        else:
            return ComplexSIMD[cf64, 1](0, -1)

    fn __neg__(self) -> ComplexScalar[cf64]:
        """
        Returns the negation of the imaginary unit: -1j.

        Returns:
            ComplexScalar representing -1j.

        Examples:
            ```mojo
            from numojo.prelude import *

            var result = -`1j`  # -1j
            print(result)  # (0 - 1j)
            ```
        """
        return ComplexSIMD[cf64, 1](0, -1)

    fn __str__(self) -> String:
        """
        Returns the string representation of the imaginary unit.

        Returns:
            String representing the imaginary unit as "(0 + 1 j)".
        """
        return "(0 + 1 j)"

    fn write_to[W: Writer](self, mut writer: W):
        """
        Writes the string representation of the imaginary unit to a writer.

        Args:
            writer: Writer instance to write the string representation to.
        """
        writer.write("(0 + 1 j)")

    fn __bool__(self) -> Bool:
        """
        Returns the boolean value of the imaginary unit.

        Returns:
            Always True, since the imaginary unit 1j is non-zero.

        Examples:
            ```mojo
            from numojo.prelude import *

            if `1j`:
                print("Imaginary unit is truthy")  # This will execute
            ```
        """
        return True
