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

from math import sqrt

from numojo.core.complex.complex_dtype import ComplexDType

# ComplexScalar alias is for internal purposes
alias ComplexScalar[cdtype: ComplexDType] = ComplexSIMD[cdtype, width=1]
# CScalar is short alias for ComplexScalar for user convenience
alias CScalar[cdtype: ComplexDType] = ComplexSIMD[cdtype, width=1]
# CSIMD is short alias for ComplexSIMD with width=1 for user convenience
alias CSIMD[cdtype: ComplexDType] = ComplexSIMD[cdtype, width=1]


@register_passable("trivial")
struct ComplexSIMD[cdtype: ComplexDType, width: Int = 1](Stringable, Writable):
    """
    A SIMD-enabled complex number type that supports vectorized operations.

    Parameters:
        cdtype: The complex data type (like cf32 or cf64) that determines precision.
        width: The SIMD vector width, defaulting to 1 for scalar operations.

    The struct contains two SIMD vectors - one for the real part and one for the
    imaginary part. This allows complex arithmetic to be performed efficiently using
    SIMD operations. When width=1 it acts as a regular complex scalar type.

    Example:
    ```mojo
    import numojo as nm
    var A = nm.ComplexSIMD[nm.cf32](1.0, 2.0)
    var B = nm.ComplexSIMD[nm.cf32](3.0, 4.0)
    var C = A + B
    print(C) # Output: (4.0 + 6.0 j)

    var A1 = nm.ComplexSIMD[nm.cf32, 2](SIMD[nm.f32](1.0, 1.0), SIMD[nm.f32](2.0, 2.0))
    print(A1) # Output: ([1.0, 1.0] + [2.0, 2.0] j)
    ```
    """

    # FIELDS
    alias dtype: DType = cdtype._dtype  # the corresponding DType
    # The underlying data real and imaginary parts of the complex number.
    var re: SIMD[Self.dtype, width]
    var im: SIMD[Self.dtype, width]

    @always_inline
    fn __init__(out self, other: Self):
        """
        Initializes a ComplexSIMD instance by copying another instance.

        Arguments:
            other: Another ComplexSIMD instance to copy from.
        """
        self = other

    @always_inline
    fn __init__(
        out self,
        re: SIMD[Self.dtype, Self.width],
        im: SIMD[Self.dtype, Self.width],
    ):
        """
        Initializes a ComplexSIMD instance with specified real and imaginary parts.

        Arguments:
            re: The real part of the complex number.
            im: The imaginary part of the complex number.

        Example:
        ```mojo
        import numojo as nm
        var A = nm.ComplexSIMD[nm.cf32](1.0, 2.0)
        var B = nm.ComplexSIMD[nm.cf32](3.0, 4.0)
        var C = A + B
        print(C)
        ```
        """
        self.re = re
        self.im = im

    @always_inline
    fn __init__(out self, val: SIMD[Self.dtype, Self.width]):
        """
        Initializes a ComplexSIMD instance with specified real and imaginary parts.

        Arguments:
            re: The real part of the complex number.
            im: The imaginary part of the complex number.
        """
        self.re = val
        self.im = val

    fn __add__(self, other: Self) -> Self:
        """
        Adds two ComplexSIMD instances.

        Arguments:
            other: The ComplexSIMD instance to add.

        Returns:
            Self: A new ComplexSIMD instance representing the sum.
        """
        return Self(self.re + other.re, self.im + other.im)

    fn __iadd__(mut self, other: Self):
        """
        Performs in-place addition of another ComplexSIMD instance.

        Arguments:
            other: The ComplexSIMD instance to add.
        """
        self.re += other.re
        self.im += other.im

    fn __sub__(self, other: Self) -> Self:
        """
        Subtracts another ComplexSIMD instance from this one.

        Arguments:
            other: The ComplexSIMD instance to subtract.

        Returns:
            Self: A new ComplexSIMD instance representing the difference.
        """
        return Self(self.re - other.re, self.im - other.im)

    fn __isub__(mut self, other: Self):
        """
        Performs in-place subtraction of another ComplexSIMD instance.

        Arguments:
            other: The ComplexSIMD instance to subtract.
        """
        self.re -= other.re
        self.im -= other.im

    fn __mul__(self, other: Self) -> Self:
        """
        Multiplies two ComplexSIMD instances.

        Arguments:
            other: The ComplexSIMD instance to multiply with.

        Returns:
            Self: A new ComplexSIMD instance representing the product.
        """
        return Self(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    fn __imul__(mut self, other: Self):
        """
        Performs in-place multiplication with another ComplexSIMD instance.

        Arguments:
            other: The ComplexSIMD instance to multiply with.
        """
        var re = self.re * other.re - self.im * other.im
        self.im = self.re * other.im + self.im * other.re
        self.re = re

    fn __truediv__(self, other: Self) -> Self:
        """
        Divides this ComplexSIMD instance by another.

        Arguments:
            other: The ComplexSIMD instance to divide by.

        Returns:
            Self: A new ComplexSIMD instance representing the quotient.
        """
        var denom = other.re * other.re + other.im * other.im
        return Self(
            (self.re * other.re + self.im * other.im) / denom,
            (self.im * other.re - self.re * other.im) / denom,
        )

    fn __itruediv__(mut self, other: Self):
        """
        Performs in-place division by another ComplexSIMD instance.

        Arguments:
            other: The ComplexSIMD instance to divide by.
        """
        var denom = other.re * other.re + other.im * other.im
        var re = (self.re * other.re + self.im * other.im) / denom
        self.im = (self.im * other.re - self.re * other.im) / denom
        self.re = re

    fn __pow__(self, other: Self) -> Self:
        """
        Raises this ComplexSIMD instance to the power of another.

        Arguments:
            other: The ComplexSIMD instance to raise to the power of.

        Returns:
            Self: A new ComplexSIMD instance representing the result.
        """
        return Self(self.re**other.re, self.im**other.im)

    fn __pow__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Raises this ComplexSIMD instance to the power of a scalar.

        Arguments:
            other: The scalar to raise to the power of.

        Returns:
            Self: A new ComplexSIMD instance representing the result.
        """
        return Self(self.re**other, self.im**other)

    fn __pos__(self) -> Self:
        """
        Returns the ComplexSIMD instance itself.

        Returns:
            Self: The ComplexSIMD instance itself.
        """
        return self

    fn __neg__(self) -> Self:
        """
        Negates the ComplexSIMD instance.

        Returns:
            Self: The negated ComplexSIMD instance.
        """
        return self * Self(-1, -1)

    fn __eq__(self, other: Self) -> Bool:
        """
        Checks if two ComplexSIMD instances are equal.

        Arguments:
            self: The first ComplexSIMD instance.
            other: The second ComplexSIMD instance to compare with.

        Returns:
            Bool: True if the instances are equal, False otherwise.
        """
        return (self.re == other.re).reduce_and() and (
            self.im == other.im
        ).reduce_add()

    fn __ne__(self, other: Self) -> Bool:
        """
        Checks if two ComplexSIMD instances are not equal.

        Arguments:
            self: The first ComplexSIMD instance.
            other: The second ComplexSIMD instance to compare with.

        Returns:
            Bool: True if the instances are not equal, False otherwise.
        """
        return (self.re != other.re).reduce_or() or (
            self.im != other.im
        ).reduce_or()

    fn __str__(self) -> String:
        """
        Returns a string representation of the ComplexSIMD instance.

        Returns:
            String: The string representation of the ComplexSIMD instance.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Writes the ComplexSIMD instance to a writer.

        Arguments:
            self: The ComplexSIMD instance to write.
            writer: The writer to write to.
        """
        try:
            writer.write(String("({} + {} j)").format(self.re, self.im))
        except e:
            writer.write("Cannot convert ComplexSIMD to string")

    fn __repr__(self) raises -> String:
        """
        Returns a string representation of the ComplexSIMD instance.

        Returns:
            String: The string representation of the ComplexSIMD instance.
        """
        return String("ComplexSIMD[{}]({}, {})").format(
            String(Self.dtype), self.re, self.im
        )

    fn __getitem__(self, idx: Int) raises -> SIMD[Self.dtype, Self.width]:
        """
        Gets the real or imaginary part of the ComplexSIMD instance.

        Arguments:
            self: The ComplexSIMD instance.
            idx: The index to access (0 for real, 1 for imaginary).

        Returns:
            SIMD[Self.dtype, 1]: The requested part of the ComplexSIMD instance.
        """
        if idx == 0:
            return self.re
        elif idx == 1:
            return self.im
        else:
            raise Error("Index out of range")

    fn __setitem__(
        mut self, idx: Int, value: SIMD[Self.dtype, Self.width]
    ) raises:
        """
        Sets the real and imaginary parts of the ComplexSIMD instance.

        Arguments:
            self: The ComplexSIMD instance to modify.
            idx: The index to access (0 for real, 1 for imaginary).
            value: The new value to set.
        """
        if idx == 0:
            self.re = value
        elif idx == 1:
            self.im = value
        else:
            raise Error("Index out of range")

    fn __setitem__(mut self, idx: Int, value: Self) raises:
        """
        Sets the real and imaginary parts of the ComplexSIMD instance.

        Arguments:
            self: The ComplexSIMD instance to modify.
            idx: The index to access (0 for real, 1 for imaginary).
            value: The new value to set.
        """
        if idx == 0:
            self.re = value.re
        elif idx == 1:
            self.im = value.im
        else:
            raise Error("Index out of range")

    fn item(self, idx: Int) raises -> SIMD[Self.dtype, Self.width]:
        """
        Gets the real or imaginary part of the ComplexSIMD instance.
        """
        return self[idx]

    fn itemset(mut self, val: ComplexSIMD[cdtype, Self.width]):
        """
        Sets the real and imaginary parts of the ComplexSIMD instance.

        Arguments:
            self: The ComplexSIMD instance to modify.
            val: The new value for the real and imaginary parts.
        """
        self.re = val.re
        self.im = val.im

    fn __abs__(self) -> SIMD[Self.dtype, Self.width]:
        """
        Returns the magnitude of the ComplexSIMD instance.
        """
        return sqrt(self.re * self.re + self.im * self.im)

    fn norm(self) -> SIMD[Self.dtype, Self.width]:
        """
        Returns the squared magnitude of the ComplexSIMD instance.
        """
        return sqrt(self.re * self.re + self.im * self.im)

    fn conj(self) -> Self:
        """
        Returns the complex conjugate of the ComplexSIMD instance.
        """
        return Self(self.re, -self.im)

    fn real(self) -> SIMD[Self.dtype, Self.width]:
        """
        Returns the real part of the ComplexSIMD instance.
        """
        return self.re

    fn imag(self) -> SIMD[Self.dtype, Self.width]:
        """
        Returns the imaginary part of the ComplexSIMD instance.
        """
        return self.im
