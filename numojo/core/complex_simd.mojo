from math import sqrt

from ._complex_dtype import CDType

# alias ComplexScalar = ComplexSIMD[_, CDType.to_dtype[_](), 1]


@register_passable("trivial")
struct ComplexSIMD[
    ctype: CDType,
    *,
    type: DType = CDType.to_dtype[ctype](),
    size: Int = 1
]():
    """
    Represents a SIMD[dtype, 1] Complex number with real and imaginary parts.
    """

    # FIELDS
    """The underlying data type of the complex number."""
    var re: SIMD[type, size]
    var im: SIMD[type, size]

    @always_inline
    fn __init__(out self, other: Self):
        """
        Initializes a ComplexSIMD instance by copying another instance.

        Arguments:
            other: Another ComplexSIMD instance to copy from.
        """
        self = other

    @always_inline
    fn __init__(out self, re: SIMD[type, size], im: SIMD[type, size]):
        """
        Initializes a ComplexSIMD instance with specified real and imaginary parts.

        Arguments:
            re: The real part of the complex number.
            im: The imaginary part of the complex number.

        Example:
        ```mojo
        var A = ComplexSIMD[cf32](SIMD[f32, 1](1.0), SIMD[f32, 1](2.0))
        var B = ComplexSIMD[cf32](SIMD[f32, 1](3.0), SIMD[f32, 1](4.0))
        var C = A + B
        print(C)
        ```
        """

        self.re = re
        self.im = im

    @always_inline
    fn __init__(out self, val: SIMD[type, size]):
        """
        Initializes a ComplexSIMD instance with specified real and imaginary parts.

        Arguments:
            re: The real part of the complex number.
            im: The imaginary part of the complex number.
        """
        self.re = rebind[Scalar[type]](val)
        self.im = rebind[Scalar[type]](val)

    fn __add__(self, other: Self) -> Self:
        """
        Adds two ComplexSIMD instances.

        Arguments:
            other: The ComplexSIMD instance to add.

        Returns:
            Self: A new ComplexSIMD instance representing the sum.
        """
        return Self(self.re + other.re, self.im + other.im)

    # fn __add__(self, other: ComplexSIMD[ctype, type]) -> ComplexSIMD[ctype, type]:
    #     """
    #     Adds two ComplexSIMD instances.

    #     Arguments:
    #         other: The ComplexSIMD instance to add.

    #     Returns:
    #         Self: A new ComplexSIMD instance representing the sum.
    #     """
    #     return ComplexSIMD[ctype, type](rebind[Scalar[CDType.to_dtype[ctype]()]](self.re) + other.re, rebind[Scalar[CDType.to_dtype[ctype]()]](self.im) + other.im)

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
            writer.write("({} + {} j)".format(self.re, self.im))
        except e:
            writer.write("Cannot convert ComplexSIMD to string")

    fn __repr__(self) raises -> String:
        """
        Returns a string representation of the ComplexSIMD instance.

        Returns:
            String: The string representation of the ComplexSIMD instance.
        """
        return "ComplexSIMD[{}]({}, {})".format(str(type), self.re, self.im)

    fn __getitem__(self, idx: Int) raises -> SIMD[type, size]:
        """
        Gets the real or imaginary part of the ComplexSIMD instance.

        Arguments:
            self: The ComplexSIMD instance.
            idx: The index to access (0 for real, 1 for imaginary).

        Returns:
            SIMD[dtype, 1]: The requested part of the ComplexSIMD instance.
        """
        if idx == 0:
            return self.re
        elif idx == 1:
            return self.im
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

    fn __setitem__(mut self, idx: Int, re: SIMD[type, size], im: SIMD[type, size]):
        """
        Sets the real and imaginary parts of the ComplexSIMD instance.

        Arguments:
            self: The ComplexSIMD instance to modify.
            idx: The index to access (0 for real, 1 for imaginary).
            re: The new value for the real part.
            im: The new value for the imaginary part.
        """
        self.re = re
        self.im = im

    fn __abs__(self) -> SIMD[type, size]:
        """
        Returns the magnitude of the ComplexSIMD instance.
        """
        return sqrt(self.re * self.re + self.im * self.im)

    fn norm(self) -> SIMD[type, size]:
        """
        Returns the squared magnitude of the ComplexSIMD instance.
        """
        return sqrt(self.re * self.re + self.im * self.im)

    fn conj(self) -> Self:
        """
        Returns the complex conjugate of the ComplexSIMD instance.
        """
        return Self(self.re, -self.im)

    fn real(self) -> SIMD[type, size]:
        """
        Returns the real part of the ComplexSIMD instance.
        """
        return self.re

    fn imag(self) -> SIMD[type, size]:
        """
        Returns the imaginary part of the ComplexSIMD instance.
        """
        return self.im
