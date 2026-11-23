# ===----------------------------------------------------------------------=== #
# NuMojo: Floating-point routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""
Floating (numojo.routines.math.floating).
=========================================
Floating-point specific operations for NDArrays.

Implements floating-point helper functions such as `copysign` for element-wise
sign manipulation.

Exports
-------
- `copysign`: Copy sign from one array to another.
- `reciprocal`: Element-wise multiplicative inverse.
- `spacing`: Distance to the next representable floating-point value.
- `signbit`: Whether the sign bit is set.
- `heaviside`: The Heaviside step function.
- `modf`: Fractional and integral parts.
- `frexp`, `ldexp`: Mantissa/exponent decomposition and reconstruction.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
import std.math as math

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines import HostExecutor

# ===----------------------------------------------------------------------=== #
# Sign Copy
# ===----------------------------------------------------------------------=== #


def copysign[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Copy the sign of one array onto another.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Raises:
        Error if shape of `array1` and `array2` do not match.

    Returns:
        A NDArray with the magnitude of `array2` and the sign of `array1`.
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) capturing -> SIMD[
        dtype, simd_w
    ]:
        return math.copysign(simd1, simd2)

    return HostExecutor.apply_binary[dtype, _kernel](array1, array2)


# ===----------------------------------------------------------------------=== #
# Reciprocal
# ===----------------------------------------------------------------------=== #


def reciprocal[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise multiplicative inverse of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to 1 / array.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[1.0, 2.0, 4.0]")
        print(nm.reciprocal(a))  # [1.0, 0.5, 0.25]
    ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) {} -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.recip(simd)

    return HostExecutor.apply_unary_float[dtype, _kernel](array)


# ===----------------------------------------------------------------------=== #
# Spacing
# ===----------------------------------------------------------------------=== #


def spacing[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the signed distance between each element and the nearest
    adjacent floating-point value in the direction away from zero.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray of the spacing of each element of `array`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[1.0, -1.0]")
        print(nm.spacing(a))
    ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) {} -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.copysign(math.ulp(simd), simd)

    return HostExecutor.apply_unary_float[dtype, _kernel](array)


# ===----------------------------------------------------------------------=== #
# Signbit
# ===----------------------------------------------------------------------=== #


def signbit[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    DType.bool
] where dtype.is_floating_point():
    """
    Return whether the sign bit is set for each element of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A boolean NDArray that is `True` where `array` is negative
        (including negative zero).

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[1.0, -1.0, 0.0, -0.0]")
        print(nm.signbit(a))  # [False, True, False, True]
    ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) capturing -> SIMD[DType.bool, simd_w]:
        return simd._to_bits_signed().lt(0)

    return HostExecutor.apply_unary_predicate[dtype, _kernel](array)


# ===----------------------------------------------------------------------=== #
# Heaviside
# ===----------------------------------------------------------------------=== #


def heaviside[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the Heaviside step function element-wise.

    Matches `numpy.heaviside`: returns 0 where `array1` is negative, 1
    where `array1` is positive, and the corresponding element of
    `array2` where `array1` is zero.

    Parameters:
        dtype: The element type.

    Args:
        array1: The input array.
        array2: The value to use where `array1` is zero.

    Returns:
        A NDArray equal to the element-wise Heaviside step function.

    Constraints:
        Both arrays must have the same shapes.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[-1.0, 0.0, 2.0]")
        var b = nm.fromstring("[0.5, 0.5, 0.5]")
        print(nm.heaviside(a, b))  # [0.0, 0.5, 1.0]
    ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) capturing -> SIMD[
        dtype, simd_w
    ]:
        return simd1.lt(0).select[dtype](
            0, simd1.eq(0).select(simd2, 1)
        )

    return HostExecutor.apply_binary[dtype, _kernel](array1, array2)


# ===----------------------------------------------------------------------=== #
# Fractional / integral decomposition
# ===----------------------------------------------------------------------=== #


def modf[
    dtype: DType
](array: NDArray[dtype]) raises -> Tuple[
    NDArray[dtype], NDArray[dtype]
] where dtype.is_floating_point():
    """
    Return the fractional and integral parts of each array element.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A tuple `(fractional, integral)`, each the same shape as
        `array`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[3.5, -2.25]")
        var result = nm.modf(a)
        print(result[0])  # [0.5, -0.25]
        print(result[1])  # [3.0, -2.0]
    ```
    """
    var a_c = array.contiguous()
    var frac = NDArray[dtype](Shape(a_c.shape))
    var intg = NDArray[dtype](Shape(a_c.shape))
    for i in range(a_c.size):
        var parts = math.modf(a_c.unsafe_get(i))
        intg.unsafe_set(i, parts[0])
        frac.unsafe_set(i, parts[1])

    return Tuple(frac^, intg^)


# ===----------------------------------------------------------------------=== #
# Mantissa / exponent decomposition
# ===----------------------------------------------------------------------=== #


def frexp[
    dtype: DType
](array: NDArray[dtype]) raises -> Tuple[
    NDArray[dtype], NDArray[DType.int32]
] where dtype.is_floating_point():
    """
    Decompose each array element into a normalized mantissa and a
    base-2 exponent.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A tuple `(mantissa, exponent)` such that
        `array == mantissa * 2**exponent` and `0.5 <= |mantissa| < 1`
        for non-zero elements.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[8.0]")
        var result = nm.frexp(a)
        print(result[0])  # [0.5]
        print(result[1])  # [4]
    ```
    """
    var a_c = array.contiguous()
    var mantissa = NDArray[dtype](Shape(a_c.shape))
    var exponent = NDArray[DType.int32](Shape(a_c.shape))
    for i in range(a_c.size):
        var parts = math.frexp(a_c.unsafe_get(i))
        mantissa.unsafe_set(i, parts[0])
        exponent.unsafe_set(i, parts[1].cast[DType.int32]())

    return Tuple(mantissa^, exponent^)


def ldexp[
    dtype: DType
](x: NDArray[dtype], exp: NDArray[DType.int32]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Multiply each element of `x` by two raised to the corresponding
    element of `exp`. The inverse of `frexp`.

    Parameters:
        dtype: The element type.

    Args:
        x: A NDArray of mantissas.
        exp: A NDArray of exponents, the same shape as `x`.

    Returns:
        A NDArray equal to `x * 2**exp`.

    Raises:
        NumojoError: If `x` and `exp` do not have the same size.

    Constraints:
        Both arrays must have the same shapes.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var x = nm.fromstring("[0.5]")
        var exp = nm.fromstring[nm.i32]("[4]")
        print(nm.ldexp(x, exp))  # [8.0]
    ```
    """
    if x.size != exp.size:
        raise Error(
            NumojoError(
                category="shape",
                message="`ldexp` requires `x` and `exp` to be the same"
                " size.",
                location="ldexp",
            )
        )

    var x_c = x.contiguous()
    var exp_c = exp.contiguous()
    var result = NDArray[dtype](Shape(x_c.shape))
    for i in range(x_c.size):
        result.unsafe_set(i, math.ldexp(x_c.unsafe_get(i), exp_c.unsafe_get(i)))

    return result^
