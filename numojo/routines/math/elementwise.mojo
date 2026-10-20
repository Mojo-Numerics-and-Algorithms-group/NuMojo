# ===----------------------------------------------------------------------=== #
# NuMojo: Element-wise mathematical operations
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Element-wise math (numojo.routines.math.elementwise).
=====================================================
Common NumPy-compatible unary and binary mathematical operations.

Exports
-------
- `sign`, `square`, `negative`, `positive`: Unary operations.
- `power`: Element-wise exponentiation (also exported as `pow`).
- `fmax`, `fmin`: NaN-ignoring extrema.
- `fmod`: Remainder with the sign of the dividend.
- `rint`: Round floating-point values to the nearest integer.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
import std.math as math
from std.utils.numerics import isnan

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.ndarray import NDArray
from numojo.routines import HostExecutor


def sign[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[dtype] where dtype.is_numeric():
    """Return the element-wise indication of the sign of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: The input array.

    Returns:
        An array containing -1, 0, or 1 for each input value. NaNs are
        preserved.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var values = nm.array[f64]("[-3.5, 0.0, 2.5]")
        var result = nm.sign(values)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](value: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
        var result = value.lt(0).select[dtype](
            -1, value.eq(0).select[dtype](0, 1)
        )
        comptime if dtype.is_floating_point():
            return isnan(value).select(value, result)
        return result

    return HostExecutor.apply_unary[dtype, _kernel](array)


def square[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[dtype] where dtype.is_numeric():
    """Return the element-wise square of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: The input array.

    Returns:
        An array containing each input value multiplied by itself.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var values = nm.array[f64]("[-2.0, 0.0, 3.0]")
        var result = nm.square(values)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](value: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
        return value * value

    return HostExecutor.apply_unary[dtype, _kernel](array)


def negative[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[dtype] where dtype.is_numeric():
    """Return the numerical negative of each array element.

    Parameters:
        dtype: The element type.

    Args:
        array: The input array.

    Returns:
        An array containing the negated input values.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var values = nm.array[f64]("[-2.0, 0.0, 3.0]")
        var result = nm.negative(values)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](value: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
        return -value

    return HostExecutor.apply_unary[dtype, _kernel](array)


def positive[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[dtype] where dtype.is_numeric():
    """Return the numerical positive of each array element.

    Parameters:
        dtype: The element type.

    Args:
        array: The input array.

    Returns:
        A new array containing the unchanged input values.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var values = nm.array[f64]("[-2.0, 0.0, 3.0]")
        var result = nm.positive(values)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](value: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
        return +value

    return HostExecutor.apply_unary[dtype, _kernel](array)


def power[
    dtype: DType
](base: NDArray[dtype], exponent: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_numeric():
    """Raise each element in one array to the corresponding array exponent.

    Parameters:
        dtype: The element type.

    Args:
        base: The base array.
        exponent: The exponent array.

    Returns:
        The element-wise exponentiation result.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var base = nm.array[f64]("[2.0, 3.0, 4.0]")
        var exponent = nm.array[f64]("[3.0, 2.0, 0.5]")
        var result = nm.power(base, exponent)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](base: SIMD[dtype, width], exponent: SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ]:
        return base**exponent

    return HostExecutor.apply_binary[dtype, _kernel](base, exponent)


def power[
    dtype: DType
](base: NDArray[dtype], exponent: Scalar[dtype]) raises -> NDArray[
    dtype
] where dtype.is_numeric():
    """Raise each array element to a scalar exponent.

    Parameters:
        dtype: The element type.

    Args:
        base: The base array.
        exponent: The scalar exponent.

    Returns:
        The element-wise exponentiation result.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var base = nm.array[f64]("[2.0, 3.0, 4.0]")
        var exponent: Scalar[f64] = 2.0
        var result = nm.power(base, exponent)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](base: SIMD[dtype, width], exponent: SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ]:
        return base**exponent

    return HostExecutor.apply_binary[dtype, _kernel](base, exponent)


def power[
    dtype: DType
](base: Scalar[dtype], exponent: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_numeric():
    """Raise a scalar base to each exponent in an array.

    Parameters:
        dtype: The element type.

    Args:
        base: The scalar base.
        exponent: The exponent array.

    Returns:
        The element-wise exponentiation result.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var base: Scalar[f64] = 2.0
        var exponent = nm.array[f64]("[1.0, 2.0, 3.0]")
        var result = nm.power(base, exponent)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](base: SIMD[dtype, width], exponent: SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ]:
        return base**exponent

    return HostExecutor.apply_binary[dtype, _kernel](base, exponent)


def _fmax_value[
    dtype: DType, width: Int
](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime if dtype.is_floating_point():
        var numeric_result = lhs.gt(rhs).select(lhs, rhs)
        var zero_result = lhs + rhs
        var result = (lhs.eq(0) & rhs.eq(0)).select(zero_result, numeric_result)
        return isnan(lhs).select(rhs, isnan(rhs).select(lhs, result))
    return lhs.gt(rhs).select(lhs, rhs)


def fmax[
    dtype: DType
](lhs: NDArray[dtype], rhs: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_numeric():
    """Return the element-wise maximum while ignoring a single NaN.

    Parameters:
        dtype: The element type.

    Args:
        lhs: The first input array.
        rhs: The second input array.

    Returns:
        The element-wise NaN-ignoring maximum.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var lhs = nm.array[f64]("[1.0, 5.0, 3.0]")
        var rhs = nm.array[f64]("[2.0, 4.0, 6.0]")
        var result = nm.fmax(lhs, rhs)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ]:
        return _fmax_value(lhs, rhs)

    return HostExecutor.apply_binary[dtype, _kernel](lhs, rhs)


def fmax[
    dtype: DType
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[
    dtype
] where dtype.is_numeric():
    """Return the maximum of each array element and a scalar, ignoring NaN.

    Parameters:
        dtype: The element type.

    Args:
        array: The input array.
        scalar: The scalar input.

    Returns:
        The element-wise NaN-ignoring maximum.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var values = nm.array[f64]("[1.0, 5.0, 3.0]")
        var limit: Scalar[f64] = 4.0
        var result = nm.fmax(values, limit)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ]:
        return _fmax_value(lhs, rhs)

    return HostExecutor.apply_binary[dtype, _kernel](array, scalar)


def fmax[
    dtype: DType
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_numeric():
    """Return the maximum of a scalar and each array element, ignoring NaN.

    Parameters:
        dtype: The element type.

    Args:
        scalar: The scalar input.
        array: The input array.

    Returns:
        The element-wise NaN-ignoring maximum.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var limit: Scalar[f64] = 4.0
        var values = nm.array[f64]("[1.0, 5.0, 3.0]")
        var result = nm.fmax(limit, values)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ]:
        return _fmax_value(lhs, rhs)

    return HostExecutor.apply_binary[dtype, _kernel](scalar, array)


def _fmin_value[
    dtype: DType, width: Int
](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime if dtype.is_floating_point():
        var numeric_result = lhs.lt(rhs).select(lhs, rhs)
        var zero_result = -((-lhs) + (-rhs))
        var result = (lhs.eq(0) & rhs.eq(0)).select(zero_result, numeric_result)
        return isnan(lhs).select(rhs, isnan(rhs).select(lhs, result))
    return lhs.lt(rhs).select(lhs, rhs)


def fmin[
    dtype: DType
](lhs: NDArray[dtype], rhs: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_numeric():
    """Return the element-wise minimum while ignoring a single NaN.

    Parameters:
        dtype: The element type.

    Args:
        lhs: The first input array.
        rhs: The second input array.

    Returns:
        The element-wise NaN-ignoring minimum.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var lhs = nm.array[f64]("[1.0, 5.0, 3.0]")
        var rhs = nm.array[f64]("[2.0, 4.0, 6.0]")
        var result = nm.fmin(lhs, rhs)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ]:
        return _fmin_value(lhs, rhs)

    return HostExecutor.apply_binary[dtype, _kernel](lhs, rhs)


def fmin[
    dtype: DType
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[
    dtype
] where dtype.is_numeric():
    """Return the minimum of each array element and a scalar, ignoring NaN.

    Parameters:
        dtype: The element type.

    Args:
        array: The input array.
        scalar: The scalar input.

    Returns:
        The element-wise NaN-ignoring minimum.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var values = nm.array[f64]("[1.0, 5.0, 3.0]")
        var limit: Scalar[f64] = 4.0
        var result = nm.fmin(values, limit)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ]:
        return _fmin_value(lhs, rhs)

    return HostExecutor.apply_binary[dtype, _kernel](array, scalar)


def fmin[
    dtype: DType
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_numeric():
    """Return the minimum of a scalar and each array element, ignoring NaN.

    Parameters:
        dtype: The element type.

    Args:
        scalar: The scalar input.
        array: The input array.

    Returns:
        The element-wise NaN-ignoring minimum.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var limit: Scalar[f64] = 4.0
        var values = nm.array[f64]("[1.0, 5.0, 3.0]")
        var result = nm.fmin(limit, values)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ]:
        return _fmin_value(lhs, rhs)

    return HostExecutor.apply_binary[dtype, _kernel](scalar, array)


def _fmod_value[
    dtype: DType, width: Int
](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
    var quotient = lhs / rhs
    comptime if dtype.is_floating_point():
        quotient = quotient.__trunc__()
        return math.copysign(lhs - quotient * rhs, lhs)
    return lhs - quotient * rhs


def fmod[
    dtype: DType
](lhs: NDArray[dtype], rhs: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_numeric():
    """Return the element-wise remainder with the dividend's sign.

    Parameters:
        dtype: The element type.

    Args:
        lhs: The dividend array.
        rhs: The divisor array.

    Returns:
        The element-wise truncated-division remainder.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var lhs = nm.array[f64]("[-7.5, 5.0, 7.5]")
        var rhs = nm.array[f64]("[2.0, -2.0, 2.0]")
        var result = nm.fmod(lhs, rhs)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ]:
        return _fmod_value(lhs, rhs)

    return HostExecutor.apply_binary[dtype, _kernel](lhs, rhs)


def fmod[
    dtype: DType
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[
    dtype
] where dtype.is_numeric():
    """Return each array element modulo a scalar with the dividend's sign.

    Parameters:
        dtype: The element type.

    Args:
        array: The dividend array.
        scalar: The scalar divisor.

    Returns:
        The element-wise truncated-division remainder.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var values = nm.array[f64]("[-7.5, 5.0, 7.5]")
        var divisor: Scalar[f64] = 2.0
        var result = nm.fmod(values, divisor)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ]:
        return _fmod_value(lhs, rhs)

    return HostExecutor.apply_binary[dtype, _kernel](array, scalar)


def fmod[
    dtype: DType
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_numeric():
    """Return a scalar modulo each array element with the dividend's sign.

    Parameters:
        dtype: The element type.

    Args:
        scalar: The scalar dividend.
        array: The divisor array.

    Returns:
        The element-wise truncated-division remainder.

    Constraints:
        The element type must be numeric.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var dividend: Scalar[f64] = 8.0
        var divisors = nm.array[f64]("[3.0, -3.0, 5.0]")
        var result = nm.fmod(dividend, divisors)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ]:
        return _fmod_value(lhs, rhs)

    return HostExecutor.apply_binary[dtype, _kernel](scalar, array)


def rint[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """Round each floating-point element to the nearest integer value.

    Parameters:
        dtype: The element type.

    Args:
        array: The input array.

    Returns:
        An array rounded to integral values while retaining its dtype.

    Constraints:
        The element type must be floating point.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var values = nm.array[f64]("[-2.5, -1.5, 1.5, 2.5]")
        var result = nm.rint(values)
        ```
    """

    def _kernel[
        dtype: DType, width: Int
    ](value: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
        return value.__round__()

    return HostExecutor.apply_unary[dtype, _kernel](array)
