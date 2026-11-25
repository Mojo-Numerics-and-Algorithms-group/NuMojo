# ===----------------------------------------------------------------------=== #
# NuMojo: Exponential routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""
Exponential routines (numojo.routines.math.exponents).
======================================================
Exponential and logarithmic functions for arrays.

Element-wise exponential functions (exp, exp2, expm1) and logarithmic functions
(log, log2, log10, log1p) for NDArrays.

Exports
-------
- `exp`, `exp2`, `expm1`: Exponential functions.
- `log`, `log2`, `log10`, `log1p`: Logarithmic functions.
- `logaddexp`, `logaddexp2`: Overflow-safe log-sum-exp functions.
- `float_power`: `power` constrained to floating-point input.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
import std.math as math
from max.algorithm import parallelize
from std.algorithm import Static2DTileUnitFunc as Tile2DFunc
from std.utils import Variant
from std.utils.numerics import isinf, isnan

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.ndarray import NDArray
from numojo.routines import HostExecutor

# ===------------------------------------------------------------------------===#
# Exponential functions
# ===------------------------------------------------------------------------===#


def exp[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise exponential of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is e**x for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var arr = nm.linspace[f64](0.0, 1.0, 10)
        var result = nm.exp(arr)
        ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) {} -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.exp(simd)

    return HostExecutor.apply_unary_float[dtype, _kernel](array)


def exp[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute the exponential of a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to e**value.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var value: Scalar[f32] = 1.0
        var result = nm.exp(value)
        ```
    """
    return math.exp(value)


def exp2[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise base-2 exponential of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is 2**x for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var arr = nm.linspace[f64](0.0, 1.0, 10)
        var result = nm.exp2(arr)
        ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) {} -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.exp2(simd)

    return HostExecutor.apply_unary_float[dtype, _kernel](array)


def exp2[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute the base-2 exponential of a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to 2**value.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var value: Scalar[f32] = 1.0
        var result = nm.exp2(value)
        ```
    """
    return math.exp2(value)


def expm1[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise exp(x) - 1 of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is exp(x) - 1 for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var arr = nm.linspace[f64](0.0, 1.0, 10)
        var result = nm.expm1(arr)
        ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) {} -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.expm1(simd)

    return HostExecutor.apply_unary_float[dtype, _kernel](array)


def expm1[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute exp(value) - 1 for a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to exp(value) - 1.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var value: Scalar[f32] = 1.0
        var result = nm.expm1(value)
        ```
    """
    return math.expm1(value)


# ===------------------------------------------------------------------------===#
# Logarithmic functions
# ===------------------------------------------------------------------------===#


def log[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise natural logarithm of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is ln(x) for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var arr = nm.arange[f64](1.0, 10.0, 1.0)
        var result = nm.log(arr)
        ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) {} -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.log(simd)

    return HostExecutor.apply_unary_float[dtype, _kernel](array)


def log[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute the natural logarithm of a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to ln(value).

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var result = nm.log(10.0)
        ```
    """
    return math.log(value)


def log2[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise base-2 logarithm of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is log2(x) for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var arr = nm.arange[f64](1.0, 10.0, 1.0)
        var result = nm.log2(arr)
        ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) {} -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.log2(simd)

    return HostExecutor.apply_unary_float[dtype, _kernel](array)


def log2[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute the base-2 logarithm of a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to log2(value).

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var result = nm.log2(10.0)
        ```
    """
    return math.log2(value)


def log10[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise base-10 logarithm of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is log10(x) for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var arr = nm.arange[f64](1.0, 10.0, 1.0)
        var result = nm.log10(arr)
        ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) {} -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.log10(simd)

    return HostExecutor.apply_unary_float[dtype, _kernel](array)


def log10[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute the base-10 logarithm of a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to log10(value).

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var result = nm.log10(10.0)
        ```
    """
    return math.log10(value)


def log1p[
    dtype: DType
](array: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the element-wise ln(1 + x) of an array.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray where each element is ln(1 + x) for the corresponding element x.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var arr = nm.linspace[f64](0.0, 1.0, 10)
        var result = nm.log1p(arr)
        ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) {} -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return math.log1p(simd)

    return HostExecutor.apply_unary_float[dtype, _kernel](array)


def log1p[
    dtype: DType
](value: Scalar[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Compute ln(1 + value) for a scalar.

    Parameters:
        dtype: The element type.

    Args:
        value: A Scalar.

    Returns:
        A scalar equal to ln(1 + value).

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *
        var result = nm.log1p(1.0)
        ```
    """
    return math.log1p(value)


# ===------------------------------------------------------------------------===#
# Log-sum-exp
# ===------------------------------------------------------------------------===#


def logaddexp[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute `log(exp(array1) + exp(array2))`, element-wise, without
    overflow.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to `log(exp(array1) + exp(array2))`.

    Constraints:
        Both arrays must have the same shapes.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.fromstring("[1.0, 2.0, 3.0]")
        var b = nm.fromstring("[4.0, 5.0, 6.0]")
        var result = nm.logaddexp(a, b)
        ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) {} -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        var mx = simd1.gt(simd2).select(simd1, simd2)
        var absdiff = (simd1 - simd2).__abs__()
        var stable = mx + math.log1p(math.exp(-absdiff))
        var neg_inf = isinf(mx) & mx.lt(0)
        return neg_inf.select(mx, stable)

    return HostExecutor.apply_binary_float[dtype, _kernel](array1, array2)


def logaddexp2[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute `log2(2**array1 + 2**array2)`, element-wise, without
    overflow.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to `log2(2**array1 + 2**array2)`.

    Constraints:
        Both arrays must have the same shapes.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var a = nm.fromstring("[1.0, 2.0, 3.0]")
        var b = nm.fromstring("[4.0, 5.0, 6.0]")
        var result = nm.logaddexp2(a, b)
        ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) {} -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        var mx = simd1.gt(simd2).select(simd1, simd2)
        var absdiff = (simd1 - simd2).__abs__()
        # `math.exp2` crashes on an infinite/NaN input, so an infinite
        # `absdiff` (same magnitudes are infinite and apart) is replaced
        # with a value that still underflows `exp2` to exactly 0.
        var bad = isinf(absdiff) | isnan(absdiff)
        var safe_neg_absdiff = bad.select(SIMD[dtype, simd_w](-1100), -absdiff)
        var stable = mx + math.log2(1 + math.exp2(safe_neg_absdiff))
        var neg_inf = isinf(mx) & mx.lt(0)
        return neg_inf.select(mx, stable)

    return HostExecutor.apply_binary_float[dtype, _kernel](array1, array2)


# ===------------------------------------------------------------------------===#
# Float power
# ===------------------------------------------------------------------------===#


def float_power[
    dtype: DType
](base: NDArray[dtype], exponent: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Raise each element in one array to the corresponding array exponent,
    constrained to floating-point input.

    Unlike `numpy.float_power`, this does not promote integer input to
    float64; cast with `astype` first if needed.

    Parameters:
        dtype: The element type.

    Args:
        base: The base array.
        exponent: The exponent array.

    Returns:
        A NDArray equal to `base**exponent`.

    Constraints:
        Both arrays must have the same shapes.

    Examples:
        ```mojo
        import numojo as nm
        from numojo.prelude import *

        var base = nm.fromstring("[1.0, 2.0, 3.0]")
        var exponent = nm.fromstring("[2.0, 3.0, 2.0]")
        var result = nm.float_power(base, exponent)
        ```
    """

    def _kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) {} -> SIMD[
        dtype, simd_w
    ] where dtype.is_floating_point():
        return simd1**simd2

    return HostExecutor.apply_binary_float[dtype, _kernel](base, exponent)
