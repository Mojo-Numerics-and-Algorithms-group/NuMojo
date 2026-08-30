# ===----------------------------------------------------------------------=== #
# NuMojo: Contents
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Contents (numojo.routines.logic.contents).
==========================================
Element properties and content checking for arrays.

Functions for checking element properties (NaN, infinite, finite) and array
contents (not SIMD due to bool bit packing issue).

Exports
-------
- `isinf`: Check for infinite elements.
- `isfinite`: Check for finite elements.
- `isnan`: Check for NaN elements.
- `isneginf`: Check for negative infinity.
- `isposinf`: Check for positive infinity.
"""

# TODO: Implement commented functions now that Mojo supports them in SIMD.
# FIXME: Make all SIMD vectorized once bool bit-packing issue is resolved.

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
import std.math as math
from std.utils.numerics import inf, neg_inf

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.ndarray import NDArray
from numojo.routines import HostExecutor

# ===----------------------------------------------------------------------=== #
# Check operations
# ===----------------------------------------------------------------------=== #


def isinf[dtype: DType](array: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Checks if each element of the input array is infinite.

    Parameters:
        dtype: Data type of the input array.

    Args:
        array: Input array to check.

    Returns:
        An array of the same shape as `array` with True for infinite elements and False for others.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.contents import isinf

        def main() raises:
            var arr = linspace(0, 10, 5)  # Example array: [0.0, 2.5, 5.0, 7.5, 10.0]
            print(isinf(arr))  # Output: [False, False, False, False, False]
        ```
    """

    @parameter
    def is_inf_kernel[
        dtype: DType, simd_width: Int
    ](x: SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]:
        return math.isinf(x)

    return HostExecutor.apply_unary_predicate[dtype, is_inf_kernel](array)


def isinf[dtype: DType](value: Scalar[dtype]) raises -> Scalar[DType.bool]:
    """
    Checks if the input scalar is infinite.

    Parameters:
        dtype: Data type of the input scalar.

    Args:
        value: Input scalar to check.

    Returns:
        True if `value` is positive or negative infinity, False otherwise.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.contents import isinf
        from std.utils.numerics import inf

        def main() raises:
            print(isinf(inf[f64]()))  # Output: True
            print(isinf(Scalar[f64](1.0)))  # Output: False
        ```
    """
    return math.isinf(value)


def isfinite[dtype: DType](array: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Checks if each element of the input array is finite.

    Parameters:
        dtype: Data type of the input array.

    Args:
        array: Input array to check.

    Returns:
        An array of the same shape as `array` with True for finite elements and False for others.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.contents import isfinite

        def main() raises:
            var arr = nm.array[nm.f64]([1.0, Float64.MAX, Float64.MIN], shape=[3])
            print(isfinite(arr))  # Output: [True, True, True]
        ```
    """

    @parameter
    def is_finite_kernel[
        dtype: DType, simd_width: Int
    ](x: SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]:
        return math.isfinite(x)

    return HostExecutor.apply_unary_predicate[dtype, is_finite_kernel](array)


def isfinite[dtype: DType](value: Scalar[dtype]) raises -> Scalar[DType.bool]:
    """
    Checks if the input scalar is finite.

    Parameters:
        dtype: Data type of the input scalar.

    Args:
        value: Input scalar to check.

    Returns:
        True if `value` is neither infinite nor NaN, False otherwise.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.contents import isfinite

        def main() raises:
            print(isfinite(Scalar[f64](1.0)))  # Output: True
            print(isfinite(Float64.MAX))  # Output: True
        ```
    """
    return math.isfinite(value)


def isnan[dtype: DType](array: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Checks if each element of the input array is NaN.

    Parameters:
        dtype: Data type of the input array.

    Args:
        array: Input array to check.

    Returns:
        An array of the same shape as `array` with True for NaN elements and False for others.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.contents import isnan

        def main() raises:
            var arr = nm.array[nm.f64]([1.0, 0.0, Float64.MAX], shape=[3])
            print(isnan(arr))  # Output: [False, False, False]
        ```
    """

    @parameter
    def is_nan_kernel[
        dtype: DType, simd_width: Int
    ](x: SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]:
        return math.isnan(x)

    return HostExecutor.apply_unary_predicate[dtype, is_nan_kernel](array)


def isnan[dtype: DType](value: Scalar[dtype]) raises -> Scalar[DType.bool]:
    """
    Checks if the input scalar is NaN.

    Parameters:
        dtype: Data type of the input scalar.

    Args:
        value: Input scalar to check.

    Returns:
        True if `value` is NaN, False otherwise.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.contents import isnan
        from std.utils.numerics import nan

        def main() raises:
            print(isnan(nan[f64]()))  # Output: True
            print(isnan(Scalar[f64](1.0)))  # Output: False
        ```
    """
    return math.isnan(value)


def isneginf[dtype: DType](array: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Checks if each element of the input array is negative infinity.

    Parameters:
        dtype: Data type of the input array.

    Args:
        array: Input array to check.

    Returns:
        An array of the same shape as `array` with True for negative infinite elements and False for others.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.contents import isneginf

        def main() raises:
            var arr = nm.array[nm.f64]([1.0, 0.0, -1.0], shape=[3])
            print(isneginf(arr))  # Output: [False, False, False]
        ```
    """

    @parameter
    def is_neginf[
        dtype: DType, simd_width: Int
    ](x: SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]:
        return x.eq(SIMD[dtype, simd_width](neg_inf[dtype]()))

    return HostExecutor.apply_unary_predicate[dtype, is_neginf](array)


def isneginf[dtype: DType](value: Scalar[dtype]) raises -> Scalar[DType.bool]:
    """
    Checks if the input scalar is negative infinity.

    Parameters:
        dtype: Data type of the input scalar.

    Args:
        value: Input scalar to check.

    Returns:
        True if `value` is negative infinity, False otherwise.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.contents import isneginf
        from std.utils.numerics import neg_inf

        def main() raises:
            print(isneginf(neg_inf[f64]()))  # Output: True
            print(isneginf(Scalar[f64](-1.0)))  # Output: False
        ```
    """
    return value.eq(neg_inf[dtype]())


def isposinf[dtype: DType](array: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Checks if each element of the input array is positive infinity.

    Parameters:
        dtype: Data type of the input array.

    Args:
        array: Input array to check.

    Returns:
        An array of the same shape as `array` with True for positive infinite elements and False for others.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.contents import isposinf

        def main() raises:
            var arr = nm.array[nm.f64]([1.0, 0.0, -1.0], shape=[3])
            print(isposinf(arr))  # Output: [False, False, False]
        ```
    """

    @parameter
    def is_posinf[
        dtype: DType, simd_width: Int
    ](x: SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]:
        return x.eq(SIMD[dtype, simd_width](inf[dtype]()))

    return HostExecutor.apply_unary_predicate[dtype, is_posinf](array)


def isposinf[dtype: DType](value: Scalar[dtype]) raises -> Scalar[DType.bool]:
    """
    Checks if the input scalar is positive infinity.

    Parameters:
        dtype: Data type of the input scalar.

    Args:
        value: Input scalar to check.

    Returns:
        True if `value` is positive infinity, False otherwise.

    Examples:
        ```mojo
        from numojo.prelude import *
        from numojo.routines.logic.contents import isposinf
        from std.utils.numerics import inf

        def main() raises:
            print(isposinf(inf[f64]()))  # Output: True
            print(isposinf(Scalar[f64](1.0)))  # Output: False
        ```
    """
    return value.eq(inf[dtype]())
