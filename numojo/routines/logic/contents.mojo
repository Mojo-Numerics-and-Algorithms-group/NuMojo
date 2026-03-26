# ===----------------------------------------------------------------------=== #
# NuMojo: Contents
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Contents routines (numojo.routines.logic.contents)

Implements Checking routines: currently not SIMD due to bool bit packing issue
"""

import std.math as math
from std.utils.numerics import neg_inf, inf

from numojo.routines import HostExecutor
from numojo.core.ndarray import NDArray
from numojo.core.matrix import Matrix

# TODO: Add scalar overloads of these functions.
# TODO: Remove matrix operations in future.
# TODO: Implement the commented out functions now that mojo supports these functions in SIMD.
# FIXME: Make all SIMD vectorized operations once bool bit-packing issue is resolved.

# def is_power_of_2[
#     dtype: DType
# ](array: NDArray[dtype]) -> NDArray[DType.bool]:
#     return backend().math_func_is[dtype, math.is_power_of_2](array)


# def is_even[
#     dtype: DType
# ](array: NDArray[dtype]) -> NDArray[DType.bool]:
#     return backend().math_func_is[dtype, math.is_even](array)


# def is_odd[
#     dtype: DType
# ](array: NDArray[dtype]) -> NDArray[DType.bool]:
#     return backend().math_func_is[dtype, math.is_odd](array)


# ===------------------------------------------------------------------------===#
# Check operations
# ===------------------------------------------------------------------------===#


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
    return HostExecutor.apply_unary_predicate[dtype, math.isinf](array)


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
    return HostExecutor.apply_unary_predicate[dtype, math.isfinite](array)


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
    return HostExecutor.apply_unary_predicate[dtype, math.isnan](array)


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

    def is_neginf[
        dtype: DType, simd_width: Int
    ](x: SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]:
        return x.eq(SIMD[dtype, simd_width](neg_inf[dtype]()))

    return HostExecutor.apply_unary_predicate[dtype, is_neginf](array)


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

    def is_posinf[
        dtype: DType, simd_width: Int
    ](x: SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]:
        return x.eq(SIMD[dtype, simd_width](inf[dtype]()))

    return HostExecutor.apply_unary_predicate[dtype, is_posinf](array)


def isneginf[dtype: DType](matrix: Matrix[dtype]) raises -> Matrix[DType.bool]:
    """
    Checks if each element of the input Matrix is negative infinity.

    Parameters:
        dtype: DType - Data type of the input Matrix.

    Args:
        matrix: Input Matrix to check.

    Returns:
        A Matrix of the same shape as `matrix` with True for negative infinite elements and False for others.
    """
    var result_array: Matrix[DType.bool] = Matrix[DType.bool](matrix.shape)
    for i in range(result_array.size):
        result_array.store(i, neg_inf[dtype]() == matrix.load(i))
    return result_array^


def isposinf[dtype: DType](matrix: Matrix[dtype]) raises -> Matrix[DType.bool]:
    """
    Checks if each elements of the input Matrix is positive infinity.

    Parameters:
        dtype: DType - Data type of the input Matrix.

    Args:
        matrix: Input Matrix to check.

    Returns:
        A Matrix of the same shape as `Matrix` with True for positive infinite elements and False for others.
    """
    var result_array: Matrix[DType.bool] = Matrix[DType.bool](matrix.shape)
    for i in range(result_array.size):
        result_array.store(i, inf[dtype]() == matrix.load(i))
    return result_array^
