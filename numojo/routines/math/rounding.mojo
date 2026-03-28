# ===----------------------------------------------------------------------=== #
# NuMojo: Rounding routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Rounding routines for NuMojo (numojo.routines.math.rounding).

Implements rounding, truncation, absolute value, and next-after helpers for NDArrays.
"""

import std.math as builtin_math
from std.utils.numerics import nextafter as builtin_nextafter

from numojo.core.ndarray import NDArray
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix
from numojo.routines import HostExecutor

# ===------------------------------------------------------------------------===#
# Matrix Rounding
# ===------------------------------------------------------------------------===#


def round[dtype: DType](A: Matrix[dtype], decimals: Int = 0) -> Matrix[dtype]:
    # FIXME
    # The built-in `round` function is not working now.
    # It will be fixed in future.
    if not A.is_c_contiguous():
        return round(A.contiguous(), decimals)
    var res = Matrix.zeros[dtype](A.shape)
    for i in range(A.size):
        res._buf.ptr[i] = builtin_math.round(A._buf.ptr[i], ndigits=decimals)
    return res^


# ===------------------------------------------------------------------------===#
# Absolute Value
# ===------------------------------------------------------------------------===#


def tabs[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise absolute value of a NDArray.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to abs(array).
    """
    return HostExecutor.apply_unary[dtype, SIMD.__abs__](array)


# ===------------------------------------------------------------------------===#
# Rounding (NDArray)
# ===------------------------------------------------------------------------===#


def tfloor[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise floor of a NDArray.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to floor(array).
    """
    return HostExecutor.apply_unary[dtype, SIMD.__floor__](array)


def tceil[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise ceiling of a NDArray.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to ceil(array).
    """
    return HostExecutor.apply_unary[dtype, SIMD.__ceil__](array)


def ttrunc[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise truncation of a NDArray.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to trunc(array).
    """
    return HostExecutor.apply_unary[dtype, SIMD.__trunc__](array)


def tround[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise rounding of a NDArray to a whole number.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to round(array).
    """
    return HostExecutor.apply_unary[dtype, SIMD.__round__](array)


def roundeven[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise banker's rounding of a NDArray.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise rounding of `array` to the nearest integer with ties to even.
    """
    return HostExecutor.apply_unary[dtype, SIMD.__round__](array)


# def round_half_down[
#     dtype: DType
# ](NDArray: NDArray[dtype]) -> NDArray[dtype]:
#     """
#     Rounds ties towards the smaller integer.

#     Parameters:
#         dtype: The dtype of the input and output array.
#         backend: Sets utility function origin, defaults to `Vectorized`.

#     Args:
#         NDArray: array to perform rounding on.

#     Returns:
#     The element-wise rounding of x evaluating ties towards the smaller integer.
#     """
#     return backend().math_func_1_array_in_one_array_out[
#         dtype, SIMD.__round_half_down
#     ](NDArray)


# def round_half_up[
#     dtype: DType
# ](NDArray: NDArray[dtype]) -> NDArray[dtype]:
#     """
#     Rounds ties towards the larger integer.

#     Parameters:
#         dtype: The dtype of the input and output array.
#         backend: Sets utility function origin, defaults to `Vectorized`.

#     Args:
#         NDArray: array to perform rounding on.

#     Returns:
#     The element-wise rounding of x evaluating ties towards the larger integer.
#     """
#     return backend().math_func_1_array_in_one_array_out[
#         dtype, math.round_half_up
#     ](NDArray)

# ===------------------------------------------------------------------------===#
# Next After
# ===------------------------------------------------------------------------===#


def nextafter[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[
    dtype
] where dtype.is_floating_point():
    """
    Compute the next representable value after one array toward another.

    Parameters:
        dtype: The element type.

    Args:
        array1: The first input array.
        array2: The second input array.

    Constraints:
        Datatype `dtype` must be a floating-point type.

    Returns:
        The element-wise nextafter of `array1` toward `array2`.
    """
    return HostExecutor.apply_binary[dtype, builtin_nextafter](array1, array2)
