# ===----------------------------------------------------------------------=== #
# NuMojo: Rounding routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Rounding (numojo.routines.math.rounding).
=========================================
Rounding, truncation, and floating-point operations.

Element-wise rounding (floor, ceiling, truncation), absolute value, banker's
rounding, and next-after floating-point operations for NDArrays.

Exports
-------
- `tabs`: Absolute value.
- `tfloor`: Floor.
- `tceil`: Ceiling.
- `ttrunc`: Truncation.
- `tround`: Rounding.
- `roundeven`: Banker's rounding.
- `nextafter`: Next representable value.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.utils.numerics import nextafter as builtin_nextafter

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.ndarray import NDArray
from numojo.routines import HostExecutor

# ===----------------------------------------------------------------------=== #
# Absolute Value
# ===----------------------------------------------------------------------=== #


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[dtype, simd_w]:
        return simd.__abs__()

    return HostExecutor.apply_unary[dtype, _kernel](array)


# ===----------------------------------------------------------------------=== #
# Rounding (NDArray)
# ===----------------------------------------------------------------------=== #


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[dtype, simd_w]:
        return simd.__floor__()

    return HostExecutor.apply_unary[dtype, _kernel](array)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[dtype, simd_w]:
        return simd.__ceil__()

    return HostExecutor.apply_unary[dtype, _kernel](array)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[dtype, simd_w]:
        return simd.__trunc__()

    return HostExecutor.apply_unary[dtype, _kernel](array)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[dtype, simd_w]:
        return simd.__round__()

    return HostExecutor.apply_unary[dtype, _kernel](array)


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd: SIMD[dtype, simd_w]) -> SIMD[dtype, simd_w]:
        return simd.__round__()

    return HostExecutor.apply_unary[dtype, _kernel](array)


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

# ===----------------------------------------------------------------------=== #
# Next After
# ===----------------------------------------------------------------------=== #


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

    @parameter
    def _kernel[
        dtype: DType, simd_w: Int
    ](simd1: SIMD[dtype, simd_w], simd2: SIMD[dtype, simd_w]) -> SIMD[
        dtype, simd_w
    ]:
        return builtin_nextafter(simd1, simd2)

    return HostExecutor.apply_binary[dtype, _kernel](array1, array2)
