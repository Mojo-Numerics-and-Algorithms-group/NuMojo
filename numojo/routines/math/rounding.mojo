# ===------------------------------------------------------------------------===#
# Rounding
# ===------------------------------------------------------------------------===#

from builtin import math as builtin_math
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from utils import Variant
from utils.numerics import nextafter as builtin_nextafter

import numojo.routines.math._math_funcs as _mf
from numojo.core.ndarray import NDArray
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix, MatrixImpl


fn round[
    dtype: DType
](A: MatrixImpl[dtype, **_], decimals: Int = 0) -> Matrix[dtype]:
    # FIXME
    # The built-in `round` function is not working now.
    # It will be fixed in future.
    var res = Matrix.zeros[dtype](A.shape)
    for i in range(A.size):
        res._buf.ptr[i] = builtin_math.round(A._buf.ptr[i], ndigits=decimals)
    return res^


fn tabs[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise absolute value of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to abs(NDArray).
    """
    return backend().math_func_1_array_in_one_array_out[dtype, SIMD.__abs__](
        array
    )


fn tfloor[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise round down to nearest whole number of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to floor(NDArray).
    """
    return backend().math_func_1_array_in_one_array_out[dtype, SIMD.__floor__](
        array
    )


fn tceil[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise round up to nearest whole number of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to ceil(NDArray).
    """
    return backend().math_func_1_array_in_one_array_out[dtype, SIMD.__ceil__](
        array
    )


fn ttrunc[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise remove decimal value from float whole number of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to trunc(NDArray).
    """
    return backend().math_func_1_array_in_one_array_out[dtype, SIMD.__trunc__](
        array
    )


fn tround[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise round NDArray to whole number.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to trunc(NDArray).
    """
    return backend().math_func_1_array_in_one_array_out[dtype, SIMD.__round__](
        array
    )


fn roundeven[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Performs element-wise banker's rounding on the elements of a NDArray.

    Parameters:
        dtype: The dtype of the input and output array.
        backend: Sets utility function origin, defaults to `Vectorized`.

    Args:
        array: Array to perform rounding on.

    Returns:
    The element-wise banker's rounding of NDArray.

    This rounding goes to the nearest integer with ties toward the nearest even integer.
    """
    return backend().math_func_1_array_in_one_array_out[dtype, SIMD.__round__](
        array
    )


# fn round_half_down[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
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


# fn round_half_up[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
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


fn nextafter[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Computes the nextafter of the inputs.

    Parameters:
        dtype: The dtype of the input and output array. Constraints: must be a floating-point type.
        backend: Sets utility function origin, default to `Vectorized`.


    Args:
        array1: The first input argument.
        array2: The second input argument.

    Returns:
        The nextafter of the inputs.
    """
    return backend().math_func_2_array_in_one_array_out[
        dtype, builtin_nextafter
    ](array1, array2)
