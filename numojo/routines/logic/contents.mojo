"""
Implements Checking routines: currently not SIMD due to bool bit packing issue
"""
# ===----------------------------------------------------------------------=== #
# Array contents
# ===----------------------------------------------------------------------=== #


import math
from utils.numerics import neg_inf, inf

import numojo.routines.math._math_funcs as _mf
from numojo.core.ndarray import NDArray

# fn is_power_of_2[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](array: NDArray[dtype]) -> NDArray[DType.bool]:
#     return backend().math_func_is[dtype, math.is_power_of_2](array)


# fn is_even[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](array: NDArray[dtype]) -> NDArray[DType.bool]:
#     return backend().math_func_is[dtype, math.is_even](array)


# fn is_odd[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](array: NDArray[dtype]) -> NDArray[DType.bool]:
#     return backend().math_func_is[dtype, math.is_odd](array)

# FIXME: Make all SIMD vectorized operations once bool bit-packing issue is resolved.
fn isinf[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Checks if each element of the input array is infinite.

    Parameters:
        dtype: DType - Data type of the input array.
        backend: _mf.Backend - Backend to use for the operation. Defaults to _mf.Vectorized.

    Args:
        array: NDArray[dtype] - Input array to check.

    Returns:
        NDArray[DType.bool] - A array of the same shape as `array` with True for infinite elements and False for others.
    """
    # return backend().math_func_is[dtype, math.isinf](array)

    var result_array: NDArray[DType.bool] = NDArray[DType.bool](array.shape)
    for i in range(result_array.size):
        result_array.store(i, math.isinf(array.load(i)))
    return result_array^


fn isfinite[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Checks if each element of the input array is finite.

    Parameters:
        dtype: DType - Data type of the input array.
        backend: _mf.Backend - Backend to use for the operation. Defaults to _mf.Vectorized.

    Args:
        array: NDArray[dtype] - Input array to check.

    Returns:
        NDArray[DType.bool] - A array of the same shape as `array` with True for finite elements and False for others.
    """
    # return backend().math_func_is[dtype, math.isfinite](array)
    var result_array: NDArray[DType.bool] = NDArray[DType.bool](array.shape)
    for i in range(result_array.size):
        result_array.store(i, math.isfinite(array.load(i)))
    return result_array^


fn isnan[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Checks if each element of the input array is NaN.

    Parameters:
        dtype: DType - Data type of the input array.
        backend: _mf.Backend - Backend to use for the operation. Defaults to _mf.Vectorized.

    Args:
        array: NDArray[dtype] - Input array to check.

    Returns:
        NDArray[DType.bool] - A array of the same shape as `array` with True for NaN elements and False for others.
    """
    # return backend().math_func_is[dtype, math.isnan](array)
    var result_array: NDArray[DType.bool] = NDArray[DType.bool](array.shape)
    for i in range(result_array.size):
        result_array.store(i, math.isnan(array.load(i)))
    return result_array^

fn isneginf[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Checks if each element of the input array is negative infinity.

    Parameters:
        dtype: DType - Data type of the input array.
        backend: _mf.Backend - Backend to use for the operation. Defaults to _mf.Vectorized.

    Args:
        array: NDArray[dtype] - Input array to check.

    Returns:
        NDArray[DType.bool] - A array of the same shape as `array` with True for negative infinite elements and False for others.
    """
    var result_array: NDArray[DType.bool] = NDArray[DType.bool](array.shape)
    for i in range(result_array.size):
        result_array.store(i, neg_inf[dtype]() == array.load(i))
    return result_array^

fn isposinf[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[DType.bool]:
    """
    Checks if each element of the input array is positive infinity.
    Parameters:
        dtype: DType - Data type of the input array.
        backend: _mf.Backend - Backend to use for the operation. Defaults to _mf.Vectorized.

    Args:
        array: NDArray[dtype] - Input array to check.

    Returns:
        NDArray[DType.bool] - A array of the same shape as `array` with True for positive infinite elements and False for others.
    """
    var result_array: NDArray[DType.bool] = NDArray[DType.bool](array.shape)
    for i in range(result_array.size):
        result_array.store(i, inf[dtype]() == array.load(i))
    return result_array^
