"""
# ===----------------------------------------------------------------------=== #
# CHECK ROUTINES
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #
"""

import math
import . _math_funcs as _mf
from ..core.ndarray import NDArray

# fn is_power_of_2[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](array: NDArray[dtype]) -> NDArray[DType.bool]:
#     return backend()._math_func_is[dtype, math.is_power_of_2](array)


# fn is_even[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](array: NDArray[dtype]) -> NDArray[DType.bool]:
#     return backend()._math_func_is[dtype, math.is_even](array)


# fn is_odd[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](array: NDArray[dtype]) -> NDArray[DType.bool]:
#     return backend()._math_func_is[dtype, math.is_odd](array)


fn isinf[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[DType.bool]:
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
    return backend()._math_func_is[dtype, math.isinf](array)


fn isfinite[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[DType.bool]:
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
    return backend()._math_func_is[dtype, math.isfinite](array)


fn isnan[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[DType.bool]:
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
    return backend()._math_func_is[dtype, math.isnan](array)
