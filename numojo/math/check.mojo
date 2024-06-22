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

fn any(array: NDArray[DType.bool]) -> Scalar[DType.bool]:
    """
    If any True.

    Args:
        array: A NDArray.
    Returns:
        A boolean scalar
    """
    var result = Scalar[DType.bool](False)
    alias opt_nelts: Int = simdwidthof[DType.bool]()

    @parameter
    fn vectorize_sum[simd_width: Int](idx: Int) -> None:
        var simd_data = array.load[width=simd_width](idx)
        result &= (simd_data.reduce_or())

    vectorize[vectorize_sum, opt_nelts](array.num_elements())
    return result

fn all(array: NDArray[DType.bool]) -> Scalar[DType.bool]:
    """
    If all True.

    Args:
        array: A NDArray.
    Returns:
        A boolean scalar
    """
    var result = Scalar[DType.bool](True)
    alias opt_nelts: Int = simdwidthof[DType.bool]()

    @parameter
    fn vectorize_sum[simd_width: Int](idx: Int) -> None:
        var simd_data = array.load[width=simd_width](idx)
        result |= (simd_data.reduce_and())

    vectorize[vectorize_sum, opt_nelts](array.num_elements())
    return result