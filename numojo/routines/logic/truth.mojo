# ===----------------------------------------------------------------------=== #
# Truth value testing
# ===----------------------------------------------------------------------=== #

import math

import numojo.core._math_funcs as _mf
from numojo.core.ndarray import NDArray


fn any(array: NDArray[DType.bool]) raises -> Scalar[DType.bool]:
    """
    If any True.

    Args:
        array: A NDArray.
    Returns:
        A boolean scalar
    """
    var result = Scalar[DType.bool](False)
    # alias opt_nelts: Int = simdwidthof[DType.bool]()

    # @parameter
    # fn vectorize_sum[simd_width: Int](idx: Int) -> None:
    #     var simd_data = array.load[width=simd_width](idx)
    #     result &= simd_data.reduce_or()

    # vectorize[vectorize_sum, opt_nelts](array.num_elements())
    # return result
    for i in range(array.size):
        result |= array.load(i)
    return result


fn allt(array: NDArray[DType.bool]) raises -> Scalar[DType.bool]:
    """
    If all True.

    Args:
        array: A NDArray.
    Returns:
        A boolean scalar
    """
    var result = Scalar[DType.bool](True)
    # alias opt_nelts: Int = simdwidthof[DType.bool]()

    # @parameter
    # fn vectorize_sum[simd_width: Int](idx: Int) -> None:
    #     var simd_data = array.load[width=simd_width](idx)
    #     result |= simd_data.reduce_and()

    # vectorize[vectorize_sum, opt_nelts](array.num_elements())
    # return result
    for i in range(array.size):
        result &= array.load(i)
    return result
