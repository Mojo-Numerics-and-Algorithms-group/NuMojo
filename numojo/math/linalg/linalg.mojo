"""
# ===----------------------------------------------------------------------=== #
# implements basic Linear Algebra functions
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #
"""

import math
import .._math_funcs as _mf
from ...core.ndarray import NDArray, NDArrayShape
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc

fn cross[
    in_dtype: DType, out_dtype: DType = DType.float32
](array1: NDArray[in_dtype], array2: NDArray[in_dtype]) raises -> NDArray[
    out_dtype
]:
    """
    Compute the cross product of two tensors.

    Parameters
        in_dtype: Input data type.
        out_dtype: Output data type, defaults to float32.

    Args:
        array1: A array.
        array2: A array.

    Constraints:
        `array1` and `array2` must be of shape (3,).

    Returns:
        The cross product of two tensors.
    """

    if array1.shape() == array2.shape() == 3:
        var array3: NDArray[out_dtype] = NDArray[out_dtype](NDArrayShape(3))
        array3[0] = (array1[1] * array2[2] - array1[2] * array2[1]).cast[
            out_dtype
        ]()
        array3[1] = (array1[2] * array2[0] - array1[0] * array2[2]).cast[
            out_dtype
        ]()
        array3[2] = (array1[0] * array2[1] - array1[1] * array2[0]).cast[
            out_dtype
        ]()
        return array3
    else:
        raise Error(
            "Cross product is not supported for tensors of shape "
            + array1.shape().__str__()
            + " and "
            + array2.shape().__str__()
        )