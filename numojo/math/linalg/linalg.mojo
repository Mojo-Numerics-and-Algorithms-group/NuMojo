"""
Linear Algebra misc. functions
"""
# ===----------------------------------------------------------------------=== #
# implements basic Linear Algebra functions
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #


import math
import .. math_funcs as _mf
from ...core.ndarray import NDArray, NDArrayShape
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc


fn cross[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Compute the cross product of two arrays.

    Parameters
        dtype: The element type.

    Args:
        array1: A array.
        array2: A array.

    Constraints:
        `array1` and `array2` must be of shape (3,).

    Returns:
        The cross product of two arrays.
    """

    if array1.ndshape.ndlen == array2.ndshape.ndlen == 3:
        var array3: NDArray[dtype] = NDArray[dtype](NDArrayShape(3))
        array3.store(
            0,
            (
                array1.get_scalar(1) * array2.get_scalar(2)
                - array1.get_scalar(2) * array2.get_scalar(1)
            ),
        )
        array3.store(
            1,
            (
                array1.get_scalar(2) * array2.get_scalar(0)
                - array1.get_scalar(0) * array2.get_scalar(2)
            ),
        )
        array3.store(
            2,
            (
                array1.get_scalar(0) * array2.get_scalar(1)
                - array1.get_scalar(1) * array2.get_scalar(0)
            ),
        )
        return array3
    else:
        raise Error(
            "Cross product is not supported for arrays of shape "
            + array1.shape().__str__()
            + " and "
            + array2.shape().__str__()
        )
