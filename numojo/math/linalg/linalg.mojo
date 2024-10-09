"""
Linear Algebra misc. functions
"""
# ===----------------------------------------------------------------------=== #
# implements basic Linear Algebra functions
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #


import math
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc

import .. math_funcs as _mf
from ...core.ndarray import NDArray, NDArrayShape


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

    if (array1.ndshape.ndsize == array2.ndshape.ndsize == 3) and (
        array1.ndshape.ndlen == array2.ndshape.ndlen == 1
    ):
        var array3: NDArray[dtype] = NDArray[dtype](NDArrayShape(3))
        array3.store(
            0,
            (array1.get(1) * array2.get(2) - array1.get(2) * array2.get(1)),
        )
        array3.store(
            1,
            (array1.get(2) * array2.get(0) - array1.get(0) * array2.get(2)),
        )
        array3.store(
            2,
            (array1.get(0) * array2.get(1) - array1.get(1) * array2.get(0)),
        )
        return array3
    else:
        raise Error(
            "Cross product is not supported for arrays of shape "
            + array1.shape().__str__()
            + " and "
            + array2.shape().__str__()
        )


# TODO: implement other cases for dot function
fn dot[
    dtype: DType = DType.float64
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Compute the dot product of two arrays.

    Parameters
        dtype: The element type.

    Args:
        array1: A array.
        array2: A array.

    Constraints:
        `array1` and `array2` must be 1 dimensional.

    Returns:
        The dot product of two arrays.
    """

    alias width = simdwidthof[dtype]()
    if array1.ndshape.ndlen == array2.ndshape.ndlen == 1:
        var result: NDArray[dtype] = NDArray[dtype](
            NDArrayShape(array1.ndshape.ndsize)
        )

        @parameter
        fn vectorized_dot[simd_width: Int](idx: Int) -> None:
            result.store[width=simd_width](
                idx,
                array1.load[width=simd_width](idx)
                * array2.load[width=simd_width](idx),
            )

        vectorize[vectorized_dot, width](array1.ndshape.ndsize)
        return result^
    else:
        raise Error(
            "Cross product is not supported for arrays of shape "
            + array1.shape().__str__()
            + " and "
            + array2.shape().__str__()
        )
