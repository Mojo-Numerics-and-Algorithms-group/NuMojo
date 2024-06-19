# ===----------------------------------------------------------------------=== #
# implements basic Diferential Calculus functions
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #

import math
import .._math_funcs as _mf
from ...core.ndarray import NDArray, NDArrayShape
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc

fn diff[
    in_dtype: DType, out_dtype: DType = in_dtype
](array: NDArray[in_dtype], n: Int) raises -> NDArray[out_dtype]:
    """
    Compute the n-th order difference of the input array.

    Parameters:
        in_dtype: Input data type.
        out_dtype: Output data type, defaults to float32.

    Args:
        array: A array.
        n: The order of the difference.

    Returns:
        The n-th order difference of the input array.
    """

    var array1: NDArray[out_dtype] = NDArray[out_dtype](
        NDArrayShape(array.num_elements())
    )
    for i in range(array.num_elements()):
        array1[i] = array[i].cast[out_dtype]()

    for num in range(n):
        var result: NDArray[out_dtype] = NDArray[out_dtype](
            NDArrayShape(array.num_elements() - (num + 1))
        )
        for i in range(array1.num_elements() - 1):
            result[i] = (array1.load[1](i + 1) - array1.load[1](i)).cast[
                out_dtype
            ]()
        array1 = result
    return array1