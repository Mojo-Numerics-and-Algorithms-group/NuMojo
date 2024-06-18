"""
# ===----------------------------------------------------------------------=== #
# Implements RANDOM SAMPLING
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #
"""

import random
from .ndarray import NDArray
from .ndarray import arrayDescriptor

fn rand[dtype: DType](*shape: Int) -> NDArray[dtype]:
    """
    Example:
        numojo.core.random.rand[numojo.i8](3,2,4)
        Returns an random array with shape 3 x 2 x 4.
    """
    var dimension: Int = shape.__len__()
    var first_index: Int = 0
    var size: Int = 1
    var shapeInfo: List[Int] = List[Int]()
    var strides: List[Int] = List[Int]()

    for i in range(dimension):
        shapeInfo.append(shape[i])
        size *= shape[i]
        var temp: Int = 1
        for j in range(i + 1, dimension):  # temp
            temp *= shape[j]
        strides.append(temp)

    var arr = NDArray[dtype]()
    arr._arr = DTypePointer[dtype].alloc(size)
    memset_zero(arr._arr, size)
    arr.info = arrayDescriptor[dtype](
        dimension, first_index, size, shapeInfo, strides
    )
    random.rand[dtype](arr._arr, size)
    return arr