# ===----------------------------------------------------------------------=== #
# Searching
# ===----------------------------------------------------------------------=== #

import math
from algorithm import vectorize
from sys import simdwidthof
from collections.optional import Optional

from numojo.core.ndarray import NDArray
from numojo.core.ndshape import NDArrayShape
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix
from numojo.core.utility import is_inttype, is_floattype
from numojo.routines.sorting import binary_sort
from numojo.routines.math.extrema import _max, _min


# * for loop version works fine for argmax and argmin, need to vectorize it
fn argmax[dtype: DType](array: NDArray[dtype]) raises -> Int:
    """
    Argmax of a array.

    Parameters:
        dtype: The element type.

    Args:
        array: A array.
    Returns:
        The index of the maximum value of the array.
    """
    if array.size == 0:
        raise Error("array is empty")

    var idx: Int = 0
    var max_val: Scalar[dtype] = array.load(0)
    for i in range(1, array.size):
        if array.load(i) > max_val:
            max_val = array.load(i)
            idx = i
    return idx


fn argmax[dtype: DType](A: Matrix[dtype]) raises -> Scalar[DType.index]:
    """
    Index of the max. It is first flattened before sorting.
    """

    var max_index: Scalar[DType.index]
    _, max_index = _max(A, 0, A.size - 1)

    return max_index


fn argmax[
    dtype: DType
](A: Matrix[dtype], axis: Int) raises -> Matrix[DType.index]:
    """
    Index of the max along the given axis.
    """
    if axis == 1:
        var B = Matrix[DType.index](shape=(A.shape[0], 1))
        for i in range(A.shape[0]):
            B._store(
                i,
                0,
                _max(A, start=i * A.strides[0], end=(i + 1) * A.strides[0] - 1)[
                    1
                ]
                - i * A.strides[0],
            )
        return B^
    elif axis == 0:
        return transpose(argmax(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn argmin[dtype: DType](array: NDArray[dtype]) raises -> Int:
    """
    Argmin of a array.
    Parameters:
        dtype: The element type.

    Args:
        array: A array.
    Returns:
        The index of the minimum value of the array.
    """
    if array.size == 0:
        raise Error("array is empty")

    var idx: Int = 0
    var min_val: Scalar[dtype] = array.load(0)

    for i in range(1, array.size):
        if array.load(i) < min_val:
            min_val = array.load(i)
            idx = i
    return idx


fn argmin[dtype: DType](A: Matrix[dtype]) raises -> Scalar[DType.index]:
    """
    Index of the min. It is first flattened before sorting.
    """

    var min_index: Scalar[DType.index]
    _, min_index = _min(A, 0, A.size - 1)

    return min_index


fn argmin[
    dtype: DType
](A: Matrix[dtype], axis: Int) raises -> Matrix[DType.index]:
    """
    Index of the min along the given axis.
    """
    if axis == 1:
        var B = Matrix[DType.index](shape=(A.shape[0], 1))
        for i in range(A.shape[0]):
            B._store(
                i,
                0,
                _min(A, start=i * A.strides[0], end=(i + 1) * A.strides[0] - 1)[
                    1
                ]
                - i * A.strides[0],
            )
        return B^
    elif axis == 0:
        return transpose(argmin(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))
