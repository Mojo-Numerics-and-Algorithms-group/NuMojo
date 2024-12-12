# ===----------------------------------------------------------------------=== #
# Searching
# ===----------------------------------------------------------------------=== #

import math
from algorithm import vectorize
from sys import simdwidthof
from collections.optional import Optional

from numojo.core.ndarray import NDArray
from numojo.core.ndshape import NDArrayShape
from numojo.core.utility import is_inttype, is_floattype
from numojo.routines.sorting import binary_sort


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
    if array.num_elements() == 0:
        raise Error("array is empty")

    var idx: Int = 0
    var max_val: Scalar[dtype] = array.get(0)
    for i in range(1, array.num_elements()):
        if array.get(i) > max_val:
            max_val = array.get(i)
            idx = i
    return idx


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
    if array.num_elements() == 0:
        raise Error("array is empty")

    var idx: Int = 0
    var min_val: Scalar[dtype] = array.get(0)

    for i in range(1, array.num_elements()):
        if array.get(i) < min_val:
            min_val = array.get(i)
            idx = i
    return idx
