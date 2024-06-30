"""
# ===----------------------------------------------------------------------=== #
# Sort Module - Implements sort functions
# Last updated: 2024-06-20
# ===----------------------------------------------------------------------=== #
"""

import math
from algorithm import vectorize
from ..core.ndarray import NDArray, NDArrayShape

"""
TODO: 
1) Add more sorting algorithms.
"""

# ===------------------------------------------------------------------------===#
# Bubble sort
# ===------------------------------------------------------------------------===#

fn bubble_sort[
    dtype: DType
    ](
        ndarray: NDArray[dtype]
        ) raises -> NDArray[dtype]:
    """
    Bubble sort the NDArray.
    Average complexity: O(n^2) comparisons, O(n^2) swaps.
    Worst-case complexity: O(n^2) comparisons, O(n^2) swaps.
    Worst-case space complexity: O(n).

    Parameters:
        dtype: The input element type.

    Args:
        ndarray: A NDArray.

    Returns:
        The sorted NDArray.
    """
    var result: NDArray[dtype] = ndarray
    var length = ndarray.size()

    for i in range(length):
        for j in range(length-i-1):
            if result.data.load[width=1](j) > result.data.load[width=1](j+1):
                var temp = result.data.load[width=1](j)
                result.data.store[width=1](j, result.data.load[width=1](j+1))
                result.data.store[width=1](j+1, temp)

    return result
