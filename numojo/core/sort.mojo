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
2) Add argument "inplace" for some functions.
3) Add axis.
"""

# ===------------------------------------------------------------------------===#
# Bubble sort
# ===------------------------------------------------------------------------===#


fn bubble_sort[dtype: DType](ndarray: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Bubble sort the NDArray.
    Average complexity: O(n^2) comparisons, O(n^2) swaps.
    Worst-case complexity: O(n^2) comparisons, O(n^2) swaps.
    Worst-case space complexity: O(n).

    Parameters:
        dtype: The input element type.

    Args:
        ndarray: An NDArray.

    Returns:
        The sorted NDArray.
    """
    var result: NDArray[dtype] = ndarray
    var length = ndarray.size()

    for i in range(length):
        for j in range(length - i - 1):
            if result.data.load[width=1](j) > result.data.load[width=1](j + 1):
                var temp = result.data.load[width=1](j)
                result.data.store[width=1](j, result.data.load[width=1](j + 1))
                result.data.store[width=1](j + 1, temp)

    return result


# ===------------------------------------------------------------------------===#
# Quick sort
# ===------------------------------------------------------------------------===#


fn _partition(
    inout ndarray: NDArray, left: Int, right: Int, pivot_index: Int
) raises -> Int:
    var pivot_value = ndarray[pivot_index]
    ndarray[pivot_index], ndarray[right] = ndarray[right], ndarray[pivot_index]
    var store_index = left

    for i in range(left, right):
        if ndarray[i] < pivot_value:
            ndarray[store_index], ndarray[i] = ndarray[i], ndarray[store_index]
            store_index = store_index + 1
    ndarray[right], ndarray[store_index] = ndarray[store_index], ndarray[right]

    return store_index


fn quick_sort_inplace[
    dtype: DType
](inout ndarray: NDArray[dtype], left: Int, right: Int,) raises:
    """
    Quick sort (in-place) the NDArray.

    Parameters:
        dtype: The input element type.

    Args:
        ndarray: An NDArray.
        left: Left index of the partition.
        right: Right index of the partition.
    """

    if right > left:
        var pivot_index = left + (right - left) // 2
        var pivot_new_index = _partition(ndarray, left, right, pivot_index)
        quick_sort_inplace(ndarray, left, pivot_new_index - 1)
        quick_sort_inplace(ndarray, pivot_new_index + 1, right)


fn quick_sort[
    dtype: DType
](ndarray: NDArray[dtype],) raises -> NDArray[dtype]:
    """
    Quick sort the NDArray.
    Adopt in-place partition.
    Average complexity: O(nlogn).
    Worst-case complexity: O(n^2).
    Worst-case space complexity: O(n).
    Unstable.

    Parameters:
        dtype: The input element type.

    Args:
        ndarray: An NDArray.

    Returns:
        The sorted NDArray.
    """

    var result: NDArray[dtype] = ndarray
    var length = ndarray.size()

    quick_sort_inplace(result, 0, length - 1)

    return result


# ===------------------------------------------------------------------------===#
# Binary sort
# ===------------------------------------------------------------------------===#


fn binary_sort[
    in_dtype: DType, out_dtype: DType = DType.float64
](array: NDArray[in_dtype]) raises -> NDArray[out_dtype]:
    """
    Binary sorting of NDArray.

    Parameters:
        in_dtype: The input element type.
        out_dtype: The output element type.

    Args:
        array: A NDArray.

    Returns:
        The sorted NDArray of type `out_dtype`.
    """
    var result: NDArray[out_dtype] = NDArray[out_dtype](array.shape())
    for i in range(array.ndshape._size):
        result[i] = array[i].cast[out_dtype]()

    var n = array.num_elements()
    for end in range(n, 1, -1):
        for i in range(1, end):
            if result[i - 1] > result[i]:
                var temp: Scalar[out_dtype] = result[i - 1]
                result[i - 1] = result[i]
                result[i] = temp
    return result


# ===------------------------------------------------------------------------===#
# Argsort using quick sort algorithm
# ===------------------------------------------------------------------------===#


fn _argsort_partition(
    inout ndarray: NDArray,
    inout idx_array: NDArray,
    left: Int,
    right: Int,
    pivot_index: Int,
) raises -> Int:
    var pivot_value = ndarray[pivot_index]
    ndarray[pivot_index], ndarray[right] = ndarray[right], ndarray[pivot_index]
    idx_array[pivot_index], idx_array[right] = (
        idx_array[right],
        idx_array[pivot_index],
    )
    var store_index = left

    for i in range(left, right):
        if ndarray[i] < pivot_value:
            ndarray[store_index], ndarray[i] = ndarray[i], ndarray[store_index]
            idx_array[store_index], idx_array[i] = (
                idx_array[i],
                idx_array[store_index],
            )
            store_index = store_index + 1

    ndarray[right], ndarray[store_index] = ndarray[store_index], ndarray[right]
    idx_array[right], idx_array[store_index] = (
        idx_array[store_index],
        idx_array[right],
    )

    return store_index


fn argsort_inplace[
    dtype: DType
](
    inout ndarray: NDArray[dtype],
    inout idx_array: NDArray[DType.index],
    left: Int,
    right: Int,
) raises:
    """
    Conduct Argsort (in-place) based on the NDArray using quick sort.

    Parameters:
        dtype: The input element type.

    Args:
        ndarray: An NDArray.
        idx_array: An NDArray of the indices.
        left: Left index of the partition.
        right: Right index of the partition.
    """

    if right > left:
        var pivot_index = left + (right - left) // 2
        var pivot_new_index = _argsort_partition(
            ndarray, idx_array, left, right, pivot_index
        )
        argsort_inplace(ndarray, idx_array, left, pivot_new_index - 1)
        argsort_inplace(ndarray, idx_array, pivot_new_index + 1, right)


fn argsort[
    dtype: DType
](ndarray: NDArray[dtype],) raises -> NDArray[DType.index]:
    """
    Argsort of the NDArray using quick sort algorithm.

    Parameters:
        dtype: The input element type.

    Args:
        ndarray: An NDArray.

    Returns:
        The indices of the sorted NDArray.
    """

    var array: NDArray[dtype] = ndarray
    var length = array.size()

    var idx_array = NDArray[DType.index](length)
    for i in range(length):
        idx_array[i] = i

    argsort_inplace(array, idx_array, 0, length - 1)

    return idx_array
