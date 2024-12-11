"""
Implements sort functions
"""
# ===----------------------------------------------------------------------=== #
# Sort Module - Implements sort functions
# Last updated: 2024-06-20
# ===----------------------------------------------------------------------=== #


import math
from algorithm import vectorize

from ..core.ndarray import NDArray, NDArrayShape

"""
TODO:
1) Add more sorting algorithms.
2) Add axis.
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

    Example:
        ```py
        var arr = numojo.core.random.rand[numojo.i16](100)
        var sorted_arr = numojo.core.sort.bubble_sort(arr)
        print(sorted_arr)
        ```

    Parameters:
        dtype: The input element type.

    Args:
        ndarray: An NDArray.

    Returns:
        The sorted NDArray.
    """
    var result: NDArray[dtype] = ndarray
    var length = ndarray.size

    for i in range(length):
        for j in range(length - i - 1):
            if result._buf.load[width=1](j) > result._buf.load[width=1](j + 1):
                var temp = result._buf.load[width=1](j)
                result._buf.store[width=1](j, result._buf.load[width=1](j + 1))
                result._buf.store[width=1](j + 1, temp)

    return result


# ===------------------------------------------------------------------------===#
# Quick sort
# ===------------------------------------------------------------------------===#


fn _partition(
    inout ndarray: NDArray, left: Int, right: Int, pivot_index: Int
) raises -> Int:
    """Do partition for the data buffer of ndarray.

    Args:
        ndarray: An NDArray.
        left: Left index of the partition.
        right: Right index of the partition.
        pivot_index: Input pivot index

    Returns:
        New pivot index.
    """

    var pivot_value = ndarray.get(pivot_index)
    var _value_at_pivot = ndarray.get(pivot_index)
    ndarray.set(pivot_index, ndarray.get(right))
    ndarray.set(right, _value_at_pivot)

    var store_index = left

    for i in range(left, right):
        if ndarray.get(i) < pivot_value:
            var _value_at_store = ndarray.get(store_index)
            ndarray.set(store_index, ndarray.get(i))
            ndarray.set(i, _value_at_store)
            store_index = store_index + 1

    var _value_at_store = ndarray.get(store_index)
    ndarray.set(store_index, ndarray.get(right))
    ndarray.set(right, _value_at_store)

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


fn quick_sort[dtype: DType](ndarray: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Quick sort the NDArray.
    Adopt in-place partition.
    Average complexity: O(nlogn).
    Worst-case complexity: O(n^2).
    Worst-case space complexity: O(n).
    Unstable.

    Example:
        ```py
        var arr = numojo.core.random.rand[numojo.i16](100)
        var sorted_arr = numojo.core.sort.quick_sort(arr)
        print(sorted_arr)
        ```

    Parameters:
        dtype: The input element type.

    Args:
        ndarray: An NDArray.

    """

    var result: NDArray[dtype] = ndarray
    var length = ndarray.size
    quick_sort_inplace(result, 0, length - 1)

    return result


# ===------------------------------------------------------------------------===#
# Binary sort
# ===------------------------------------------------------------------------===#


fn binary_sort[
    dtype: DType = DType.float64
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Binary sorting of NDArray.

    Example:
        ```py
        var arr = numojo.core.random.rand[numojo.i16](100)
        var sorted_arr = numojo.core.sort.binary_sort(arr)
        print(sorted_arr)
        ```

    Parameters:
         dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The sorted NDArray of type `dtype`.
    """

    @parameter
    if dtype != array.dtype:
        alias dtype = array.dtype

    var result: NDArray[dtype] = NDArray[dtype](array.shape)
    for i in range(array.size):
        result.store(i, array.get(i).cast[dtype]())

    var n = array.num_elements()
    for end in range(n, 1, -1):
        for i in range(1, end):
            if result[i - 1] > result[i]:
                var temp: Scalar[dtype] = result.get(i - 1)
                result.set(i - 1, result.get(i))
                result.store(i, temp)
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
    """Do partition for the indices of the data buffer of ndarray.

    Args:
        ndarray: An NDArray.
        idx_array: An NDArray.
        left: Left index of the partition.
        right: Right index of the partition.
        pivot_index: Input pivot index

    Returns:
        New pivot index.
    """

    var pivot_value = ndarray.get(pivot_index)

    var _value_at_pivot = ndarray.get(pivot_index)
    ndarray.set(pivot_index, ndarray.get(right))
    ndarray.set(right, _value_at_pivot)

    var _value_at_pivot_index = idx_array.get(pivot_index)
    idx_array.set(pivot_index, idx_array.get(right))
    idx_array.set(right, _value_at_pivot_index)

    var store_index = left

    for i in range(left, right):
        if ndarray.get(i) < pivot_value:
            var _value_at_store = ndarray.get(store_index)
            ndarray.set(store_index, ndarray.get(i))
            ndarray.set(i, _value_at_store)

            var _value_at_store_index = idx_array.get(store_index)
            idx_array.set(store_index, idx_array.get(i))
            idx_array.set(i, _value_at_store_index)

            store_index = store_index + 1

    var _value_at_store = ndarray.get(store_index)
    ndarray.set(store_index, ndarray.get(right))
    ndarray.set(right, _value_at_store)

    var _value_at_store_index = idx_array.get(store_index)
    idx_array.set(store_index, idx_array.get(right))
    idx_array.set(right, _value_at_store_index)

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

    Example:
        ```py
        var arr = numojo.core.random.rand[numojo.i16](100)
        var sorted_arr = numojo.core.sort.argsort(arr)
        print(sorted_arr)
        ```

    Parameters:
        dtype: The input element type.

    Args:
        ndarray: An NDArray.

    Returns:
        The indices of the sorted NDArray.
    """

    var array: NDArray[dtype] = ndarray
    var length = array.size

    var idx_array = NDArray[DType.index](Shape(length))
    for i in range(length):
        idx_array.set(i, SIMD[DType.index, 1](i))

    argsort_inplace(array, idx_array, 0, length - 1)

    return idx_array
