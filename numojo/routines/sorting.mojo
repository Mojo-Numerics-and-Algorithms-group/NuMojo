# ===----------------------------------------------------------------------=== #
# NuMojo: Sorting routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Sorting routines (numojo.routines.sorting).
===========================================
Array sorting and indexing operations.

Sorting routines for NDArrays including sort and argsort functions
using multiple backend algorithms (binary sort, bubble sort, quick sort).

Exports
-------
- `sort`: Sort array elements in-place.
- `argsort`: Return indices that would sort array.
- `partition`: Partially sort an array so the k-th element is in its
  sorted position.
- `argpartition`: Return indices that would partition an array.
- `lexsort`: Return indices that would sort an array using several keys.

Notes:
    - Multiple sorting methods available: binary sort, bubble sort, quick sort.
    - Quick sort is unstable but efficient.
"""

# TODO: Add more sorting algorithms (merge sort, heap sort).

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
import std.math
from std.algorithm import vectorize

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.layout import NDArrayShape
from numojo.core.ndarray import NDArray
from numojo.routines.creation import arange
from numojo.routines.functional import (
    apply_along_axis_preserve,
    apply_along_axis_inplace,
    apply_along_axis_indices,
)
from numojo.routines.manipulation import ravel, transpose


# ===----------------------------------------------------------------------=== #
# Sorting functions exposed to users
# ===----------------------------------------------------------------------=== #


# Below are overrides for NDArray type
def sort[
    dtype: DType
](a: NDArray[dtype], stable: Bool = False) raises -> NDArray[dtype]:
    """
    Sort NDArray using quick sort method.
    It is not guaranteed to be unstable.
    When no axis is given, the output array is flattened to 1d.

    Parameters:
        dtype: The input element type.

    Args:
        a: NDArray.
        stable: If True, the sorting is stable. Default is False.
    """
    if stable:
        return quick_sort_stable_1d(a)
    else:
        return quick_sort_1d(a)


def sort[
    dtype: DType
](a: NDArray[dtype], axis: Int, stable: Bool = False) raises -> NDArray[dtype]:
    """
    Sort NDArray along the given axis using quick sort method.
    It is not guaranteed to be unstable.
    When no axis is given, the array is flattened before sorting.

    Parameters:
        dtype: The input element type.

    Args:
        a: NDArray to sort.
        axis: The axis along which the array is sorted.
        stable: If True, the sorting is stable. Default is False.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `mean`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="sort",
            )
        )

    if (a.ndim == 1) and (normalized_axis == 0):
        if stable:
            return quick_sort_stable_1d(a)
        else:
            return quick_sort_1d(a)

    if stable:
        return apply_along_axis_preserve[dtype, func1d=quick_sort_stable_1d](
            a, axis=normalized_axis
        )
    else:
        return apply_along_axis_preserve[dtype, func1d=quick_sort_1d](
            a, axis=normalized_axis
        )


def sort_inplace[
    dtype: DType
](mut a: NDArray[dtype], axis: Int, stable: Bool = False) raises:
    """
    Sort NDArray in-place along the given axis using quick sort method.
    It is not guaranteed to be unstable.

    Parameters:
        dtype: The input element type.

    Args:
        a: NDArray to sort.
        axis: The axis along which the array is sorted.
        stable: If True, the sorting is stable. Default is False.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `mean`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="sort_inplace",
            )
        )

    if (a.ndim == 1) and (normalized_axis == 0):
        if stable:
            quick_sort_stable_inplace_1d(a)
        else:
            quick_sort_inplace_1d(a)

    if stable:
        apply_along_axis_inplace[dtype, func1d=quick_sort_stable_inplace_1d](
            a, axis=normalized_axis
        )
    else:
        apply_along_axis_inplace[dtype, func1d=quick_sort_inplace_1d](
            a, axis=normalized_axis
        )


# Array sorting overloads
def argsort[dtype: DType](a: NDArray[dtype]) raises -> NDArray[DType.int]:
    """
    Returns the indices that would sort an array.
    It is not guaranteed to be unstable.
    When no axis is given, the array is flattened before sorting.

    Parameters:
        dtype: The input element type.

    Args:
        a: NDArray.

    Returns:
        Indices that would sort an array.
    """

    var a_flattened: NDArray[dtype]
    if a.ndim == 1:
        a_flattened = a.contiguous()
    else:
        a_flattened = ravel(a)

    var indices = arange[DType.int](Scalar[DType.int](a_flattened.size))

    _quick_sort_inplace(a_flattened, indices)

    return indices^


def argsort[
    dtype: DType
](mut a: NDArray[dtype], axis: Int) raises -> NDArray[DType.int]:
    """
    Returns the indices that would sort an array.
    It is not guaranteed to be unstable.
    When no axis is given, the array is flattened before sorting.

    Raises:
        NumojoError: If the axis is out of bound.

    Parameters:
        dtype: The input element type.

    Args:
        a: NDArray to sort.
        axis: The axis along which the array is sorted.

    Returns:
        Indices that would sort an array.

    """

    var normalized_axis: Int = axis
    if normalized_axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis >= a.ndim) or (normalized_axis < 0):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `mean`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="argsort",
            )
        )

    if (a.ndim == 1) and (normalized_axis == 0):
        return argsort_quick_sort_1d(a)

    return apply_along_axis_indices[dtype, func1d=argsort_quick_sort_1d](
        a, axis=normalized_axis
    )


# ===----------------------------------------------------------------------=== #
# Partitioning
# ===----------------------------------------------------------------------=== #


def partition[
    dtype: DType, //
](a: NDArray[dtype], kth: Int) raises -> NDArray[dtype]:
    """
    Returns a partially sorted copy of a flattened array such that the
    element at position `kth` is in the position it would occupy in a
    fully sorted array; every element before it is less than or equal to
    it, and every element after it is greater than or equal to it.
    Regardless of the shape of input, the array is flattened first.

    Parameters:
        dtype: The input element type.

    Args:
        a: NDArray.
        kth: The index (into the flattened, sorted array) to partition
            around. Supports negative indices.

    Returns:
        A partially sorted, flattened copy of `a`.

    Raises:
        NumojoError: If `kth` is out of bound.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.fromstring[nm.i32]("[3, 1, 4, 1, 5, 9, 2, 6]")
        print(nm.partition(a, kth=3))  # 4-th smallest value is at index 3
        ```
    """
    var flat: NDArray[dtype]
    if a.ndim == 1:
        flat = a.contiguous()
    else:
        flat = ravel(a)

    var n = flat.size
    var k = kth
    if k < 0:
        k += n
    if (k < 0) or (k >= n):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `partition`: kth {} not in bound [-{}, {})"
                ).format(kth, n, n),
                location="partition",
            )
        )

    _quickselect_inplace(flat, k)
    return flat^


def argpartition[
    dtype: DType, //
](a: NDArray[dtype], kth: Int) raises -> NDArray[DType.int]:
    """
    Returns the indices that would partition a flattened array such that
    the element at position `kth` of the partitioned array is in the
    position it would occupy in a fully sorted array. Regardless of the
    shape of input, the array is flattened first.

    Parameters:
        dtype: The input element type.

    Args:
        a: NDArray.
        kth: The index (into the flattened, sorted array) to partition
            around. Supports negative indices.

    Returns:
        Indices that would partition `a`.

    Raises:
        NumojoError: If `kth` is out of bound.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.fromstring[nm.i32]("[3, 1, 4, 1, 5, 9, 2, 6]")
        print(nm.argpartition(a, kth=3))
        ```
    """
    var flat: NDArray[dtype]
    if a.ndim == 1:
        flat = a.contiguous()
    else:
        flat = ravel(a)

    var n = flat.size
    var k = kth
    if k < 0:
        k += n
    if (k < 0) or (k >= n):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `argpartition`: kth {} not in bound [-{}, {})"
                ).format(kth, n, n),
                location="argpartition",
            )
        )

    var indices = arange[DType.int](Scalar[DType.int](n))
    _quickselect_inplace(flat, indices, k)
    return indices^


# ===----------------------------------------------------------------------=== #
# Lexicographic sorting
# ===----------------------------------------------------------------------=== #


def lexsort[
    dtype: DType, //
](keys: List[NDArray[dtype]]) raises -> NDArray[DType.int]:
    """
    Returns the indices that would lexicographically sort a sequence of
    key arrays of equal length. The last key is treated as the primary
    sort key, the second-to-last as the secondary sort key, and so on.

    Parameters:
        dtype: The input element type.

    Args:
        keys: A non-empty list of 1-d NDArrays of equal length to sort
            by, ordered from least to most significant.

    Returns:
        Indices that would sort the keys lexicographically.

    Raises:
        NumojoError: If `keys` is empty, or if the key arrays do not all
            have the same length.

    Examples:
        ```mojo
        import numojo as nm

        var surnames = nm.fromstring[nm.i32]("[1, 2, 1, 2]")
        var ages = nm.fromstring[nm.i32]("[30, 25, 20, 25]")
        # Sort by surname, breaking ties by age.
        print(nm.lexsort(List(ages, surnames)))
        ```
    """
    if len(keys) == 0:
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "Error in `lexsort`: `keys` must contain at least one"
                    " array."
                ),
                location="lexsort",
            )
        )

    var num_keys = len(keys)
    var n = keys[0].size
    for k in range(1, num_keys):
        if keys[k].size != n:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "Error in `lexsort`: All key arrays must have the"
                        " same length."
                    ),
                    location="lexsort",
                )
            )

    var order = List[Int]()
    for i in range(n):
        order.append(i)
    for i in range(1, n):
        var key = order[i]
        var j = i - 1
        while (j >= 0) and (
            _lexsort_compare(keys, num_keys, order[j], key) > 0
        ):
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key

    var result = NDArray[DType.int](NDArrayShape(n))
    for i in range(n):
        result.unsafe_set(i, order[i])
    return result^


def _lexsort_compare[
    dtype: DType
](keys: List[NDArray[dtype]], num_keys: Int, i: Int, j: Int) raises -> Int:
    """Internal: lexicographically compares row `i` and row `j` across
    `keys`, from the last (most significant) key to the first. Returns
    -1, 0, or 1."""
    for k in range(num_keys - 1, -1, -1):
        var a = keys[k].unsafe_get(i)
        var b = keys[k].unsafe_get(j)
        if a < b:
            return -1
        if a > b:
            return 1
    return 0


def binary_sort_1d[dtype: DType](a: NDArray[dtype]) raises -> NDArray[dtype]:
    var result: NDArray[dtype] = a.contiguous()
    for end in range(result.size, 1, -1):
        for i in range(1, end):
            if result.unsafe_get(i - 1) > result.unsafe_get(i):
                var temp = result.unsafe_get(i - 1)
                result.unsafe_set(i - 1, result.unsafe_get(i))
                result.unsafe_set(i, temp)
    return result^


def binary_sort[
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

    comptime if dtype != array.dtype:
        comptime dtype = array.dtype

    var result: NDArray[dtype] = NDArray[dtype](array.shape)
    for i in range(array.size):
        result.store(i, array.load(i).cast[dtype]())

    var n = array.size
    for end in range(n, 1, -1):
        for i in range(1, end):
            if result[i - 1] > result[i]:
                var temp: Scalar[dtype] = result.load(i - 1)
                result.store(i - 1, result.load(i))
                result.store(i, temp)
    return result^


###############
# Bubble sort #
###############


def bubble_sort[dtype: DType](ndarray: NDArray[dtype]) raises -> NDArray[dtype]:
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
    # * We can make it into a in place operation to avoid copy.
    var result: NDArray[dtype] = ndarray.contiguous()
    var length: Int = ndarray.size

    for i in range(length):
        for j in range(length - i - 1):
            if result.unsafe_load[width=1](j) > result.unsafe_load[width=1](
                j + 1
            ):
                var temp = result.unsafe_load[width=1](j)
                result.unsafe_store[width=1](
                    j, result.unsafe_load[width=1](j + 1)
                )
                result.unsafe_store[width=1](j + 1, temp)

    return result^


##############
# Quick sort #
##############


def quick_sort_1d[
    dtype: DType
](a: NDArray[dtype]) capturing raises -> NDArray[dtype]:
    """
    Sort array using quick sort method.
    Regardless of the shape of input, it is treated as a 1-d array.
    It is not guaranteed to be unstable.

    Parameters:
        dtype: The input element type.

    Args:
        a: An 1-d array.
    """
    # * copies are temporary solution for now.
    var result: NDArray[dtype]
    if a.ndim == 1:
        result = a.contiguous()
    else:
        result = ravel(a)

    _quick_sort_inplace(result)

    return result^


def quick_sort_stable_1d[
    dtype: DType
](a: NDArray[dtype]) capturing raises -> NDArray[dtype]:
    """
    Sort array using quick sort method.
    Regardless of the shape of input, it is treated as a 1-d array.
    The sorting is stable.

    Parameters:
        dtype: The input element type.

    Args:
        a: An 1-d array.
    """
    var result: NDArray[dtype]
    if a.ndim == 1:
        result = a.contiguous()
    else:
        result = ravel(a)

    _quick_sort_stable_inplace(result, result.size)

    return result^


def quick_sort_inplace_1d[dtype: DType](mut a: NDArray[dtype]) capturing raises:
    """
    Sort array in-place using quick sort method.
    Regardless of the shape of input, it is treated as a 1-d array.
    It is not guaranteed to be unstable.

    Parameters:
        dtype: The input element type.

    Args:
        a: An 1-d array.
    """
    if a.ndim != 1:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Error in `quick_sort_inplace_1d`: "
                    "The input array must be 1-d array."
                ),
                location="quick_sort_inplace_1d",
            )
        )
    _quick_sort_inplace(a)
    return


def quick_sort_stable_inplace_1d[
    dtype: DType
](mut a: NDArray[dtype]) capturing raises:
    """
    Sort array in-place using quick sort method.
    Regardless of the shape of input, it is treated as a 1-d array.
    The sorting is stable.

    Parameters:
        dtype: The input element type.

    Args:
        a: An 1-d array.
    """
    if a.ndim != 1:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "Error in `quick_sort_inplace_1d`: "
                    "The input array must be 1-d array."
                ),
                location="quick_sort_stable_inplace_1d",
            )
        )
    _quick_sort_stable_inplace(a, a.size)
    return


def argsort_quick_sort_1d[
    dtype: DType
](a: NDArray[dtype]) capturing raises -> NDArray[DType.int]:
    """
    Returns the indices that would sort the buffer of an array.
    Regardless of the shape of input, it is treated as a 1-d array.
    It is not guaranteed to be unstable.

    Parameters:
        dtype: The input element type.

    Args:
        a: NDArray.

    Returns:
        Indices that would sort an array.
    """

    var result: NDArray[dtype] = a.contiguous()
    var indices = arange[DType.int](Scalar[DType.int](result.size))
    _quick_sort_inplace(result, indices)
    return indices^


def _unsafe_swap[
    dtype: DType
](mut array: NDArray[dtype], left: Int, right: Int):
    """Swap two logical flat elements without bounds checks."""
    var value = array.unsafe_get(left)
    array.unsafe_set(left, array.unsafe_get(right))
    array.unsafe_set(right, value)


def _partition_in_range(
    mut A: NDArray,
    left: Int,
    right: Int,
    pivot_index: Int,
) raises -> Int:
    """
    Do in-place partition for array buffer within given range.
    Auxiliary function for `sort`, `argsort`, and `partition`.

    Args:
        A: NDArray.
        left: Left index of the partition.
        right: Right index of the partition.
        pivot_index: Input pivot index

    Returns:
        New pivot index.
    """

    var pivot_value = A.unsafe_get(pivot_index)

    _unsafe_swap(A, pivot_index, right)

    var store_index = left

    for i in range(left, right):
        if A.unsafe_get(i) < pivot_value:
            _unsafe_swap(A, store_index, i)
            store_index = store_index + 1

    _unsafe_swap(A, store_index, right)

    return store_index


def _partition_in_range(
    mut A: NDArray,
    mut I: NDArray,
    left: Int,
    right: Int,
    pivot_index: Int,
) raises -> Int:
    """
    Do in-place partition for array buffer within given range.
    The indices are also sorted.
    Auxiliary function for `sort`, `argsort`, and `partition`.

    Args:
        A: NDArray.
        I: NDArray used to store indices.
        left: Left index of the partition.
        right: Right index of the partition.
        pivot_index: Input pivot index

    Returns:
        New pivot index.
    """

    var pivot_value = A.unsafe_get(pivot_index)

    _unsafe_swap(A, pivot_index, right)
    _unsafe_swap(I, pivot_index, right)

    var store_index = left

    for i in range(left, right):
        if A.unsafe_get(i) < pivot_value:
            _unsafe_swap(A, store_index, i)
            _unsafe_swap(I, store_index, i)
            store_index = store_index + 1

    _unsafe_swap(A, store_index, right)
    _unsafe_swap(I, store_index, right)

    return store_index


def _quickselect_inplace[dtype: DType](mut A: NDArray[dtype], k: Int) raises:
    """
    Partitions in-place the data buffer (quickselect) so that index `k`
    holds the value it would have in a fully sorted array.
    Auxiliary function for `partition`.

    Args:
        A: NDArray.
        k: The target index.
    """
    var left = 0
    var right = A.size - 1
    while left < right:
        var pivot_index = left + (right - left) // 2
        pivot_index = _partition_in_range(A, left, right, pivot_index)
        if k == pivot_index:
            break
        elif k < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1


def _quickselect_inplace[
    dtype: DType
](mut A: NDArray[dtype], mut I: NDArray[DType.int], k: Int) raises:
    """
    Partitions in-place the data buffer (quickselect) so that index `k`
    holds the value it would have in a fully sorted array. The indices
    are also partitioned.
    Auxiliary function for `argpartition`.

    Args:
        A: NDArray.
        I: NDArray used to store indices.
        k: The target index.
    """
    var left = 0
    var right = A.size - 1
    while left < right:
        var pivot_index = left + (right - left) // 2
        pivot_index = _partition_in_range(A, I, left, right, pivot_index)
        if k == pivot_index:
            break
        elif k < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1


def _quick_sort_in_range(mut A: NDArray, left: Int, right: Int) raises:
    """
    Sort in-place of the data buffer (quick-sort) within give range.
    It is not guaranteed to be stable.

    Args:
        A: NDArray.
        left: Left index of the partition.
        right: Right index of the partition.
    """

    if right > left:
        var pivot_index = left + (right - left) // 2
        var pivot_new_index = _partition_in_range(A, left, right, pivot_index)
        _quick_sort_in_range(A, left, pivot_new_index - 1)
        _quick_sort_in_range(A, pivot_new_index + 1, right)


def _quick_sort_in_range(
    mut A: NDArray, mut I: NDArray, left: Int, right: Int
) raises:
    """
    Sort in-place of the data buffer (quick-sort) within give range.
    The indices are also sorted.
    It is not guaranteed to be stable.

    Args:
        A: NDArray.
        I: NDArray used to store indices.
        left: Left index of the partition.
        right: Right index of the partition.
    """

    if right > left:
        var pivot_index = left + (right - left) // 2
        var pivot_new_index = _partition_in_range(
            A, I, left, right, pivot_index
        )
        _quick_sort_in_range(A, I, left, pivot_new_index - 1)
        _quick_sort_in_range(A, I, pivot_new_index + 1, right)


def _quick_sort_inplace[dtype: DType](mut A: NDArray[dtype]) raises:
    """
    Sort in-place array's buffer using quick sort method.
    It is not guaranteed to be unstable.
    The data buffer must be contiguous.

    Raises:
        NumojoError: If the array is not contiguous.

    Parameters:
        dtype: The input element type.

    Args:
        A: NDArray to sort.
    """

    if not A.flags.FORC:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "\nError in `_quick_sort_inplace`:"
                    "The array must be contiguous to perform in-place sorting."
                ),
                location="_quick_sort_inplace",
            )
        )

    _quick_sort_in_range(
        A,
        left=0,
        right=A.size - 1,
    )


def _quick_sort_inplace[
    dtype: DType
](mut A: NDArray[dtype], mut I: NDArray[DType.int]) raises:
    """
    Sort in-place array's buffer using quick sort method.
    The indices are also sorted.
    It is not guaranteed to be unstable.
    The data buffer must be contiguous.

    Raises:
        NumojoError: If the array is not contiguous.

    Parameters:
        dtype: The input element type.

    Args:
        A: NDArray to sort.
        I: NDArray that stores the indices.
    """

    if not A.flags.FORC:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "\nError in `_quick_sort_inplace`:"
                    "The array must be contiguous to perform in-place sorting."
                ),
                location="_quick_sort_inplace",
            )
        )

    _quick_sort_in_range(
        A,
        I,
        left=0,
        right=A.size - 1,
    )


def _quick_sort_stable_inplace[
    dtype: DType, //
](mut a: NDArray[dtype], size: Int) raises:
    """
    Sort in-place array's buffer using quick sort method.
    The data buffer must be contiguous.
    The sorting is stable

    Raises:
        NumojoError: If the array is not contiguous.

    Parameters:
        dtype: The input element type.

    Args:
        a: NDArray to sort.
        size: The size of the array.
    """

    if size <= 1:
        return

    if not a.flags.FORC:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "\nError in `_quick_sort_stable_inplace`:"
                    "The array must be contiguous to perform in-place sorting."
                ),
                location="_quick_sort_stable_inplace",
            )
        )

    var pivot_index = size // 2
    var pivot_value = a.unsafe_get(pivot_index)

    var left = NDArray[dtype](shape=NDArrayShape(size), order="C")
    var right = NDArray[dtype](shape=NDArrayShape(size), order="C")
    var left_index = 0
    var right_index = 0

    # Put items to either left or right arrays
    for i in range(size):
        if i != pivot_index:
            var value = a.unsafe_get(i)
            if value < pivot_value:
                left.unsafe_set(left_index, value)
                left_index += 1
            elif value > pivot_value:
                right.unsafe_set(right_index, value)
                right_index += 1
            else:  # value == pivot_value
                if i < pivot_index:
                    left.unsafe_set(left_index, value)
                    left_index += 1
                else:
                    right.unsafe_set(right_index, value)
                    right_index += 1

    # Sort left and right arrays
    _quick_sort_stable_inplace(left, left_index)
    _quick_sort_stable_inplace(right, right_index)

    # Combine the sorted arrays
    for i in range(left_index):
        a.unsafe_set(i, left.unsafe_get(i))
    a.unsafe_set(left_index, pivot_value)
    for i in range(right_index):
        a.unsafe_set(left_index + 1 + i, right.unsafe_get(i))
