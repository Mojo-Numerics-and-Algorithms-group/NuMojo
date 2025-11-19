# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Sorting.
"""
# ===----------------------------------------------------------------------=== #
# SECTIONS OF THIS FILE:
# 1. `sort` and `argsort` functions exposed to users.
# 2. Backend multiple sorting methods that can be used in `sort`.
#     - Binary sort.
#     - Bubble sort.
#     - Quick sort (instable).
#
# TODO: Add more sorting algorithms.
# ===----------------------------------------------------------------------=== #


import math
from algorithm import vectorize

from numojo.core.ndarray import NDArray
from numojo.core.own_data import OwnData
from numojo.core.ndshape import NDArrayShape
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix, MatrixImpl
import numojo.core.utility as utility
from numojo.routines.manipulation import ravel, transpose


# ===----------------------------------------------------------------------=== #
# Sorting functions exposed to users
# ===----------------------------------------------------------------------=== #


# Below are overrides for NDArray type
fn sort[
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


fn sort[
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
            String("Error in `mean`: Axis {} not in bound [-{}, {})").format(
                axis, a.ndim, a.ndim
            )
        )

    if (a.ndim == 1) and (normalized_axis == 0):
        if stable:
            return quick_sort_stable_1d(a)
        else:
            return quick_sort_1d(a)

    if stable:
        return numojo.apply_along_axis[func1d=quick_sort_stable_1d](
            a, axis=normalized_axis
        )
    else:
        return numojo.apply_along_axis[func1d=quick_sort_1d](
            a, axis=normalized_axis
        )


fn sort_inplace[
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
            String("Error in `mean`: Axis {} not in bound [-{}, {})").format(
                axis, a.ndim, a.ndim
            )
        )

    if (a.ndim == 1) and (normalized_axis == 0):
        if stable:
            quick_sort_stable_inplace_1d(a)
        else:
            quick_sort_inplace_1d(a)

    if stable:
        numojo.apply_along_axis[func1d=quick_sort_stable_inplace_1d](
            a, axis=normalized_axis
        )
    else:
        numojo.apply_along_axis[func1d=quick_sort_inplace_1d](
            a, axis=normalized_axis
        )


# Below are overrides for Matrix type


fn sort[dtype: DType](A: Matrix[dtype]) raises -> Matrix[dtype]:
    """
    Sort the Matrix. It is first flattened before sorting.
    """
    var I = Matrix[DType.int].zeros(shape=A.shape)
    var B = A.flatten()
    _quick_sort_inplace(B, I, 0, A.size - 1)

    return B^


fn sort[dtype: DType](var A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Sort the Matrix along the given axis.
    """
    var order = A.order()

    if axis == 1:
        var result = Matrix[dtype](shape=A.shape, order=order)

        for i in range(A.shape[0]):
            var row = Matrix[dtype](shape=(1, A.shape[1]), order="C")
            var indices = Matrix[DType.int].zeros(
                shape=(1, A.shape[1]), order="C"
            )

            for j in range(A.shape[1]):
                row._store(0, j, A._load(i, j))

            _quick_sort_inplace(row, indices, 0, row.size - 1)

            for j in range(A.shape[1]):
                result._store(i, j, row._load(0, j))

        return result^

    elif axis == 0:
        var result = Matrix[dtype](shape=A.shape, order=order)

        for j in range(A.shape[1]):
            var col = Matrix[dtype](shape=(A.shape[0], 1), order="C")
            var indices = Matrix[DType.int].zeros(
                shape=(A.shape[0], 1), order="C"
            )

            for i in range(A.shape[0]):
                col._store(i, 0, A._load(i, j))

            _quick_sort_inplace(col, indices, 0, col.size - 1)

            for i in range(A.shape[0]):
                result._store(i, j, col._load(i, 0))

        return result^
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn argsort[dtype: DType](a: NDArray[dtype]) raises -> NDArray[DType.int]:
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

    if a.ndim == 1:
        a_flattened = a.copy()
    else:
        a_flattened = ravel(a)

    var indices = arange[DType.int](a_flattened.size)

    _quick_sort_inplace(a_flattened, indices)

    return indices^


fn argsort[
    dtype: DType
](mut a: NDArray[dtype], axis: Int) raises -> NDArray[DType.int]:
    """
    Returns the indices that would sort an array.
    It is not guaranteed to be unstable.
    When no axis is given, the array is flattened before sorting.

    Raises:
        Error: If the axis is out of bound.

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
            String("Error in `mean`: Axis {} not in bound [-{}, {})").format(
                axis, a.ndim, a.ndim
            )
        )

    if (a.ndim == 1) and (normalized_axis == 0):
        return argsort_quick_sort_1d(a)

    return numojo.apply_along_axis[func1d=argsort_quick_sort_1d](
        a, axis=normalized_axis
    )


fn argsort[dtype: DType](A: MatrixImpl[dtype, **_]) raises -> Matrix[DType.int]:
    """
    Argsort the Matrix. It is first flattened before sorting.
    """
    var I = Matrix[DType.int](shape=(1, A.size), order=A.order())
    for i in range(I.size):
        I._buf.ptr[i] = i
    var B: Matrix[dtype]
    if A.order() == "C":
        B = A.flatten()
    else:
        B = A.reorder_layout().flatten().reorder_layout()
    _quick_sort_inplace(B, I, 0, A.size - 1)

    return I^


fn argsort[
    dtype: DType
](A: MatrixImpl[dtype, **_], axis: Int) raises -> Matrix[DType.int]:
    """
    Argsort the Matrix along the given axis.
    """
    var order = A.order()

    if axis == 1:
        var result = Matrix[DType.int](shape=A.shape, order=order)

        for i in range(A.shape[0]):
            var row = Matrix[dtype](shape=(1, A.shape[1]), order="C")
            var idx = Matrix[DType.int](shape=(1, A.shape[1]), order="C")

            for j in range(A.shape[1]):
                row._store(0, j, A._load(i, j))
                idx._store(0, j, j)

            _quick_sort_inplace(row, idx, 0, row.size - 1)

            for j in range(A.shape[1]):
                result._store(i, j, idx._load(0, j))

        return result^

    elif axis == 0:
        var result = Matrix[DType.int](shape=A.shape, order=order)

        for j in range(A.shape[1]):
            var col = Matrix[dtype](shape=(A.shape[0], 1), order="C")
            var idx = Matrix[DType.int](shape=(A.shape[0], 1), order="C")

            for i in range(A.shape[0]):
                col._store(i, 0, A._load(i, j))
                idx._store(i, 0, i)

            _quick_sort_inplace(col, idx, 0, col.size - 1)

            for i in range(A.shape[0]):
                result._store(i, j, idx._load(i, 0))

        return result^
    else:
        raise Error(String("The axis can either be 1 or 0!"))


# ===----------------------------------------------------------------------=== #
# Multiple sorting algorithms in the backend.
# ===----------------------------------------------------------------------=== #


###############
# Binary sort #
###############


fn binary_sort_1d[dtype: DType](a: NDArray[dtype]) raises -> NDArray[dtype]:
    var result: NDArray[dtype] = a.copy()
    for end in range(result.size, 1, -1):
        for i in range(1, end):
            if result._buf.ptr[i - 1] > result._buf.ptr[i]:
                var temp = result._buf.ptr[i - 1]
                result._buf.ptr[i - 1] = result._buf.ptr[i]
                result._buf.ptr[i] = temp
    return result^


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
    # * We can make it into a in place operation to avoid copy.
    var result: NDArray[dtype] = ndarray.copy()
    var length: Int = ndarray.size

    for i in range(length):
        for j in range(length - i - 1):
            if result._buf.ptr.load[width=1](j) > result._buf.ptr.load[width=1](
                j + 1
            ):
                var temp = result._buf.ptr.load[width=1](j)
                result._buf.ptr.store(j, result._buf.ptr.load[width=1](j + 1))
                result._buf.ptr.store(j + 1, temp)

    return result^


##############
# Quick sort #
##############


fn quick_sort_1d[dtype: DType](a: NDArray[dtype]) raises -> NDArray[dtype]:
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
        result = a.copy()
    else:
        result = ravel(a)

    _quick_sort_inplace(result)

    return result^


fn quick_sort_stable_1d[
    dtype: DType
](a: NDArray[dtype]) raises -> NDArray[dtype]:
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
        result = a.copy()
    else:
        result = ravel(a)

    _quick_sort_stable_inplace(result, result.size)

    return result^


fn quick_sort_inplace_1d[dtype: DType](mut a: NDArray[dtype]) raises:
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
            "Error in `quick_sort_inplace_1d`: "
            "The input array must be 1-d array."
        )
    _quick_sort_inplace(a)
    return


fn quick_sort_stable_inplace_1d[dtype: DType](mut a: NDArray[dtype]) raises:
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
            String(
                "Error in `quick_sort_inplace_1d`: "
                "The input array must be 1-d array."
            )
        )
    _quick_sort_stable_inplace(a, a.size)
    return


fn argsort_quick_sort_1d[
    dtype: DType
](a: NDArray[dtype]) raises -> NDArray[DType.int]:
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

    var result: NDArray[dtype] = a.copy()
    var indices = arange[DType.int](result.size)
    _quick_sort_inplace(result, indices)
    return indices^


fn _partition_in_range(
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

    var pivot_value = A._buf.ptr[pivot_index]

    A._buf.ptr[pivot_index], A._buf.ptr[right] = (
        A._buf.ptr[right],
        A._buf.ptr[pivot_index],
    )

    var store_index = left

    for i in range(left, right):
        if A._buf.ptr[i] < pivot_value:
            A._buf.ptr[store_index], A._buf.ptr[i] = (
                A._buf.ptr[i],
                A._buf.ptr[store_index],
            )
            store_index = store_index + 1

    A._buf.ptr[store_index], A._buf.ptr[right] = (
        A._buf.ptr[right],
        A._buf.ptr[store_index],
    )

    return store_index


fn _partition_in_range(
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

    var pivot_value = A._buf.ptr[pivot_index]

    A._buf.ptr[pivot_index], A._buf.ptr[right] = (
        A._buf.ptr[right],
        A._buf.ptr[pivot_index],
    )
    I._buf.ptr[pivot_index], I._buf.ptr[right] = (
        I._buf.ptr[right],
        I._buf.ptr[pivot_index],
    )

    var store_index = left

    for i in range(left, right):
        if A._buf.ptr[i] < pivot_value:
            A._buf.ptr[store_index], A._buf.ptr[i] = (
                A._buf.ptr[i],
                A._buf.ptr[store_index],
            )
            I._buf.ptr[store_index], I._buf.ptr[i] = (
                I._buf.ptr[i],
                I._buf.ptr[store_index],
            )
            store_index = store_index + 1

    A._buf.ptr[store_index], A._buf.ptr[right] = (
        A._buf.ptr[right],
        A._buf.ptr[store_index],
    )
    I._buf.ptr[store_index], I._buf.ptr[right] = (
        I._buf.ptr[right],
        I._buf.ptr[store_index],
    )

    return store_index


fn _quick_sort_partition(
    mut A: Matrix, mut I: Matrix, left: Int, right: Int, pivot_index: Int
) raises -> Int:
    """
    Do partition for the data buffer of Matrix.

    Args:
        A: A Matrix.
        I: A Matrix used to store indices.
        left: Left index of the partition.
        right: Right index of the partition.
        pivot_index: Input pivot index

    Returns:
        New pivot index.
    """

    # Boundary check due to use of unsafe way.
    if (left >= A.size) or (right >= A.size) or (pivot_index >= A.size):
        raise Error(
            String(
                "Index out of boundary! "
                "left={}, right={}, pivot_index={}, matrix.size={}"
            ).format(left, right, pivot_index, A.size)
        )

    var pivot_value = A._buf.ptr[pivot_index]

    A._buf.ptr[pivot_index], A._buf.ptr[right] = (
        A._buf.ptr[right],
        A._buf.ptr[pivot_index],
    )
    I._buf.ptr[pivot_index], I._buf.ptr[right] = (
        I._buf.ptr[right],
        I._buf.ptr[pivot_index],
    )

    var store_index = left

    for i in range(left, right):
        if A._buf.ptr[i] < pivot_value:
            A._buf.ptr[store_index], A._buf.ptr[i] = (
                A._buf.ptr[i],
                A._buf.ptr[store_index],
            )
            I._buf.ptr[store_index], I._buf.ptr[i] = (
                I._buf.ptr[i],
                I._buf.ptr[store_index],
            )
            store_index = store_index + 1

    A._buf.ptr[store_index], A._buf.ptr[right] = (
        A._buf.ptr[right],
        A._buf.ptr[store_index],
    )
    I._buf.ptr[store_index], I._buf.ptr[right] = (
        I._buf.ptr[right],
        I._buf.ptr[store_index],
    )

    return store_index


fn _quick_sort_in_range(mut A: NDArray, left: Int, right: Int) raises:
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


fn _quick_sort_in_range(
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


fn _quick_sort_inplace(
    mut A: Matrix, mut I: Matrix, left: Int, right: Int
) raises:
    """
    Sort in-place of the data buffer (quick-sort).
    It is not guaranteed to be stable.
    The data buffer must be contiguous.

    Args:
        A: A Matrix.
        I: A Matrix used to store indices.
        left: Left index of the partition.
        right: Right index of the partition.
    """

    if right > left:
        var pivot_index = left + (right - left) // 2
        var pivot_new_index = _quick_sort_partition(
            A, I, left, right, pivot_index
        )
        _quick_sort_inplace(A, I, left, pivot_new_index - 1)
        _quick_sort_inplace(A, I, pivot_new_index + 1, right)


fn _quick_sort_inplace[dtype: DType](mut A: NDArray[dtype]) raises:
    """
    Sort in-place array's buffer using quick sort method.
    It is not guaranteed to be unstable.
    The data buffer must be contiguous.

    Raises:
        Error: If the array is not contiguous.

    Parameters:
        dtype: The input element type.

    Args:
        A: NDArray to sort.
    """

    if not A.flags.FORC:
        raise Error(
            String(
                "\nError in `_quick_sort_inplace`:"
                "The array must be contiguous to perform in-place sorting."
            )
        )

    _quick_sort_in_range(
        A,
        left=0,
        right=A.size - 1,
    )


fn _quick_sort_inplace[
    dtype: DType
](mut A: NDArray[dtype], mut I: NDArray[DType.int]) raises:
    """
    Sort in-place array's buffer using quick sort method.
    The indices are also sorted.
    It is not guaranteed to be unstable.
    The data buffer must be contiguous.

    Raises:
        Error: If the array is not contiguous.

    Parameters:
        dtype: The input element type.

    Args:
        A: NDArray to sort.
        I: NDArray that stores the indices.
    """

    if not A.flags.FORC:
        raise Error(
            String(
                "\nError in `_quick_sort_inplace`:"
                "The array must be contiguous to perform in-place sorting."
            )
        )

    _quick_sort_in_range(
        A,
        I,
        left=0,
        right=A.size - 1,
    )


fn _quick_sort_stable_inplace[
    dtype: DType, //
](mut a: NDArray[dtype], size: Int) raises:
    """
    Sort in-place array's buffer using quick sort method.
    The data buffer must be contiguous.
    The sorting is stable

    Raises:
        Error: If the array is not contiguous.

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
            String(
                "\nError in `_quick_sort_stable_inplace`:"
                "The array must be contiguous to perform in-place sorting."
            )
        )

    var pivot_index = size // 2
    var pivot_value = a._buf.ptr[pivot_index]

    var left = NDArray[dtype](shape=(size), order="C")
    var right = NDArray[dtype](shape=(size), order="C")
    var left_index = 0
    var right_index = 0

    # Put items to either left or right arrays
    for i in range(size):
        if i != pivot_index:
            var value = a._buf.ptr[i]
            if value < pivot_value:
                left._buf.ptr[left_index] = value
                left_index += 1
            elif value > pivot_value:
                right._buf.ptr[right_index] = value
                right_index += 1
            else:  # value == pivot_value
                if i < pivot_index:
                    left._buf.ptr[left_index] = value
                    left_index += 1
                else:
                    right._buf.ptr[right_index] = value
                    right_index += 1

    # Sort left and right arrays
    _quick_sort_stable_inplace(left, left_index)
    _quick_sort_stable_inplace(right, right_index)

    # Combine the sorted arrays
    for i in range(left_index):
        a._buf.ptr[i] = left._buf.ptr[i]
    a._buf.ptr[left_index] = pivot_value
    for i in range(right_index):
        a._buf.ptr[left_index + 1 + i] = right._buf.ptr[i]
