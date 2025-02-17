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
from numojo.core.ndshape import NDArrayShape
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix
import numojo.core.utility as utility
from numojo.routines.manipulation import ravel, transpose


# ===----------------------------------------------------------------------=== #
# Sorting functions exposed to users
# ===----------------------------------------------------------------------=== #


fn sort[dtype: DType](a: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Sort NDArray using quick sort method.
    It is not guaranteed to be unstable.
    When no axis is given, the array is flattened before sorting.

    Parameters:
        dtype: The input element type.

    Args:
        a: NDArray.
    """

    return quick_sort_1d(a)


fn sort[
    dtype: DType
](owned a: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Sort NDArray along the given axis using quick sort method.
    It is not guaranteed to be unstable.
    When no axis is given, the array is flattened before sorting.

    Parameters:
        dtype: The input element type.

    Args:
        a: NDArray to sort.
        axis: The axis along which the array is sorted.

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
        return quick_sort_1d(a)

    return utility.apply_func_on_array_without_dim_reduction[
        func=quick_sort_1d
    ](a, axis=normalized_axis)


fn sort[dtype: DType](A: Matrix[dtype]) raises -> Matrix[dtype]:
    """
    Sort the Matrix. It is first flattened before sorting.
    """
    var I = Matrix.zeros[DType.index](shape=A.shape)
    var B = A.flatten()
    _sort_inplace(B, I, 0, A.size - 1)

    return B^


fn sort[
    dtype: DType
](owned A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Sort the Matrix along the given axis.
    """
    if axis == 1:
        var I = Matrix.zeros[DType.index](shape=A.shape)
        for i in range(A.shape[0]):
            _sort_inplace(
                A, I, left=i * A.strides[0], right=(i + 1) * A.strides[0] - 1
            )
        return A^
    elif axis == 0:
        return transpose(sort(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn argsort[dtype: DType](a: NDArray[dtype]) raises -> NDArray[DType.index]:
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
        res = a
    else:
        res = ravel(a)

    var indices = arange[DType.index](res.size)

    _sort_inplace(res, indices)

    return indices^


fn argsort[
    dtype: DType
](a: NDArray[dtype], axis: Int) raises -> NDArray[DType.index]:
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

    var normalized_axis = axis
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

    return utility.apply_func_on_array_without_dim_reduction[
        func=argsort_quick_sort_1d
    ](a, axis=normalized_axis)


fn argsort[dtype: DType](A: Matrix[dtype]) raises -> Matrix[DType.index]:
    """
    Argsort the Matrix. It is first flattened before sorting.
    """
    var I = Matrix[DType.index](shape=(1, A.size))
    for i in range(I.size):
        I._buf.ptr[i] = i
    var B = A.flatten()
    _sort_inplace(B, I, 0, A.size - 1)

    return I^


fn argsort[
    dtype: DType
](owned A: Matrix[dtype], axis: Int) raises -> Matrix[DType.index]:
    """
    Argsort the Matrix along the given axis.
    """
    if axis == 1:
        var I = Matrix[DType.index](shape=A.shape)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                I._store(i, j, j)

        for i in range(A.shape[0]):
            _sort_inplace(
                A, I, left=i * A.strides[0], right=(i + 1) * A.strides[0] - 1
            )
        return I^
    elif axis == 0:
        return transpose(argsort(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))


# ===----------------------------------------------------------------------=== #
# Multiple sorting algorithms in the backend.
# ===----------------------------------------------------------------------=== #


###############
# Binary sort #
###############


fn binary_sort_1d[dtype: DType](a: NDArray[dtype]) raises -> NDArray[dtype]:
    var res = a
    for end in range(res.size, 1, -1):
        for i in range(1, end):
            if res._buf.ptr[i - 1] > res._buf.ptr[i]:
                var temp = res._buf.ptr[i - 1]
                res._buf.ptr[i - 1] = res._buf.ptr[i]
                res._buf.ptr[i] = temp
    return res


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

    var n = array.num_elements()
    for end in range(n, 1, -1):
        for i in range(1, end):
            if result[i - 1] > result[i]:
                var temp: Scalar[dtype] = result.load(i - 1)
                result.store(i - 1, result.load(i))
                result.store(i, temp)
    return result


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
    var result: NDArray[dtype] = ndarray
    var length = ndarray.size

    for i in range(length):
        for j in range(length - i - 1):
            if result._buf.ptr.load[width=1](j) > result._buf.ptr.load[width=1](
                j + 1
            ):
                var temp = result._buf.ptr.load[width=1](j)
                result._buf.ptr.store(j, result._buf.ptr.load[width=1](j + 1))
                result._buf.ptr.store(j + 1, temp)

    return result


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
    var res: NDArray[dtype]
    if a.ndim == 1:
        res = a
    else:
        res = ravel(a)

    _sort_inplace(res)

    return res^


fn argsort_quick_sort_1d[
    dtype: DType
](a: NDArray[dtype]) raises -> NDArray[DType.index]:
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

    var res = a
    var indices = arange[DType.index](res.size)
    _sort_inplace(res, indices)
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


fn _sort_partition(
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


fn _sort_in_range(mut A: NDArray, left: Int, right: Int) raises:
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
        _sort_in_range(A, left, pivot_new_index - 1)
        _sort_in_range(A, pivot_new_index + 1, right)


fn _sort_in_range(mut A: NDArray, mut I: NDArray, left: Int, right: Int) raises:
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
        _sort_in_range(A, I, left, pivot_new_index - 1)
        _sort_in_range(A, I, pivot_new_index + 1, right)


fn _sort_inplace(mut A: Matrix, mut I: Matrix, left: Int, right: Int) raises:
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
        var pivot_new_index = _sort_partition(A, I, left, right, pivot_index)
        _sort_inplace(A, I, left, pivot_new_index - 1)
        _sort_inplace(A, I, pivot_new_index + 1, right)


fn _sort_inplace[dtype: DType](mut A: NDArray[dtype]) raises:
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
                "\nError in `_sort_inplace`:"
                "The array must be contiguous to perform in-place sorting."
            )
        )

    _sort_in_range(
        A,
        left=0,
        right=A.size - 1,
    )


fn _sort_inplace[
    dtype: DType
](mut A: NDArray[dtype], mut I: NDArray[DType.index]) raises:
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
                "\nError in `_sort_inplace`:"
                "The array must be contiguous to perform in-place sorting."
            )
        )

    _sort_in_range(
        A,
        I,
        left=0,
        right=A.size - 1,
    )


fn _sort_inplace[
    dtype: DType
](mut A: NDArray[dtype], mut I: NDArray[DType.index], owned axis: Int) raises:
    """
    Sort in-place NDArray along the given axis using quick sort method.
    It is not guaranteed to be unstable.

    Parameters:
        dtype: The input element type.

    Args:
        A: NDArray to sort.
        I: NDArray that stores the indices.
        axis: The axis along which the array is sorted.

    """

    if axis < 0:
        axis = A.ndim + axis

    if (axis >= A.ndim) or (axis < 0):
        raise Error(
            String("Axis {} is invalid for array of {} dimensions").format(
                axis, A.ndim
            )
        )

    var array_order = "C" if A.flags.C_CONTIGUOUS else "F"
    var continous_axis = A.ndim - 1 if array_order == "C" else A.ndim - 2
    """Contiguously stored axis. -1 if row-major, -2 if col-major."""

    if axis == continous_axis:  # Last axis
        I = NDArray[DType.index](shape=A.shape)
        for i in range(A.size // A.shape[continous_axis]):
            for j in range(A.shape[continous_axis]):
                (
                    I._buf.ptr + i * A.shape[continous_axis] + j
                ).init_pointee_copy(j)
            _sort_in_range(
                A,
                I,
                left=i * A.shape[continous_axis],
                right=(i + 1) * A.shape[continous_axis] - 1,
            )
    else:
        var transposed_axes = List[Int](capacity=A.ndim)
        for i in range(A.ndim):
            transposed_axes.append(i)
        transposed_axes[axis], transposed_axes[continous_axis] = (
            transposed_axes[continous_axis],
            transposed_axes[axis],
        )
        A = transpose(A, axes=transposed_axes)
        _sort_inplace(A, I, axis=-1)
        A = transpose(A, axes=transposed_axes)
        I = transpose(I, axes=transposed_axes)
