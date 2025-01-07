# ===----------------------------------------------------------------------=== #
# Sorting
# ===----------------------------------------------------------------------=== #


import math
from algorithm import vectorize

from numojo.core.ndarray import NDArray
from numojo.core.ndshape import NDArrayShape
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix
from numojo.routines.manipulation import ravel, transpose

"""
TODO:
1) Add more sorting algorithms.
2) Add axis.
"""

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


fn _partition_in_range(
    mut A: NDArray,
    mut I: NDArray,
    left: Int,
    right: Int,
    pivot_index: Int,
) raises -> Int:
    """
    Do in-place partition for array buffer within given range.
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

    # (Unsafe) Boundary checks are not done for sake of speed:
    # if (left >= A.size) or (right >= A.size) or (pivot_index >= A.size):

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

    var pivot_value = A._buf[pivot_index]

    A._buf[pivot_index], A._buf[right] = A._buf[right], A._buf[pivot_index]
    I._buf[pivot_index], I._buf[right] = I._buf[right], I._buf[pivot_index]

    var store_index = left

    for i in range(left, right):
        if A._buf[i] < pivot_value:
            A._buf[store_index], A._buf[i] = A._buf[i], A._buf[store_index]
            I._buf[store_index], I._buf[i] = I._buf[i], I._buf[store_index]
            store_index = store_index + 1

    A._buf[store_index], A._buf[right] = A._buf[right], A._buf[store_index]
    I._buf[store_index], I._buf[right] = I._buf[right], I._buf[store_index]

    return store_index


fn _sort_in_range(mut A: NDArray, mut I: NDArray, left: Int, right: Int) raises:
    """
    Sort in-place of the data buffer (quick-sort) within give range.
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

    var array_order = "C" if A.flags["C_CONTIGUOUS"] else "F"
    var continous_axis = A.ndim - 1 if array_order == "C" else A.ndim - 2
    """Continuously stored axis. -1 if row-major, -2 if col-major."""

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


fn sort[dtype: DType](owned A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Sort NDArray using quick sort method.
    It is not guaranteed to be unstable.

    When no axis is given, the array is flattened before sorting.

    Parameters:
        dtype: The input element type.

    Args:
        A: NDArray.
    """

    A = ravel(A)
    var _I = NDArray[DType.index](A.shape)
    _sort_inplace(A, _I, axis=0)
    return A^


fn sort[
    dtype: DType
](owned A: NDArray[dtype], owned axis: Int) raises -> NDArray[dtype]:
    """
    Sort NDArray along the given axis using quick sort method.
    It is not guaranteed to be unstable.

    When no axis is given, the array is flattened before sorting.

    Parameters:
        dtype: The input element type.

    Args:
        A: NDArray to sort.
        axis: The axis along which the array is sorted.

    """

    var _I = NDArray[DType.index](A.shape)
    _sort_inplace(A, _I, axis)
    return A^


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


###############
# Binary sort #
###############


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


# ===----------------------------------------------------------------------=== #
# Searching
# ===----------------------------------------------------------------------=== #


fn argsort[
    dtype: DType
](owned A: NDArray[dtype]) raises -> NDArray[DType.index]:
    """
    Returns the indices that would sort an array.
    It is not guaranteed to be unstable.

    When no axis is given, the array is flattened before sorting.

    Parameters:
        dtype: The input element type.

    Args:
        A: NDArray.

    Returns:
        Indices that would sort an array.
    """

    A = ravel(A)
    var I = NDArray[DType.index](A.shape)
    _sort_inplace(A, I, axis=0)
    return I^


fn argsort[
    dtype: DType
](owned A: NDArray[dtype], owned axis: Int) raises -> NDArray[DType.index]:
    """
    Returns the indices that would sort an array.
    It is not guaranteed to be unstable.

    When no axis is given, the array is flattened before sorting.

    Parameters:
        dtype: The input element type.

    Args:
        A: NDArray to sort.
        axis: The axis along which the array is sorted.

    Returns:
        Indices that would sort an array.

    """

    var I = NDArray[DType.index](A.shape)
    _sort_inplace(A, I, axis)
    return I^


fn argsort[dtype: DType](A: Matrix[dtype]) raises -> Matrix[DType.index]:
    """
    Argsort the Matrix. It is first flattened before sorting.
    """
    var I = Matrix[DType.index](shape=(1, A.size))
    for i in range(I.size):
        I._buf[i] = i
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
