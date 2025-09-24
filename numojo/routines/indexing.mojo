# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Implement indexing routines.

- Generating index arrays
- Indexing-like operations
- Inserting data into arrays
- Iterating over arrays
"""

from memory import memcpy
from sys import simd_width_of
from algorithm import vectorize
from numojo.core.ndarray import NDArray
from numojo.core.ndstrides import NDArrayStrides
import numojo.core.utility as utility

# ===----------------------------------------------------------------------=== #
# Generating index arrays
# ===----------------------------------------------------------------------=== #


fn where[
    dtype: DType
](
    mut x: NDArray[dtype], scalar: SIMD[dtype, 1], mask: NDArray[DType.bool]
) raises:
    """
    Replaces elements in `x` with `scalar` where `mask` is True.

    Parameters:
        dtype: DType.

    Args:
        x: A NDArray.
        scalar: A SIMD value.
        mask: A NDArray.

    """
    for i in range(x.size):
        if mask._buf.ptr[i] == True:
            x._buf.ptr.store(i, scalar)


# TODO: do it with vectorization
fn where[
    dtype: DType
](mut x: NDArray[dtype], y: NDArray[dtype], mask: NDArray[DType.bool]) raises:
    """
    Replaces elements in `x` with elements from `y` where `mask` is True.

    Raises:
        ShapeMismatchError: If the shapes of `x` and `y` do not match.

    Parameters:
        dtype: DType.

    Args:
        x: NDArray[dtype].
        y: NDArray[dtype].
        mask: NDArray[DType.bool].

    """
    if x.shape != y.shape:
        raise Error("Shape mismatch error: x and y must have the same shape")
    for i in range(x.size):
        if mask._buf.ptr[i] == True:
            x._buf.ptr.store(i, y._buf.ptr[i])


# ===----------------------------------------------------------------------=== #
# Indexing-like operations
# ===----------------------------------------------------------------------=== #


fn compress[
    dtype: DType
](
    condition: NDArray[DType.bool], a: NDArray[dtype], axis: Int
) raises -> NDArray[dtype]:
    # TODO: @forFudan try using parallelization for this function
    """
    Return selected slices of an array along given axis.
    If no axis is provided, the array is flattened before use.

    Raises:
        Error: If the axis is out of bound for the given array.
        Error: If the condition is not 1-D array.
        Error: If the condition length is out of bound for the given axis.
        Error: If the condition contains no True values.

    Parameters:
        dtype: DType.

    Args:
        condition: 1-D array of booleans that selects which entries to return.
            If length of condition is less than the size of the array along the
            given axis, then output is filled to the length of the condition
            with False.
        a: The array.
        axis: The axis along which to take slices.

    Returns:
        An array.
    """

    var normalized_axis: Int = axis
    if normalized_axis < 0:
        normalized_axis = a.ndim + normalized_axis
    if (normalized_axis >= a.ndim) or (normalized_axis < 0):
        raise Error(
            String(
                "\nError in `compress`: Axis {} is out of bound for array with"
                " {} dimensions"
            ).format(axis, a.ndim)
        )

    if condition.ndim != 1:
        raise Error(
            String(
                "\nError in `compress`: Condition must be 1-D array, got {}"
            ).format(condition.ndim)
        )
    if condition.size > a.shape[normalized_axis]:
        raise Error(
            String(
                "\nError in `compress`: Condition length {} is out of bound for"
                " axis {} with size {}"
            ).format(condition.size, axis, a.shape[normalized_axis])
        )

    var number_of_true: Int = 0
    for i in range(condition.size):
        number_of_true += Int(condition._buf.ptr[i])

    if number_of_true == 0:
        raise Error(
            String("\nError in `compress`: Condition contains no True values.")
        )

    var shape_of_res: NDArrayShape = a.shape
    shape_of_res[normalized_axis] = number_of_true

    var result: NDArray[dtype] = NDArray[dtype](Shape(shape_of_res))
    var res_strides: NDArrayStrides = NDArrayStrides(ndim=result.ndim, initialized=False)
    var temp: Int = 1
    for i in range(result.ndim - 1, -1, -1):
        if i != normalized_axis:
            (res_strides._buf + i).init_pointee_copy(temp)
            temp *= result.shape[i]
    (res_strides._buf + normalized_axis).init_pointee_copy(temp)

    var iterator = a.iter_over_dimension(normalized_axis)

    var count: Int = 0
    for i in range(len(condition)):
        if condition.item(i):
            var current_slice = iterator.ith(i)
            for offset in range(current_slice.size):
                var remainder: Int = count

                var item: Item = Item(ndim=result.ndim, initialized=False)

                # First along the axis
                var j = normalized_axis
                (item._buf + j).init_pointee_copy(
                    remainder // res_strides._buf[j]
                )
                remainder %= res_strides._buf[j]

                # Then along other axes
                for j in range(result.ndim):
                    if j != normalized_axis:
                        (item._buf + j).init_pointee_copy(
                            remainder // res_strides._buf[j]
                        )
                        remainder %= res_strides._buf[j]

                (
                    result._buf.ptr + utility._get_offset(item, result.strides)
                ).init_pointee_copy(current_slice._buf.ptr[offset])

                count += 1

    return result^


fn compress[
    dtype: DType
](condition: NDArray[DType.bool], a: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Return selected slices of an array along given axis.
    If no axis is provided, the array is flattened before use.
    This is a function ***OVERLOAD***.

    Raises:
        Error: If the condition is not 1-D array.
        Error: If the condition length is out of bound for the given axis.
        Error: If the condition contains no True values.

    Parameters:
        dtype: DType.

    Args:
        condition: 1-D array of booleans that selects which entries to return.
            If length of condition is less than the size of the array along the
            given axis, then output is filled to the length of the condition
            with False.
        a: The array.

    Returns:
        An array.

    """

    if condition.ndim != 1:
        raise Error(
            String(
                "\nError in `compress`: Condition must be 1-D array, got {}"
            ).format(condition.ndim)
        )

    if a.ndim == 1:
        return compress(condition, a, axis=0)

    else:
        return compress(condition, ravel(a), axis=0)


fn take_along_axis[
    dtype: DType, //,
](
    arr: NDArray[dtype], indices: NDArray[DType.index], axis: Int = 0
) raises -> NDArray[dtype]:
    """
    Takes values from the input array along the given axis based on indices.

    Raises:
        Error: If the axis is out of bounds for the given array.
        Error: If the ndim of arr and indices are not the same.
        Error: If the shape of indices does not match the shape of the
            input array except along the given axis.

    Parameters:
        dtype: DType of the input array.

    Args:
        arr: The source array.
        indices: The indices array.
        axis: The axis along which to take values. Default is 0.

    Returns:
        An array with the same shape as indices with values taken from the
            input array along the given axis.

    Examples:

    ```console
    > var a = nm.arange[i8](12).reshape(Shape(3, 4))
    > print(a)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    > ind = nm.array[intp]("[[0, 1, 2, 0], [1, 0, 2, 1]]")
    > print(ind)
    [[0 1 2 0]
     [1 0 2 1]]
    > print(nm.indexing.take_along_axis(a, ind, axis=0))
    [[ 0  5 10  3]
     [ 4  1 10  7]]
    ```
    .
    """
    var normalized_axis = axis
    if normalized_axis < 0:
        normalized_axis = arr.ndim + normalized_axis
    if (normalized_axis >= arr.ndim) or (normalized_axis < 0):
        raise Error(
            String(
                "\nError in `take_along_axis`: Axis {} is out of bound for"
                " array with {} dimensions"
            ).format(axis, arr.ndim)
        )

    # Check if the ndim of arr and indices are same
    if arr.ndim != indices.ndim:
        raise Error(
            String(
                "\nError in `take_along_axis`: The ndim of arr and indices must"
                " be same. Got {} and {}."
            ).format(arr.ndim, indices.ndim)
        )

    # broadcast indices to the shape of arr if necessary
    # When broadcasting, the shape of indices must match the shape of arr
    # except along the axis

    var broadcasted_indices: NDArray[DType.index] = indices.copy() # make this owned and don't copy

    if arr.shape != indices.shape:
        var arr_shape_new = arr.shape
        arr_shape_new[normalized_axis] = indices.shape[normalized_axis]

        try:
            broadcasted_indices = numojo.broadcast_to(indices, arr_shape_new)
        except e:
            raise Error(
                String(
                    "\nError in `take_along_axis`: Shape of indices must match"
                    " shape of array except along the given axis. "
                    + String(e)
                )
            )

    # Create output array with same shape as broadcasted_indices
    var result = NDArray[dtype](Shape(broadcasted_indices.shape))

    var arr_iterator = arr.iter_along_axis(normalized_axis)
    var indices_iterator = broadcasted_indices.iter_along_axis(normalized_axis)
    var length_of_iterator: Int = result.size // result.shape[normalized_axis]

    if normalized_axis == arr.ndim - 1:
        # If axis is the last axis, the data is contiguous.
        for i in range(length_of_iterator):
            var arr_slice = arr_iterator.ith(i)
            var indices_slice = indices_iterator.ith(i)
            var arr_slice_after_applying_indices: NDArray[dtype] = arr_slice[indices_slice]
            memcpy(
                result._buf.ptr + i * result.shape[normalized_axis],
                arr_slice_after_applying_indices._buf.ptr,
                result.shape[normalized_axis],
            )
    else:
        # If axis is not the last axis, the data is not contiguous.
        for i in range(length_of_iterator):
            var indices_slice_offsets: NDArray[DType.index]
            var indices_slice: NDArray[DType.index]
            var indices_slice_offsets_slice = indices_iterator.ith_with_offsets(i)
            indices_slice_offsets = indices_slice_offsets_slice[0].copy()
            indices_slice = indices_slice_offsets_slice[1].copy()
            var arr_slice = arr_iterator.ith(i)
            var arr_slice_after_applying_indices = arr_slice[indices_slice]
            for j in range(arr_slice_after_applying_indices.size):
                (
                    result._buf.ptr + Int(indices_slice_offsets[j])
                ).init_pointee_copy(
                    arr_slice_after_applying_indices._buf.ptr[j]
                )

    return result^
