"""
Indexing routines.
"""

from memory import memcpy
from sys import simdwidthof
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

    var normalized_axis = axis
    if normalized_axis < 0:
        normalized_axis = a.ndim + normalized_axis
    if (normalized_axis >= a.ndim) or (normalized_axis < 0):
        raise Error(
            String(
                "\nError in `compress`: Axis {} is out of bound for array with"
                " {} dimensions".format(axis, a.ndim)
            )
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
                " axis {} with size {}".format(
                    condition.size, axis, a.shape[normalized_axis]
                )
            )
        )

    var number_of_true: Int = 0
    for i in range(condition.size):
        number_of_true += Int(condition._buf.ptr[i])

    if number_of_true == 0:
        raise Error(
            String("\nError in `compress`: Condition contains no True values.")
        )

    var shape_of_res = a.shape
    shape_of_res[normalized_axis] = number_of_true

    var res = NDArray[dtype](Shape(shape_of_res))
    var res_strides = NDArrayStrides(ndim=res.ndim, initialized=False)
    var temp = 1
    for i in range(res.ndim - 1, -1, -1):
        if i != normalized_axis:
            (res_strides._buf + i).init_pointee_copy(temp)
            temp *= res.shape[i]
    (res_strides._buf + normalized_axis).init_pointee_copy(temp)

    var iterator = a.iter_over_dimension(normalized_axis)

    var count = 0
    for i in range(len(condition)):
        if condition.item(i):
            var current_slice = iterator.ith(i)
            for offset in range(current_slice.size):
                var remainder = count

                var item = Item(ndim=res.ndim, initialized=False)

                # First along the axis
                var j = normalized_axis
                (item._buf + j).init_pointee_copy(
                    remainder // res_strides._buf[j]
                )
                remainder %= res_strides._buf[j]

                # Then along other axes
                for j in range(res.ndim):
                    if j != normalized_axis:
                        (item._buf + j).init_pointee_copy(
                            remainder // res_strides._buf[j]
                        )
                        remainder %= res_strides._buf[j]

                (
                    res._buf.ptr + utility._get_offset(item, res.strides)
                ).init_pointee_copy(current_slice._buf.ptr[offset])

                count += 1

    return res


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
