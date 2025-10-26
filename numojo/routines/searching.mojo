# ===----------------------------------------------------------------------=== #
# Searching
# ===----------------------------------------------------------------------=== #

import builtin.math as builtin_math
import math
from algorithm import vectorize
from sys import simd_width_of
from collections.optional import Optional

from numojo.core.ndarray import NDArray
from numojo.core.ndshape import NDArrayShape
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix
from numojo.core.utility import is_inttype, is_floattype
from numojo.routines.sorting import binary_sort
from numojo.routines.math.extrema import _max, _min


fn argmax_1d[dtype: DType](a: NDArray[dtype]) raises -> Scalar[DType.int]:
    """Returns the index of the maximum value in the buffer.
    Regardless of the shape of input, it is treated as a 1-d array.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        The index of the maximum value in the buffer.
    """

    var ptr = a._buf.ptr
    var value = ptr[]
    var result: Int = 0

    for i in range(a.size):
        if ptr[] > value:
            result = i
            value = ptr[]
        ptr += 1

    return result


fn argmin_1d[dtype: DType](a: NDArray[dtype]) raises -> Scalar[DType.int]:
    """Returns the index of the minimum value in the buffer.
    Regardless of the shape of input, it is treated as a 1-d array.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        The index of the minimum value in the buffer.
    """

    var ptr = a._buf.ptr
    var value = ptr[]
    var result: Int = 0

    for i in range(a.size):
        if ptr[] < value:
            result = i
            value = ptr[]
        ptr += 1

    return result


fn argmax[dtype: DType, //](a: NDArray[dtype]) raises -> Scalar[DType.int]:
    """Returns the indices of the maximum values of the array along an axis.
    When no axis is specified, the array is flattened.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        Returns the indices of the maximum values of the array along an axis.

    Notes:

    If there are multiple occurrences of the maximum values, the indices
    of the first occurrence are returned.
    """

    if a.ndim == 1:
        return argmax_1d(a)
    else:
        return argmax_1d(ravel(a))


fn argmax[
    dtype: DType, //
](a: NDArray[dtype], axis: Int) raises -> NDArray[DType.int]:
    """Returns the indices of the maximum values of the array along an axis.
    When no axis is specified, the array is flattened.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.
        axis: The axis along which to operate.

    Returns:
        Returns the indices of the maximum values of the array along an axis.

    Notes:

    If there are multiple occurrences of the maximum values, the indices
    of the first occurrence are returned.

    Examples:

    ```mojo
    from numojo.prelude import *
    from python import Python

    fn main() raises:
        var np = Python.import_module("numpy")
        # Test with argmax to get maximum values
        var a = nm.random.randint(5, 4, low=0, high=10)
        var a_np = a.to_numpy()
        print(a)
        print(a_np)
        # Get indices of maximum values along axis=1
        var max_indices = nm.argmax(a, axis=1)
        var max_indices_np = np.argmax(a_np, axis=1)
        # Reshape indices for take_along_axis
        var reshaped_indices = max_indices.reshape(Shape(max_indices.shape[0], 1))
        var reshaped_indices_np = max_indices_np.reshape(max_indices_np.shape[0], 1)
        print(reshaped_indices)
        print(reshaped_indices_np)
        # Get maximum values using take_along_axis
        print(nm.indexing.take_along_axis(a, reshaped_indices, axis=1))
        print(np.take_along_axis(a_np, reshaped_indices_np, axis=1))
    ```
    End of examples.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            String("Error in `argmax`: Axis {} not in bound [-{}, {})").format(
                axis, a.ndim, a.ndim
            )
        )

    return numojo.apply_along_axis[func1d=argmax_1d](a=a, axis=normalized_axis)


@always_inline
fn find_extrema_index[
    dtype: DType, find_max: Bool
](A: Matrix[dtype]) raises -> Scalar[DType.int]:
    """Find index of min/max value, either in whole matrix or along an axis."""

    var extreme_val = A[0, 0]
    var extreme_idx: Scalar[DType.int] = 0

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            var current = A[i, j]
            var linear_idx = i * A.shape[1] + j

            if find_max:
                if current > extreme_val:
                    extreme_val = current
                    extreme_idx = linear_idx
            else:
                if current < extreme_val:
                    extreme_val = current
                    extreme_idx = linear_idx

    return extreme_idx


@always_inline
fn find_extrema_index[
    dtype: DType, find_max: Bool
](A: Matrix[dtype], axis: Optional[Int]) raises -> Matrix[DType.int]:
    """Find index of min/max value, either in whole matrix or along an axis."""

    if axis != 0 and axis != 1:
        raise Error(String("The axis can either be 1 or 0!"))

    var B = Matrix[DType.int](
        shape=(A.shape[0], 1) if axis == 1 else (1, A.shape[1])
    )

    if axis == 1:
        for i in range(A.shape[0]):
            var extreme_val = A[i, 0]
            var extreme_idx = 0

            for j in range(1, A.shape[1]):
                var current = A[i, j]

                if find_max:
                    if current > extreme_val:
                        extreme_val = current
                        extreme_idx = j
                else:
                    if current < extreme_val:
                        extreme_val = current
                        extreme_idx = j

            B[i, 0] = extreme_idx
    else:
        for j in range(A.shape[1]):
            var extreme_val = A[0, j]
            var extreme_idx = 0

            for i in range(1, A.shape[0]):
                var current = A[i, j]

                if find_max:
                    if current > extreme_val:
                        extreme_val = current
                        extreme_idx = i
                else:
                    if current < extreme_val:
                        extreme_val = current
                        extreme_idx = i

            B[0, j] = extreme_idx

    return B^


fn argmax[dtype: DType](A: Matrix[dtype]) raises -> Scalar[DType.int]:
    """Find index of max value in a flattened matrix."""
    return find_extrema_index[dtype, True](A)


fn argmax[
    dtype: DType
](A: Matrix[dtype], axis: Int) raises -> Matrix[DType.int]:
    """Find indices of max values along the given axis."""
    return find_extrema_index[dtype, True](A, axis)


fn argmin[dtype: DType, //](a: NDArray[dtype]) raises -> Scalar[DType.int]:
    """Returns the indices of the minimum values of the array along an axis.
    When no axis is specified, the array is flattened.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        Returns the indices of the minimum values of the array along an axis.

    Notes:

    If there are multiple occurrences of the minimum values, the indices
    of the first occurrence are returned.
    """

    if a.ndim == 1:
        return argmin_1d(a)
    else:
        return argmin_1d(ravel(a))


fn argmin[
    dtype: DType, //
](a: NDArray[dtype], axis: Int) raises -> NDArray[DType.int]:
    """Returns the indices of the minimum values of the array along an axis.
    When no axis is specified, the array is flattened.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.
        axis: The axis along which to operate.

    Returns:
        Returns the indices of the minimum values of the array along an axis.

    Notes:

    If there are multiple occurrences of the minimum values, the indices
    of the first occurrence are returned.
    """

    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            String("Error in `argmin`: Axis {} not in bound [-{}, {})").format(
                axis, a.ndim, a.ndim
            )
        )

    return numojo.apply_along_axis[func1d=argmin_1d](a=a, axis=normalized_axis)


fn argmin[dtype: DType](A: Matrix[dtype]) raises -> Scalar[DType.int]:
    """
    Index of the min. It is first flattened before sorting.
    """
    return find_extrema_index[dtype, False](A)


fn argmin[
    dtype: DType
](A: Matrix[dtype], axis: Int) raises -> Matrix[DType.int]:
    """
    Index of the min along the given axis.
    """
    return find_extrema_index[dtype, False](A, axis)
