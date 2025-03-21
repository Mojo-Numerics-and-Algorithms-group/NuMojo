# ===----------------------------------------------------------------------=== #
# Searching
# ===----------------------------------------------------------------------=== #

import builtin.math as builtin_math
import math
from algorithm import vectorize
from sys import simdwidthof
from collections.optional import Optional

from numojo.core.ndarray import NDArray
from numojo.core.ndshape import NDArrayShape
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix
from numojo.core.utility import is_inttype, is_floattype
from numojo.routines.sorting import binary_sort
from numojo.routines.math.extrema import _max, _min


fn argmax_1d[dtype: DType](a: NDArray[dtype]) raises -> Scalar[DType.index]:
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


fn argmin_1d[dtype: DType](a: NDArray[dtype]) raises -> Scalar[DType.index]:
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


fn argmax[dtype: DType, //](a: NDArray[dtype]) raises -> Scalar[DType.index]:
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
](a: NDArray[dtype], axis: Int) raises -> NDArray[DType.index]:
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


fn argmax[dtype: DType](A: Matrix[dtype]) raises -> Scalar[DType.index]:
    """
    Index of the max. It is first flattened before sorting.
    """

    var max_index: Scalar[DType.index]
    _, max_index = _max(A, 0, A.size - 1)

    return max_index


fn argmax[
    dtype: DType
](A: Matrix[dtype], axis: Int) raises -> Matrix[DType.index]:
    """
    Index of the max along the given axis.
    """
    if axis == 1:
        var B = Matrix[DType.index](shape=(A.shape[0], 1))
        for i in range(A.shape[0]):
            B._store(
                i,
                0,
                _max(A, start=i * A.strides[0], end=(i + 1) * A.strides[0] - 1)[
                    1
                ]
                - i * A.strides[0],
            )
        return B^
    elif axis == 0:
        return transpose(argmax(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn argmin[dtype: DType, //](a: NDArray[dtype]) raises -> Scalar[DType.index]:
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
](a: NDArray[dtype], axis: Int) raises -> NDArray[DType.index]:
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


fn argmin[dtype: DType](A: Matrix[dtype]) raises -> Scalar[DType.index]:
    """
    Index of the min. It is first flattened before sorting.
    """

    var min_index: Scalar[DType.index]
    _, min_index = _min(A, 0, A.size - 1)

    return min_index


fn argmin[
    dtype: DType
](A: Matrix[dtype], axis: Int) raises -> Matrix[DType.index]:
    """
    Index of the min along the given axis.
    """
    if axis == 1:
        var B = Matrix[DType.index](shape=(A.shape[0], 1))
        for i in range(A.shape[0]):
            B._store(
                i,
                0,
                _min(A, start=i * A.strides[0], end=(i + 1) * A.strides[0] - 1)[
                    1
                ]
                - i * A.strides[0],
            )
        return B^
    elif axis == 0:
        return transpose(argmin(transpose(A), axis=1))
    else:
        raise Error(String("The axis can either be 1 or 0!"))
