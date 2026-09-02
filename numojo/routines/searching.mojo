# ===----------------------------------------------------------------------=== #
# NuMojo: Searching routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Searching routines (numojo.routines.searching).
===============================================
Search operations for finding array extrema indices.

Functions for finding indices of maximum and minimum values in arrays.

Exports
-------
- `argmax`: Index of maximum value.
- `argmin`: Index of minimum value.
- `count_nonzero`: Count of non-zero elements.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.layout import NDArrayShape
from numojo.core.ndarray import NDArray
from numojo.routines.creation import zeros
from numojo.routines.functional import apply_along_axis_reduce_to_int
from numojo.routines.manipulation import ravel


def argmax_1d[
    dtype: DType
](a: NDArray[dtype]) capturing raises -> Scalar[DType.int]:
    """Returns the index of the maximum value in the buffer.
    Regardless of the shape of input, it is treated as a 1-d array.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        The index of the maximum value in the buffer.
    """

    if not a.is_c_contiguous():
        return argmax_1d(a.contiguous())

    var ptr = a.unsafe_ptr()
    var value = ptr[]
    var result: Int = 0

    for i in range(a.size):
        if ptr[] > value:
            result = i
            value = ptr[]
        ptr = ptr.unsafe_offset(1)

    return Scalar[DType.int](result)


def argmin_1d[
    dtype: DType
](a: NDArray[dtype]) capturing raises -> Scalar[DType.int]:
    """Returns the index of the minimum value in the buffer.
    Regardless of the shape of input, it is treated as a 1-d array.

    Parameters:
        dtype: The element type.

    Args:
        a: An array.

    Returns:
        The index of the minimum value in the buffer.
    """

    if not a.is_c_contiguous():
        return argmin_1d(a.contiguous())

    var ptr = a.unsafe_ptr()
    var value = ptr[]
    var result: Int = 0

    for i in range(a.size):
        if ptr[] < value:
            result = i
            value = ptr[]
        ptr = ptr.unsafe_offset(1)

    return Scalar[DType.int](result)


def argmax[dtype: DType, //](a: NDArray[dtype]) raises -> Scalar[DType.int]:
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


def argmax[
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

    def main() raises:
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
            NumojoError(
                category="index",
                message=String(
                    "Error in `argmax`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="argmax",
            )
        )

    return apply_along_axis_reduce_to_int[dtype, func1d=argmax_1d](
        a=a, axis=normalized_axis
    )


def argmin[dtype: DType, //](a: NDArray[dtype]) raises -> Scalar[DType.int]:
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


def argmin[
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
            NumojoError(
                category="index",
                message=String(
                    "Error in `argmin`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="argmin",
            )
        )

    return apply_along_axis_reduce_to_int[dtype, func1d=argmin_1d](
        a=a, axis=normalized_axis
    )


def count_nonzero[
    dtype: DType,
    //,
](a: NDArray[dtype]) raises -> Int:
    """Counts the number of non-zero elements in an array.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Input array.

    Returns:
        The number of elements in `a` that are non-zero.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.array[nm.i32]("[3, 0, 5, 0, 2]")
        print(nm.count_nonzero(a))  # 3
        ```
    """
    var a_c = a.contiguous()
    var count: Int = 0
    for i in range(a_c.size):
        if a_c.unsafe_get(i) != 0:
            count += 1
    return count


def count_nonzero[
    dtype: DType,
    //,
](a: NDArray[dtype], axis: Int) raises -> NDArray[DType.int]:
    """Counts non-zero elements along a given axis.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Input array.
        axis: The axis along which to count non-zero elements.

    Returns:
        An integer `NDArray` with `a`'s shape minus the reduced axis,
        containing the count of non-zero elements along that axis.

    Raises:
        NumojoError: If the axis is out of bound.
        NumojoError: If the array is 1-D (use `count_nonzero(a)` instead).

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.array[nm.i32]("[[1, 0, 3], [0, 0, 4]]")
        print(nm.count_nonzero(a, axis=0))  # [1, 0, 2]
        ```
    """
    var normalized_axis = axis
    if axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `count_nonzero`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="count_nonzero",
            )
        )
    if a.ndim == 1:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "Cannot use axis with 1D array. Call `count_nonzero(a)`"
                    " without axis, or reshape a to 2D or higher."
                ),
                location="count_nonzero",
            )
        )

    var result_shape: List[Int] = List[Int]()
    var size_of_axis: Int = a.shape[normalized_axis]
    var slices: List[Slice] = List[Slice]()
    for i in range(a.ndim):
        if i != normalized_axis:
            result_shape.append(a.shape[i])
            slices.append(Slice(0, a.shape[i]))
        else:
            slices.append(Slice(0, 0))  # Temp value

    var result: NDArray[DType.int] = zeros[DType.int](
        NDArrayShape(result_shape)
    )
    for i in range(size_of_axis):
        slices[normalized_axis] = Slice(i, i + 1)
        var arr_slice: NDArray[dtype] = a._getitem_list_slices(slices.copy())
        var mask = NDArray[DType.int](arr_slice.shape)
        for j in range(arr_slice.size):
            mask.unsafe_set(
                j, Scalar[DType.int](1) if arr_slice.unsafe_get(j) != 0 else 0
            )
        result += mask

    return result^
