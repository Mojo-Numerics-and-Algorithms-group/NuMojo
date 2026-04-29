# ===----------------------------------------------------------------------=== #
# NuMojo: Manipulation
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Manipulation routines (numojo.routines.manipulation)

This module implements routines that manipulate the shape and layout of arrays, such as reshaping, transposing, broadcasting, and flipping.
"""

from std.memory import UnsafePointer, memcpy
from std.sys import simd_width_of
from std.algorithm import vectorize

from numojo.core.ndarray import NDArray
from numojo.core.complex import ComplexNDArray
from numojo.core.layout import NDArrayShape
from numojo.core.layout import NDArrayStrides
from numojo.core.type_aliases import Shape
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix
from numojo.core.indexing import (
    IndexMethods,
    TraverseMethods,
)
from numojo.core.indexing.utility import (
    _list_of_flipped_range,
)

# ===----------------------------------------------------------------------=== #
# TODO:
# - When `DataContainer` is supported, re-write `broadcast_to()`.`
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
# Basic operations
# ===----------------------------------------------------------------------=== #


def copy_to[dtype: DType](dst: NDArray[dtype], src: NDArray[dtype]) raises:
    """
    Copies the array from src to dst.

    Args:
        dst: The destination array.
        src: The source array.
    """
    if dst.size != src.size:
        raise NumojoError(
            category="value",
            message=(
                t"`copy_to`: size mismatch (dst: {dst.size}, src: {src.size})."
            ),
            location="copy_to()",
        )

    if dst.is_c_contiguous() and src.is_c_contiguous():
        memcpy(
            dest=dst._buf.ptr + dst.offset,
            src=src._buf.ptr + src.offset,
            count=src.size,
        )
    else:
        for i in range(dst.size):
            var remainder = i
            var src_offset = src.offset
            var dst_offset = dst.offset
            for dim in range(dst.ndim - 1, -1, -1):
                var coord = remainder % dst.shape[dim]
                remainder = remainder // dst.shape[dim]
                src_offset += coord * src.strides[dim]
                dst_offset += coord * dst.strides[dim]
            dst._buf.ptr[dst_offset] = src._buf.ptr[src_offset]


def ndim[dtype: DType](array: NDArray[dtype]) -> Int:
    """
    Returns the number of dimensions of the NDArray.

    Args:
        array: A NDArray.

    Returns:
        The number of dimensions of the NDArray.
    """
    return array.ndim


def ndim[cdtype: ComplexDType](array: ComplexNDArray[cdtype]) -> Int:
    """
    Returns the number of dimensions of the NDArray.

    Args:
        array: A NDArray.

    Returns:
        The number of dimensions of the NDArray.
    """
    return array.ndim


def shape[dtype: DType](array: NDArray[dtype]) -> NDArrayShape:
    """
    Returns the shape of the NDArray.

    Args:
        array: A NDArray.

    Returns:
        The shape of the NDArray.
    """
    return array.shape


def shape[cdtype: ComplexDType](array: ComplexNDArray[cdtype]) -> NDArrayShape:
    """
    Returns the shape of the NDArray.

    Args:
        array: A NDArray.

    Returns: The shape of the NDArray.
    """
    return array.shape


def size[dtype: DType](array: NDArray[dtype], axis: Int) raises -> Int:
    """
    Returns the size of the NDArray.

    Args:
        array: A NDArray.
        axis: The axis to get the size of.

    Returns:
        The size of the NDArray.
    """
    return array.shape[axis]


def size[
    cdtype: ComplexDType
](array: ComplexNDArray[cdtype], axis: Int) raises -> Int:
    """
    Returns the size of the NDArray.

    Args:
        array: A NDArray.
        axis: The axis to get the size of.

    Returns:
        The size of the NDArray.
    """
    return array.shape[axis]


# ===----------------------------------------------------------------------=== #
# Changing array shape
# ===----------------------------------------------------------------------=== #


def reshape[
    dtype: DType
](
    A: NDArray[dtype], shape: NDArrayShape, order: String = "C"
) raises -> NDArray[dtype]:
    """
    Returns an array of the same data with a new shape.

    Raises:
        Error: If the number of elements do not match.

    Args:
        A: A NDArray.
        shape: New shape.
        order: "C" or "F". Read in this order from the original array and
            write in this order into the new array.

    Returns:
        Array of the same data with a new shape.
    """
    if A.size != shape.size():
        raise Error("Cannot reshape: Number of elements do not match.")

    # View safety guard: ensure input is C-contiguous before memcpy.
    if not A.is_c_contiguous():
        return reshape(A.contiguous(), shape, order)

    var array_order: String = String("C") if A.is_c_contiguous() else String(
        "F"
    )

    var B: NDArray[dtype]
    if array_order != order:
        var temp: NDArray[dtype] = ravel(A, order=order)
        B = NDArray[dtype](shape=shape, order=order)
        memcpy(dest=B._buf.ptr, src=temp._buf.ptr, count=A.size)
    else:
        # Write in this order into the new array
        B = NDArray[dtype](shape=shape, order=order)
        memcpy(dest=B._buf.ptr, src=A._buf.ptr, count=A.size)

    return B^


def ravel[
    dtype: DType
](a: NDArray[dtype], order: String = "C") raises -> NDArray[dtype]:
    """
    Returns the raveled version of the NDArray.

    Args:
        a: NDArray.
        order: The order to flatten the array.

    Return:
        A contiguous flattened array.
    """

    # View safety guard: ensure input is C-contiguous before memcpy.
    if not a.is_c_contiguous():
        return ravel(a.contiguous(), order)

    var axis: Int
    if order == "C":
        axis = a.ndim - 1
    elif order == "F":
        axis = 0
    else:
        raise Error(
            String("\nError in `ravel()`: Invalid order: {}").format(order)
        )
    var iterator = a.iter_along_axis(axis=axis, order=order)
    var res: NDArray[dtype] = NDArray[dtype](Shape(a.size))
    var length_of_elements = a.shape[axis]
    var length_of_iterator = a.size // length_of_elements

    for i in range(length_of_iterator):
        var sub = iterator.ith(i)
        memcpy(
            dest=res._buf.ptr + i * length_of_elements,
            src=sub._buf.ptr + sub.offset,
            count=length_of_elements,
        )

    return res^


# ===----------------------------------------------------------------------=== #
# Transpose-like operations
# ===----------------------------------------------------------------------=== #


# TODO: Remove this one if the following function is working well:
# `numojo.core.utility.TraverseMethods.traverse_buffer_according_to_shape_and_strides`
def _set_values_according_to_shape_and_strides(
    mut I: NDArray[DType.int],
    mut index: Int,
    current_dim: Int,
    previous_sum: Int,
    new_shape: NDArrayShape,
    new_strides: NDArrayStrides,
) raises:
    """
    Auxiliary function for `transpose` that set values according to new shape'
    and strides for variadic number of dimensions.
    """
    for index_of_axis in range(new_shape[current_dim]):
        var current_sum = (
            previous_sum + index_of_axis * new_strides[current_dim]
        )
        if current_dim >= new_shape.ndim - 1:
            I._buf.ptr[index] = Scalar[DType.int](current_sum)
            index = index + 1
        else:
            _set_values_according_to_shape_and_strides(
                I,
                index,
                current_dim + 1,
                current_sum,
                new_shape,
                new_strides,
            )


def transpose[
    dtype: DType
](A: NDArray[dtype], axes: List[Int]) raises -> NDArray[dtype]:
    """
    Transpose array of any number of dimensions according to
    arbitrary permutation of the axes.

    If `axes` is not given, it is equal to flipping the axes.
    ```mojo
    import numojo as nm
    var A = nm.random.rand(2,3,4,5)
    print(nm.transpose(A))  # A is a 4darray.
    print(nm.transpose(A, axes=[3,2,1,0]))
    ```

    Examples.
    ```mojo
    import numojo as nm
    var arr2d = nm.random.rand(2,3)
    print(nm.transpose(arr2d, axes=[0, 1]))  # equal to transpose of matrix
    var arr3d = nm.random.rand(2,3,4)
    print(nm.transpose(arr3d, axes=[2, 1, 0]))  # transpose 0-th and 2-th dimensions
    ```
    """
    if len(axes) != A.ndim:
        raise Error(
            String(
                "Length of `axes` ({}) does not match `ndim` of array ({})"
            ).format(len(axes), A.ndim)
        )

    for i in range(A.ndim):
        if i not in axes:
            raise Error(
                String(
                    "`axes` is not a valid permutation of axes of the array. "
                    "It does not contain index {}"
                ).format(i)
            )

    # View safety guard: ensure input is C-contiguous.
    if not A.is_c_contiguous():
        return transpose(A.contiguous(), axes)

    var new_shape: NDArrayShape = NDArrayShape(shape=A.shape)
    for i in range(A.ndim):
        new_shape._buf[i] = A.shape[axes[i]]

    var new_strides: NDArrayStrides = NDArrayStrides(strides=A.strides)
    for i in range(A.ndim):
        new_strides._buf[i] = A.strides[axes[i]]

    var array_order: String = "C" if A.is_c_contiguous() else "F"
    var I = NDArray[DType.int](Shape(A.size), order=array_order)
    var ptr = I._buf.get_ptr()
    TraverseMethods.traverse_buffer_according_to_shape_and_strides(
        ptr, new_shape, new_strides
    )

    var B = NDArray[dtype](new_shape, order=array_order)
    for i in range(B.size):
        B._buf.ptr[i] = (A._buf.ptr + A.offset)[Int(I._buf.ptr[i])]
    return B^


# TODO: Make this operation in place to match numpy.
def transpose[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    (overload) Transpose the array when `axes` is not given.
    If `axes` is not given, it is equal to flipping the axes.
    See docstring of `transpose`.
    """
    if A.ndim == 1:
        return A.copy()
    # View safety guard: ensure input is C-contiguous.
    if not A.is_c_contiguous():
        return transpose(A.contiguous())
    if A.ndim == 2:
        var array_order = "C" if A.is_c_contiguous() else "F"
        var B = NDArray[dtype](Shape(A.shape[1], A.shape[0]), order=array_order)
        if A.shape[0] == 1 or A.shape[1] == 1:
            memcpy(dest=B._buf.ptr, src=A._buf.ptr, count=A.size)
        else:
            for i in range(B.shape[0]):
                for j in range(B.shape[1]):
                    B._setitem(i, j, val=A._getitem(j, i))
        return B^
    else:
        flipped_axes = List[Int]()
        for i in range(A.ndim - 1, -1, -1):
            flipped_axes.append(i)

        return transpose(A, axes=flipped_axes)


def transpose[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Transpose of matrix.
    """
    var order: String = "F"
    if A.is_c_contiguous():
        order = "C"

    var B = Matrix[dtype](Tuple(A.shape[1], A.shape[0]), order=order)

    if A.shape[0] == 1 or A.shape[1] == 1:
        memcpy(dest=B._buf.ptr, src=A._buf.ptr, count=A.size)
    else:
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B._store(i, j, A._load(j, i))
    return B^


def reorder_layout[dtype: DType](A: Matrix[dtype]) raises -> Matrix[dtype]:
    """
    Create a new Matrix with the opposite layout from A:
    if A is C-contiguous, then create a new F-contiguous matrix of the same shape.
    If A is F-contiguous, create a new C-contiguous matrix.

    Copy data into the new layout.
    """

    var rows: Int = A.shape[0]
    var cols: Int = A.shape[1]

    var new_order: String
    if A.flags["C_CONTIGUOUS"]:
        new_order = "F"
    elif A.flags["F_CONTIGUOUS"]:
        new_order = "C"
    else:
        raise Error(
            String(
                "Matrix is neither C-contiguous nor F-contiguous. Cannot"
                " reorder layout!"
            )
        )

    var B = Matrix[dtype](Tuple(rows, cols), new_order)
    if new_order == "C":
        for i in range(rows):
            for j in range(cols):
                B._buf[i * cols + j] = A._buf[i + j * rows]
    else:
        for j in range(cols):
            for i in range(rows):
                B._buf[j * rows + i] = A._buf[i * cols + j]

    return B^


# ===----------------------------------------------------------------------=== #
# Changing number of dimensions
# ===----------------------------------------------------------------------=== #


def broadcast_to[
    dtype: DType
](a: NDArray[dtype], shape: NDArrayShape) raises -> NDArray[dtype]:
    if a.shape.ndim > shape.ndim:
        raise Error(
            String("Cannot broadcast shape {} to shape {}!").format(
                a.shape, shape
            )
        )

    # View safety guard: ensure input is C-contiguous.
    if not a.is_c_contiguous():
        return broadcast_to(a.contiguous(), shape)

    # Check whether broadcasting is possible or not.
    # We compare the shape from the trailing dimensions.

    var b_strides = NDArrayStrides(
        ndim=len(shape), initialized=False
    )  # Strides of b when refer to data of a

    for i in range(a.shape.ndim):
        if a.shape[a.shape.ndim - 1 - i] == shape[shape.ndim - 1 - i]:
            b_strides[shape.ndim - 1 - i] = a.strides[a.shape.ndim - 1 - i]
        elif a.shape[a.shape.ndim - 1 - i] == 1:
            b_strides[shape.ndim - 1 - i] = 0
        else:
            raise Error(
                String("Cannot broadcast shape {} to shape {}!").format(
                    a.shape, shape
                )
            )
    for i in range(shape.ndim - a.shape.ndim):
        b_strides[i] = 0

    # Start broadcasting.
    # TODO: When `DataContainer` is supported, re-write this part.
    # We just need to change the shape and strides and re-use the data.

    var b = NDArray[dtype](shape)  # Construct array of targeted shape.
    # TODO: `b.strides = b_strides` when DataContainer

    # Iterate all items in the new array and fill in correct values.
    for offset in range(b.size):
        var remainder = offset
        var indices = Item(ndim=b.ndim)

        for i in range(b.ndim):
            indices[i] = remainder // b.strides[i]
            remainder %= b.strides[i]
            # TODO: Change b.strides to NDArrayStrides(b.shape) when DataContainer

        (b._buf.ptr + offset).init_pointee_copy(
            a._buf.ptr[
                IndexMethods.get_1d_index(indices, b_strides)
            ]  # TODO: Change b_strides to b.strides when DataContainer
        )

    return b^


def broadcast_to[
    dtype: DType
](
    A: Matrix[dtype],
    shape: Tuple[Int, Int],
    override_order: String = "",
) raises -> Matrix[dtype]:
    """
    Broadcasts the vector to the given shape.

    Example:

    ```console
    > from numojo import Matrix
    > a = Matrix.fromstring("1 2 3", shape=(1, 3))
    > print(mat.broadcast_to(a, (3, 3)))
    [[1.0   2.0     3.0]
     [1.0   2.0     3.0]
     [1.0   2.0     3.0]]
    > a = Matrix.fromstring("1 2 3", shape=(3, 1))
    > print(mat.broadcast_to(a, (3, 3)))
    [[1.0   1.0     1.0]
     [2.0   2.0     2.0]
     [3.0   3.0     3.0]]
    > a = Matrix.fromstring("1", shape=(1, 1))
    > print(mat.broadcast_to(a, (3, 3)))
    [[1.0   1.0     1.0]
     [1.0   1.0     1.0]
     [1.0   1.0     1.0]]
    > a = Matrix.fromstring("1 2", shape=(1, 2))
    > print(mat.broadcast_to(a, (1, 2)))
    [[1.0   2.0]]
    > a = Matrix.fromstring("1 2 3 4", shape=(2, 2))
    > print(mat.broadcast_to(a, (4, 2)))
    Unhandled exception caught during execution: Cannot broadcast shape 2x2 to shape 4x2!
    ```
    """
    var ord: String
    if override_order == "":
        ord = A.order()
    else:
        ord = override_order

    var B: Matrix[dtype] = Matrix[dtype](shape, order=ord)
    if (A.shape[0] == shape[0]) and (A.shape[1] == shape[1]):
        memcpy(dest=B._buf.ptr, src=A._buf.ptr, count=A.size)
    elif (A.shape[0] == 1) and (A.shape[1] == 1):
        B = Matrix[dtype].full(shape, A[0, 0], order=ord)
    elif (A.shape[0] == 1) and (A.shape[1] == shape[1]):
        for i in range(shape[0]):
            memcpy(
                dest=B._buf.offset(shape[1] * i),
                src=A._buf.ptr,
                count=shape[1],
            )
    elif (A.shape[1] == 1) and (A.shape[0] == shape[0]):
        for i in range(shape[0]):
            for j in range(shape[1]):
                B._store(i, j, A._buf.ptr[i])
    else:
        var message = String(
            "Cannot broadcast shape {}x{} to shape {}x{}!"
        ).format(A.shape[0], A.shape[1], shape[0], shape[1])
        raise Error(message)
    return B^


def broadcast_to[
    dtype: DType
](A: Scalar[dtype], shape: Tuple[Int, Int], order: String) raises -> Matrix[
    dtype
]:
    """
    Broadcasts the scalar to the given shape.
    """

    var B: Matrix[dtype] = Matrix[dtype].full(shape, A, order=order)
    return B^


def _broadcast_back_to[
    dtype: DType
](a: NDArray[dtype], shape: NDArrayShape, axis: Int) raises -> NDArray[dtype]:
    """
    Broadcasts the array back to the given shape.
    If array `b` is the result of array `a` operated along an axis,
    it has one dimension less than `a`.
    This function can broadcast `b` back to the shape of `a`.
    It is a temporary function and should not be used by users.
    When `DataContainer` is supported, this function will be removed.
    Whether broadcasting is possible or not is not checked.
    """

    var a_shape = shape
    a_shape[axis] = 1

    var b_strides = NDArrayStrides(
        a_shape
    )  # Strides of b when refer to data of a
    b_strides[axis] = 0

    # Start broadcasting.

    var b = NDArray[dtype](shape)  # Construct array of targeted shape.

    # Iterate all items in the new array and fill in correct values.
    for offset in range(b.size):
        var remainder = offset
        var indices = Item(ndim=b.ndim)

        for i in range(b.ndim):
            indices[i] = remainder // b.strides[i]
            remainder %= b.strides[i]

        (b._buf.ptr + offset).init_pointee_copy(
            a._buf.ptr[IndexMethods.get_1d_index(indices, b_strides)]
        )

    return b^


# ===----------------------------------------------------------------------=== #
# Rearranging elements
# ===----------------------------------------------------------------------=== #


def flip[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Returns flipped array and keep the shape.

    Parameters:
        dtype: DType.

    Args:
        array: A NDArray.

    Returns:
        Flipped array.
    """
    var A = array.contiguous()  # Owned, C-contiguous copy
    for i in range(A.size // 2):
        var temp = A._buf.ptr[i]
        A._buf.ptr[i] = A._buf.ptr[A.size - 1 - i]
        A._buf.ptr[A.size - 1 - i] = temp

    return A^


def flip[
    dtype: DType
](array: NDArray[dtype], var axis: Int) raises -> NDArray[dtype]:
    """
    Returns flipped array along the given axis.

    Parameters:
        dtype: DType.

    Args:
        array: A NDArray.
        axis: Axis along which to flip.

    Returns:
        Flipped array along the given axis.
    """
    var A = array.contiguous()  # Owned, C-contiguous copy
    if axis < 0:
        axis += A.ndim
    if (axis < 0) or (axis >= A.ndim):
        raise Error(
            String("Invalid index: index out of bound [0, {}).").format(A.ndim)
        )

    var I = NDArray[DType.int](Shape(A.size))
    var ptr = I._buf.ptr

    TraverseMethods.traverse_buffer_according_to_shape_and_strides(
        ptr, A.shape.move_axis_to_end(axis), A.strides.move_axis_to_end(axis)
    )

    for i in range(0, A.size, A.shape[axis]):
        for j in range(A.shape[axis] // 2):
            var temp = A._buf.ptr[I._buf.ptr[i + j]]
            A._buf.ptr[I._buf.ptr[i + j]] = A._buf.ptr[
                I._buf.ptr[i + A.shape[axis] - 1 - j]
            ]
            A._buf.ptr[I._buf.ptr[i + A.shape[axis] - 1 - j]] = temp

    return A^


# ===----------------------------------------------------------------------=== #
# Joining arrays
# ===----------------------------------------------------------------------=== #


def _concatenate_list[
    dtype: DType
](arrays: List[NDArray[dtype]], axis: Int = 0) raises -> NDArray[dtype]:
    """Internal: Join a list of arrays along an existing axis."""
    if len(arrays) == 0:
        raise Error(
            NumojoError(
                category="value",
                message="Need at least one array to concatenate.",
                location="concatenate()",
            )
        )

    if len(arrays) == 1:
        return arrays[0].contiguous()

    ref first = arrays[0]
    var ndims = first.ndim

    var ax = axis
    if ax < 0:
        ax += ndims
    if ax < 0 or ax >= ndims:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "axis {} is out of bounds for array of dimension {}."
                ).format(axis, ndims),
                location="concatenate()",
            )
        )

    # Validate shapes and compute the total size along the concat axis.
    var total_along_axis: Int = first.shape[ax]
    for i in range(1, len(arrays)):
        ref arr = arrays[i]
        if arr.ndim != ndims:
            raise Error(
                NumojoError(
                    category="value",
                    message=String(
                        "All arrays must have the same number of dimensions."
                        " Array 0 has {} dims, array {} has {} dims."
                    ).format(ndims, i, arr.ndim),
                    location="concatenate()",
                )
            )
        for d in range(ndims):
            if d != ax and arr.shape[d] != first.shape[d]:
                raise Error(
                    NumojoError(
                        category="shape",
                        message=String(
                            "All array dimensions except for the"
                            " concatenation axis must match. Dimension {}"
                            " of array {} has size {} but expected {}."
                        ).format(d, i, arr.shape[d], first.shape[d]),
                        location="concatenate()",
                    )
                )
        total_along_axis += arr.shape[ax]

    # Build the output shape.
    var out_shape_list = List[Int]()
    for d in range(ndims):
        if d == ax:
            out_shape_list.append(total_along_axis)
        else:
            out_shape_list.append(first.shape[d])
    var out_shape = NDArrayShape(out_shape_list)
    var result = NDArray[dtype](out_shape)

    # Copy data array by array.
    # We iterate over the output in C-order and figure out which source
    # array each element comes from.
    #
    # Strategy: walk the output linearly, convert flat index to
    # multi-dimensional index, map the concat-axis coordinate back to the
    # source array, read from the (contiguous) source.

    # Pre-compute the boundary offsets along the concat axis for each array.
    var boundaries = List[Int]()
    var running: Int = 0
    for i in range(len(arrays)):
        boundaries.append(running)
        running += arrays[i].shape[ax]

    # For each element in the result, determine the source array and index.
    for flat_idx in range(result.size):
        # Convert flat_idx to nd-index (C-order).
        var remainder = flat_idx
        var nd_index = List[Int]()
        for _ in range(ndims):
            nd_index.append(0)
        for d in range(ndims):
            nd_index[d] = remainder // result.strides[d]
            remainder = remainder % result.strides[d]

        # Determine which source array this element comes from.
        var coord_along_axis = nd_index[ax]
        var src_idx: Int = len(arrays) - 1
        for i in range(len(arrays) - 1, -1, -1):
            if coord_along_axis >= boundaries[i]:
                src_idx = i
                break

        # Adjust the coordinate along the concat axis to be local.
        nd_index[ax] = coord_along_axis - boundaries[src_idx]

        result._buf.ptr[flat_idx] = arrays[src_idx]._getitem(nd_index)

    return result^


def concatenate[
    dtype: DType
](*arrays: NDArray[dtype], axis: Int = 0) raises -> NDArray[dtype]:
    """Join a sequence of arrays along an existing axis.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        arrays: The arrays to concatenate. All arrays must have the same
            shape except in the dimension corresponding to `axis`.
        axis: The axis along which the arrays will be joined. Default is 0.

    Returns:
        The concatenated array.

    Raises:
        Error: If the list of arrays is empty.
        Error: If the arrays do not have the same number of dimensions.
        Error: If the array shapes are incompatible along non-concatenation axes.

    Examples:
        ```mojo
        import numojo as nm
        var a = nm.arange[nm.f64](0, 6, 1)
        var a2d = nm.reshape(a, nm.Shape(2, 3))
        var b = nm.arange[nm.f64](6, 12, 1)
        var b2d = nm.reshape(b, nm.Shape(2, 3))
        var c = nm.concatenate(a2d, b2d, axis=0)  # Shape (4, 3)
        var d = nm.concatenate(a2d, b2d, axis=1)  # Shape (2, 6)
        ```
    """
    var arr_list = List[NDArray[dtype]]()
    for i in range(len(arrays)):
        arr_list.append(arrays[i].copy())
    return _concatenate_list(arr_list, axis)


def column_stack[
    dtype: DType
](*arrays: NDArray[dtype]) raises -> NDArray[dtype]:
    """Stack 1-D arrays as columns into a 2-D array, or concatenate
    2-D+ arrays along the second axis (like `numpy.column_stack`).

    Parameters:
        dtype: The data type of the arrays.

    Args:
        arrays: The arrays to stack. 1-D arrays are treated as column
            vectors. All arrays must have the same number of rows
            (first dimension).

    Returns:
        The 2-D (or higher) array formed by stacking the inputs as columns.

    Raises:
        Error: If the list of arrays is empty.

    Examples:
        ```mojo
        import numojo as nm
        var a = nm.arange[nm.f64](0, 3, 1)   # Shape (3,)
        var b = nm.arange[nm.f64](3, 6, 1)   # Shape (3,)
        var c = nm.column_stack(a, b)         # Shape (3, 2)
        ```
    """
    if len(arrays) == 0:
        raise Error(
            NumojoError(
                category="value",
                message="Need at least one array to column_stack.",
                location="column_stack()",
            )
        )

    # Transform 1-D arrays into 2-D column vectors.
    var transformed = List[NDArray[dtype]]()
    for i in range(len(arrays)):
        if arrays[i].ndim == 1:
            # Reshape (N,) -> (N, 1)
            transformed.append(
                reshape(
                    arrays[i].copy(),
                    NDArrayShape(arrays[i].shape[0], 1),
                )
            )
        else:
            transformed.append(arrays[i].copy())

    return _concatenate_list(transformed, axis=1)


def row_stack[dtype: DType](*arrays: NDArray[dtype]) raises -> NDArray[dtype]:
    """Stack arrays vertically (row-wise), equivalent to
    `numpy.row_stack` / `numpy.vstack`.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        arrays: The arrays to stack. 1-D arrays of shape `(N,)` are
            reshaped to `(1, N)` before concatenation.

    Returns:
        The array formed by stacking the inputs vertically.

    Raises:
        Error: If the list of arrays is empty.

    Examples:
        ```mojo
        import numojo as nm
        var a = nm.arange[nm.f64](0, 3, 1)  # Shape (3,)
        var b = nm.arange[nm.f64](3, 6, 1)  # Shape (3,)
        var c = nm.row_stack(a, b)           # Shape (2, 3)
        ```
    """
    if len(arrays) == 0:
        raise Error(
            NumojoError(
                category="value",
                message="Need at least one array to row_stack.",
                location="row_stack()",
            )
        )

    var transformed = List[NDArray[dtype]]()
    for i in range(len(arrays)):
        if arrays[i].ndim == 1:
            # Reshape (N,) -> (1, N)
            transformed.append(
                reshape(
                    arrays[i].copy(),
                    NDArrayShape(1, arrays[i].shape[0]),
                )
            )
        else:
            transformed.append(arrays[i].copy())

    return _concatenate_list(transformed, axis=0)


def hstack[dtype: DType](*arrays: NDArray[dtype]) raises -> NDArray[dtype]:
    """Stack arrays in sequence horizontally (column-wise),
    equivalent to `numpy.hstack`.

    For 1-D arrays, this concatenates along axis 0.
    For 2-D+ arrays, this concatenates along axis 1.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        arrays: The arrays to stack.

    Returns:
        The array formed by stacking the inputs horizontally.

    Raises:
        Error: If the list of arrays is empty.

    Examples:
        ```mojo
        import numojo as nm
        var a = nm.arange[nm.f64](0, 3, 1)  # Shape (3,)
        var b = nm.arange[nm.f64](3, 6, 1)  # Shape (3,)
        var c = nm.hstack(a, b)              # Shape (6,)
        ```
    """
    if len(arrays) == 0:
        raise Error(
            NumojoError(
                category="value",
                message="Need at least one array to hstack.",
                location="hstack()",
            )
        )

    var arr_list = List[NDArray[dtype]]()
    for i in range(len(arrays)):
        arr_list.append(arrays[i].copy())

    # For 1-D arrays, concatenate along axis 0.
    if arr_list[0].ndim == 1:
        return _concatenate_list(arr_list, axis=0)

    return _concatenate_list(arr_list, axis=1)


def vstack[dtype: DType](*arrays: NDArray[dtype]) raises -> NDArray[dtype]:
    """Stack arrays in sequence vertically (row-wise),
    equivalent to `numpy.vstack`.

    For 1-D arrays of shape `(N,)`, they are reshaped to `(1, N)` first.
    Then concatenated along axis 0.

    Parameters:
        dtype: The data type of the arrays.

    Args:
        arrays: The arrays to stack.

    Returns:
        The array formed by stacking the inputs vertically.

    Raises:
        Error: If the list of arrays is empty.

    Examples:
        ```mojo
        import numojo as nm
        var a = nm.arange[nm.f64](0, 3, 1)  # Shape (3,)
        var b = nm.arange[nm.f64](3, 6, 1)  # Shape (3,)
        var c = nm.vstack(a, b)              # Shape (2, 3)
        ```
    """
    if len(arrays) == 0:
        raise Error(
            NumojoError(
                category="value",
                message="Need at least one array to vstack.",
                location="vstack()",
            )
        )

    var transformed = List[NDArray[dtype]]()
    for i in range(len(arrays)):
        if arrays[i].ndim == 1:
            transformed.append(
                reshape(
                    arrays[i].copy(),
                    NDArrayShape(1, arrays[i].shape[0]),
                )
            )
        else:
            transformed.append(arrays[i].copy())

    return _concatenate_list(transformed, axis=0)
