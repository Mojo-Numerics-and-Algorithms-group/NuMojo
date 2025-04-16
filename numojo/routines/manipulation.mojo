# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""
Array manipulation routines.
"""

from memory import UnsafePointer, memcpy
from sys import simdwidthof
from algorithm import vectorize

from numojo.core.ndarray import NDArray
from numojo.core.ndshape import NDArrayShape, Shape
from numojo.core.ndstrides import NDArrayStrides
import numojo.core.matrix as matrix
from numojo.core.matrix import Matrix
from numojo.core.utility import _list_of_flipped_range, _get_offset

# ===----------------------------------------------------------------------=== #
# TODO:
# - When `OwnData` is supported, re-write `broadcast_to()`.`
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
# Basic operations
# ===----------------------------------------------------------------------=== #


fn copyto():
    pass


fn ndim[dtype: DType](array: NDArray[dtype]) -> Int:
    """
    Returns the number of dimensions of the NDArray.

    Args:
        array: A NDArray.

    Returns:
        The number of dimensions of the NDArray.
    """
    return array.ndim


fn shape[dtype: DType](array: NDArray[dtype]) -> NDArrayShape:
    """
    Returns the shape of the NDArray.

    Args:
        array: A NDArray.

    Returns:
        The shape of the NDArray.
    """
    return array.shape


fn size[dtype: DType](array: NDArray[dtype], axis: Int) raises -> Int:
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


fn reshape[
    dtype: DType
](
    owned A: NDArray[dtype], shape: NDArrayShape, order: String = "C"
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

    if A.size != shape.size_of_array():
        raise Error("Cannot reshape: Number of elements do not match.")

    var array_order = "C" if A.flags.C_CONTIGUOUS else "F"

    if array_order != order:
        # Read in this order from the original array
        A = ravel(A, order=order)

    # Write in this order into the new array
    var B = NDArray[dtype](shape=shape, order=order)
    memcpy(dest=B._buf.ptr, src=A._buf.ptr, count=A.size)

    return B^


fn ravel[
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
    var res = NDArray[dtype](Shape(a.size))
    var length_of_elements = a.shape[axis]
    var length_of_iterator = a.size // length_of_elements

    for i in range(length_of_iterator):
        memcpy(
            dest=res._buf.ptr + i * length_of_elements,
            src=iterator.ith(i)._buf.ptr,
            count=length_of_elements,
        )

    return res^


# ===----------------------------------------------------------------------=== #
# Transpose-like operations
# ===----------------------------------------------------------------------=== #


# TODO: Remove this one if the following function is working well:
# `numojo.core.utility._traverse_buffer_according_to_shape_and_strides`
fn _set_values_according_to_shape_and_strides(
    mut I: NDArray[DType.index],
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
        var current_sum = previous_sum + index_of_axis * new_strides[
            current_dim
        ]
        if current_dim >= new_shape.ndim - 1:
            I._buf.ptr[index] = current_sum
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


fn transpose[
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
    print(nm.transpose(A, axes=List(3,2,1,0)))
    ```

    Examples.
    ```mojo
    import numojo as nm
    # A is a 2darray
    print(nm.transpose(A, axes=List(0, 1)))  # equal to transpose of matrix
    # A is a 3darray
    print(nm.transpose(A, axes=List(2, 1, 0)))  # transpose 0-th and 2-th dimensions
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

    var new_shape = NDArrayShape(shape=A.shape)
    for i in range(A.ndim):
        new_shape._buf[i] = A.shape[axes[i]]

    var new_strides = NDArrayStrides(strides=A.strides)
    for i in range(A.ndim):
        new_strides._buf[i] = A.strides[axes[i]]

    var array_order = "C" if A.flags.C_CONTIGUOUS else "F"
    var I = NDArray[DType.index](Shape(A.size), order=array_order)
    var ptr = I._buf.ptr
    numojo.core.utility._traverse_buffer_according_to_shape_and_strides(
        ptr, new_shape, new_strides
    )

    var B = NDArray[dtype](new_shape, order=array_order)
    for i in range(B.size):
        B._buf.ptr[i] = A._buf.ptr[I._buf.ptr[i]]
    return B^


fn transpose[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    (overload) Transpose the array when `axes` is not given.
    If `axes` is not given, it is equal to flipping the axes.
    See docstring of `transpose`.
    """

    if A.ndim == 1:
        return A
    if A.ndim == 2:
        var array_order = "C" if A.flags.C_CONTIGUOUS else "F"
        var B = NDArray[dtype](Shape(A.shape[1], A.shape[0]), order=array_order)
        if A.shape[0] == 1 or A.shape[1] == 1:
            memcpy(B._buf.ptr, A._buf.ptr, A.size)
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


fn transpose[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Transpose of matrix.
    """

    var B = Matrix[dtype](Tuple(A.shape[1], A.shape[0]))

    if A.shape[0] == 1 or A.shape[1] == 1:
        memcpy(B._buf.ptr, A._buf.ptr, A.size)
    else:
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B._store(i, j, A._load(j, i))
    return B^


fn reorder_layout[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Create a new Matrix with the opposite layout from A:
    if A is C-contiguous, then create a new F-contiguous matrix of the same shape.
    If A is F-contiguous, create a new C-contiguous matrix.

    Copy data into the new layout.
    """

    var rows = A.shape[0]
    var cols = A.shape[1]

    var new_order: String

    try:
        if A.flags["C_CONTIGUOUS"]:
            new_order = "F"
        else:
            new_order = "C"
    except Error:
        return A

    var B = Matrix[dtype](Tuple(rows, cols), new_order)

    if new_order == "C":
        for i in range(rows):
            for j in range(cols):
                B._buf.ptr[i * cols + j] = A._buf.ptr[i + j * rows]
    else:
        for j in range(cols):
            for i in range(rows):
                B._buf.ptr[j * rows + i] = A._buf.ptr[i * cols + j]

    return B^


# ===----------------------------------------------------------------------=== #
# Changing number of dimensions
# ===----------------------------------------------------------------------=== #


fn broadcast_to[
    dtype: DType
](a: NDArray[dtype], shape: NDArrayShape) raises -> NDArray[dtype]:
    if a.shape.ndim > shape.ndim:
        raise Error(
            String("Cannot broadcast shape {} to shape {}!").format(
                a.shape, shape
            )
        )

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
    # TODO: When `OwnData` is supported, re-write this part.
    # We just need to change the shape and strides and re-use the data.

    var b = NDArray[dtype](shape)  # Construct array of targeted shape.
    # TODO: `b.strides = b_strides` when OwnData

    # Iterate all items in the new array and fill in correct values.
    for offset in range(b.size):
        var remainder = offset
        var indices = Item(ndim=b.ndim, initialized=False)

        for i in range(b.ndim):
            indices[i] = remainder // b.strides[i]
            remainder %= b.strides[i]
            # TODO: Change b.strides to NDArrayStrides(b.shape) when OwnData

        (b._buf.ptr + offset).init_pointee_copy(
            a._buf.ptr[
                _get_offset(indices, b_strides)
            ]  # TODO: Change b_strides to b.strides when OwnData
        )

    return b^


fn broadcast_to[
    dtype: DType
](A: Matrix[dtype], shape: Tuple[Int, Int]) raises -> Matrix[dtype]:
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

    var B = Matrix[dtype](shape)
    if (A.shape[0] == shape[0]) and (A.shape[1] == shape[1]):
        B = A
    elif (A.shape[0] == 1) and (A.shape[1] == 1):
        B = Matrix.full[dtype](shape, A[0, 0])
    elif (A.shape[0] == 1) and (A.shape[1] == shape[1]):
        for i in range(shape[0]):
            memcpy(
                dest=B._buf.ptr.offset(shape[1] * i),
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


fn broadcast_to[
    dtype: DType
](A: Scalar[dtype], shape: Tuple[Int, Int]) raises -> Matrix[dtype]:
    """
    Broadcasts the scalar to the given shape.
    """

    var B = Matrix[dtype](shape)
    B = Matrix.full[dtype](shape, A)
    return B^


fn _broadcast_back_to[
    dtype: DType
](a: NDArray[dtype], shape: NDArrayShape, axis: Int) raises -> NDArray[dtype]:
    """
    Broadcasts the array back to the given shape.
    If array `b` is the result of array `a` operated along an axis,
    it has one dimension less than `a`.
    This function can broadcast `b` back to the shape of `a`.
    It is a temporary function and should not be used by users.
    When `OwnData` is supported, this function will be removed.
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
        var indices = Item(ndim=b.ndim, initialized=False)

        for i in range(b.ndim):
            indices[i] = remainder // b.strides[i]
            remainder %= b.strides[i]

        (b._buf.ptr + offset).init_pointee_copy(
            a._buf.ptr[_get_offset(indices, b_strides)]
        )

    return b^


# ===----------------------------------------------------------------------=== #
# Rearranging elements
# ===----------------------------------------------------------------------=== #


fn flip[dtype: DType](owned A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Returns flipped array and keep the shape.

    Parameters:
        dtype: DType.

    Args:
        A: A NDArray.

    Returns:
        Flipped array.
    """

    for i in range(A.size // 2):
        var temp = A._buf.ptr[i]
        A._buf.ptr[i] = A._buf.ptr[A.size - 1 - i]
        A._buf.ptr[A.size - 1 - i] = temp

    return A^


fn flip[
    dtype: DType
](owned A: NDArray[dtype], owned axis: Int) raises -> NDArray[dtype]:
    """
    Returns flipped array along the given axis.

    Parameters:
        dtype: DType.

    Args:
        A: A NDArray.
        axis: Axis along which to flip.

    Returns:
        Flipped array along the given axis.
    """

    if axis < 0:
        axis += A.ndim
    if (axis < 0) or (axis >= A.ndim):
        raise Error(
            String("Invalid index: index out of bound [0, {}).").format(A.ndim)
        )

    var I = NDArray[DType.index](Shape(A.size))
    var ptr = I._buf.ptr

    numojo.core.utility._traverse_buffer_according_to_shape_and_strides(
        ptr, A.shape._move_axis_to_end(axis), A.strides._move_axis_to_end(axis)
    )

    print(A.size, A.shape[axis])
    for i in range(0, A.size, A.shape[axis]):
        for j in range(A.shape[axis] // 2):
            var temp = A._buf.ptr[I._buf.ptr[i + j]]
            A._buf.ptr[I._buf.ptr[i + j]] = A._buf.ptr[
                I._buf.ptr[i + A.shape[axis] - 1 - j]
            ]
            A._buf.ptr[I._buf.ptr[i + A.shape[axis] - 1 - j]] = temp

    return A^
