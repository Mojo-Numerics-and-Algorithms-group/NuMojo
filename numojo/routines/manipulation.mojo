# ===----------------------------------------------------------------------=== #
# NuMojo: Manipulation
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Manipulation routines (numojo.routines.manipulation).
======================================================
Array shape and layout manipulation operations.

Routines for reshaping, transposing, broadcasting, flipping, concatenating,
and other shape-changing operations on arrays.

Exports
-------
- `reshape`, `ravel`: Shape changes.
- `transpose`, `flip`: Layout changes.
- `broadcast_to`: Broadcasting.
- `concatenate`, `hstack`, `vstack`, `row_stack`, `column_stack`: Joining.
- `ndim`, `shape`, `size`: Array properties.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.algorithm import vectorize
from std.memory import UnsafePointer, unsafe_memcpy
from std.sys import simd_width_of

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.complex import ComplexNDArray
from numojo.core.dtype.complex_dtype import ComplexDType
from numojo.core.error import NumojoError
from numojo.core.indexing import TraverseMethods
from numojo.core.indexing.utility import _list_of_flipped_range
from numojo.core.layout import NDArrayShape, NDArrayStrides
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape

# ===----------------------------------------------------------------------=== #
# Basic operations
# ===----------------------------------------------------------------------=== #


def copy_to[dtype: DType](mut dst: NDArray[dtype], src: NDArray[dtype]) raises:
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
        unsafe_memcpy(
            dest=dst.unsafe_ptr(),
            src=src.unsafe_ptr(),
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
            dst.unsafe_set(
                dst_offset - dst.offset, src.unsafe_get(src_offset - src.offset)
            )


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
        NumojoError: If the number of elements do not match.

    Args:
        A: A NDArray.
        shape: New shape.
        order: "C" or "F". Read in this order from the original array and
            write in this order into the new array.

    Returns:
        Array of the same data with a new shape.
    """
    if A.size != shape.size():
        raise Error(
            NumojoError(
                category="shape",
                message="Cannot reshape: Number of elements do not match.",
                location="reshape",
            )
        )

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
        unsafe_memcpy(dest=B.unsafe_ptr(), src=temp.unsafe_ptr(), count=A.size)
    else:
        # Write in this order into the new array
        B = NDArray[dtype](shape=shape, order=order)
        unsafe_memcpy(dest=B.unsafe_ptr(), src=A.unsafe_ptr(), count=A.size)

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
            NumojoError(
                category="value",
                message=String(
                    "\nError in `ravel()`: Invalid order: {}"
                ).format(order),
                location="ravel",
            )
        )
    var iterator = a.iter_along_axis(axis=axis, order=order)
    var res: NDArray[dtype] = NDArray[dtype](Shape(a.size))
    var length_of_elements = a.shape[axis]
    var length_of_iterator = a.size // length_of_elements

    for i in range(length_of_iterator):
        var sub = iterator.ith(i)
        unsafe_memcpy(
            dest=res.unsafe_ptr().unsafe_offset(i * length_of_elements),
            src=sub.unsafe_ptr(),
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
            I.unsafe_set(index, Scalar[DType.int](current_sum))
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
            NumojoError(
                category="value",
                message=String(
                    "Length of `axes` ({}) does not match `ndim` of array ({})"
                ).format(len(axes), A.ndim),
                location="transpose",
            )
        )

    for i in range(A.ndim):
        if i not in axes:
            raise Error(
                NumojoError(
                    category="value",
                    message=String(
                        "`axes` is not a valid permutation of axes of the"
                        " array. It does not contain index {}"
                    ).format(i),
                    location="transpose",
                )
            )

    # View safety guard: ensure input is C-contiguous.
    if not A.is_c_contiguous():
        return transpose(A.contiguous(), axes)

    var new_shape: NDArrayShape = NDArrayShape(shape=A.shape)
    for i in range(A.ndim):
        new_shape.unsafe_set(i, A.shape[axes[i]])

    var new_strides: NDArrayStrides = NDArrayStrides(strides=A.strides)
    for i in range(A.ndim):
        new_strides.unsafe_set(i, A.strides[axes[i]])

    var array_order: String = "C" if A.is_c_contiguous() else "F"
    var I = NDArray[DType.int](Shape(A.size), order=array_order)
    var ptr = I.unsafe_ptr()
    TraverseMethods.traverse_buffer_according_to_shape_and_strides(
        ptr, new_shape, new_strides
    )

    var B = NDArray[dtype](new_shape, order=array_order)
    for i in range(B.size):
        B.unsafe_set(i, A.unsafe_get(Int(I.unsafe_get(i))))
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
            unsafe_memcpy(dest=B.unsafe_ptr(), src=A.unsafe_ptr(), count=A.size)
        else:
            for i in range(B.shape[0]):
                for j in range(B.shape[1]):
                    B._setitem(i, j, val=A._getitem(j, i))
        return B^
    else:
        var flipped_axes = List[Int]()
        for i in range(A.ndim - 1, -1, -1):
            flipped_axes.append(i)

        return transpose(A, axes=flipped_axes)


def broadcast_to[
    dtype: DType
](a: NDArray[dtype], shape: NDArrayShape) raises -> NDArray[dtype]:
    """
    Returns a non-owning view of `a` broadcast to `shape`, following NumPy
    broadcasting rules (trailing-dimension alignment, size-1 dims stretch).

    Args:
        a: The array to broadcast.
        shape: The target shape.

    Returns:
        A broadcast view of `a` with shape `shape`.

    Raises:
        NumojoError: If `a.shape` cannot be broadcast to `shape`.

    Notes:
        The returned array shares the underlying buffer with `a` (refcounted,
        zero-copy): broadcast dimensions get stride 0, so no new memory is
        allocated. Because stride-0 dimensions are never C-contiguous, any
        operation that needs a flat contiguous buffer (e.g. SIMD elementwise
        kernels) will materialize the view via `.contiguous()` on demand.
    """
    if a.shape.ndim > shape.ndim:
        raise Error(
            NumojoError(
                category="broadcast",
                message=String("Cannot broadcast shape {} to shape {}!").format(
                    a.shape, shape
                ),
                location="broadcast_to",
            )
        )

    if not a.is_c_contiguous():
        return broadcast_to(a.contiguous(), shape)

    # Check whether broadcasting is possible or not.
    var b_strides = NDArrayStrides(ndim=shape.ndim, initialized=False)

    for i in range(a.shape.ndim):
        if a.shape[a.shape.ndim - 1 - i] == shape[shape.ndim - 1 - i]:
            b_strides[shape.ndim - 1 - i] = a.strides[a.shape.ndim - 1 - i]
        elif a.shape[a.shape.ndim - 1 - i] == 1:
            b_strides[shape.ndim - 1 - i] = 0
        else:
            raise Error(
                NumojoError(
                    category="broadcast",
                    message=String(
                        "Cannot broadcast shape {} to shape {}!"
                    ).format(a.shape, shape),
                    location="broadcast_to",
                )
            )
    for i in range(shape.ndim - a.shape.ndim):
        b_strides[i] = 0

    # view.
    return a.view_with_layout(shape, b_strides, a.offset)


def _broadcast_back_to[
    dtype: DType
](a: NDArray[dtype], shape: NDArrayShape, axis: Int) raises -> NDArray[dtype]:
    """
    Returns a zero-copy view of `a` broadcast back to `shape`.

    If array `b` is the result of array `a` operated along an axis,
    it has one dimension less than `a`.
    This function can broadcast `b` back to the shape of `a`.
    It is a temporary function and should not be used by users.
    Whether broadcasting is possible or not is not checked.
    """

    if not a.is_c_contiguous():
        return _broadcast_back_to(a.contiguous(), shape, axis)

    var a_shape = shape
    a_shape[axis] = 1

    var b_strides = NDArrayStrides(
        a_shape
    )  # Strides of the broadcast view, referring to data of `a`.
    b_strides[axis] = 0

    return a.view_with_layout(shape, b_strides, a.offset)


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
        var temp = A.unsafe_get(i)
        A.unsafe_set(i, A.unsafe_get(A.size - 1 - i))
        A.unsafe_set(A.size - 1 - i, temp)

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
            NumojoError(
                category="index",
                message=String(
                    "Invalid index: index out of bound [0, {})."
                ).format(A.ndim),
                location="flip",
            )
        )

    var I = NDArray[DType.int](Shape(A.size))
    var ptr = I.unsafe_ptr()

    TraverseMethods.traverse_buffer_according_to_shape_and_strides(
        ptr, A.shape.move_axis_to_end(axis), A.strides.move_axis_to_end(axis)
    )

    for i in range(0, A.size, A.shape[axis]):
        for j in range(A.shape[axis] // 2):
            var left = Int(I.unsafe_get(i + j))
            var right = Int(I.unsafe_get(i + A.shape[axis] - 1 - j))
            var temp = A.unsafe_get(left)
            A.unsafe_set(left, A.unsafe_get(right))
            A.unsafe_set(right, temp)

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

        result.unsafe_set(flat_idx, arrays[src_idx]._getitem(nd_index))

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
        NumojoError: If the list of arrays is empty.
        NumojoError: If the arrays do not have the same number of dimensions.
        NumojoError: If the array shapes are incompatible along non-concatenation axes.

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
        NumojoError: If the list of arrays is empty.

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
        NumojoError: If the list of arrays is empty.

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
        NumojoError: If the list of arrays is empty.

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
        NumojoError: If the list of arrays is empty.

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
