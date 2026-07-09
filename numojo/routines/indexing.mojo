# ===----------------------------------------------------------------------=== #
# NuMojo: Indexing routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Indexing routines (numojo.routines.indexing)
-----------------------------------------------
- Generating index arrays
- Indexing-like operations
- Inserting data into arrays
- Iterating over arrays.
"""

from std.memory import unsafe_memcpy
from std.sys import simd_width_of
from std.algorithm import vectorize

from numojo import broadcast_to
from numojo.core.ndarray import NDArray
from numojo.core.layout import NDArrayShape, NDArrayStrides
from numojo.core.indexing import IndexMethods
import numojo.routines.manipulation as manipulation
from numojo.routines.creation import array as _array_creation_from_list
from numojo.core.type_aliases import Shape
from numojo.core.indexing.item import Item
from numojo.routines.manipulation import ravel

# ===----------------------------------------------------------------------=== #
# Generating index arrays
# ===----------------------------------------------------------------------=== #


def `where`[
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
    var mask_c = mask.contiguous()

    for i in range(x.size):
        if mask_c.unsafe_get(i) == True:
            x.itemset(i, scalar)


# TODO: do it with vectorization
def `where`[
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

    var mask_c = mask.contiguous()
    var y_c = y.contiguous()

    for i in range(x.size):
        if mask_c.unsafe_get(i) == True:
            x.itemset(i, y_c.unsafe_get(i))


def `where`[
    dtype: DType,
    //,
](condition: NDArray[dtype],) raises -> List[NDArray[DType.int]]:
    """Returns indices where `condition` is non-zero.

    Returns one 1-D integer index array per dimension of `condition`.

    Args:
        condition: Selector array.

    Returns:
        A `List` of `condition.ndim` 1-D integer arrays.
    """
    return nonzero(condition)


def `where`[
    dtype: DType
](
    condition: NDArray[DType.bool],
    x: NDArray[dtype],
    y: NDArray[dtype],
) raises -> NDArray[dtype]:
    """Returns elements chosen from `x` or `y` depending on `condition`.

    This is the functional, non-mutating form. ``condition``, ``x``, and ``y``
    are broadcast against each other. Elements where ``condition`` is True come
    from ``x``; elements where it is False come from ``y``.

    Parameters:
        dtype: DType of `x` and `y`.

    Args:
        condition: Boolean selector array.
        x: Values used where ``condition`` is True.
        y: Values used where ``condition`` is False.

    Returns:
        New array of the broadcast shape filled from `x` where True and `y`
        where False.

    Raises:
        Error: If ``condition``, ``x``, and ``y`` are not broadcast-compatible.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.array[nm.f32]("[1.0, 2.0, 3.0, 4.0]")
        var b = nm.array[nm.f32]("[10.0, 20.0, 30.0, 40.0]")
        var mask = nm.array[nm.boolean]("[True, False, True, False]")
        print(nm.where(mask, a, b))
        # [1.0  20.0  3.0  40.0]
        ```
        .
    """
    var cond_bc = broadcast_to(
        condition, condition.shape.broadcast(x.shape.broadcast(y.shape))
    )
    var x_bc = broadcast_to(x, cond_bc.shape)
    var y_bc = broadcast_to(y, cond_bc.shape)

    var cond_c = cond_bc.contiguous()
    var x_c = x_bc.contiguous()
    var y_c = y_bc.contiguous()

    var result = NDArray[dtype](cond_c.shape)
    for i in range(result.size):
        if cond_c.unsafe_get(i):
            result.unsafe_set(i, x_c.unsafe_get(i))
        else:
            result.unsafe_set(i, y_c.unsafe_get(i))
    return result^


def `where`[
    dtype: DType
](
    condition: NDArray[DType.bool],
    x: NDArray[dtype],
    y: Scalar[dtype],
) raises -> NDArray[dtype]:
    """Returns elements from `x` or scalar `y` depending on `condition`.

    Overload of ``where`` where the false-branch is a scalar broadcast.

    Parameters:
        dtype: DType of `x` and `y`.

    Args:
        condition: Boolean selector array.
        x: Values used where ``condition`` is True.
        y: Scalar used where ``condition`` is False.

    Returns:
        New array filled from `x` where True and `y` everywhere else.

    Raises:
        Error: If ``condition`` and `x` are not broadcast-compatible.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.array[nm.f32]("[1.0, 2.0, 3.0, 4.0]")
        var mask = nm.array[nm.boolean]("[True, False, True, False]")
        print(nm.where(mask, a, Scalar[nm.f32](0.0)))
        # [1.0  0.0  3.0  0.0]
        ```
        .
    """
    var bc_shape = condition.shape.broadcast(x.shape)
    var cond_c = broadcast_to(condition, bc_shape).contiguous()
    var x_c = broadcast_to(x, bc_shape).contiguous()

    var result = NDArray[dtype](bc_shape)
    for i in range(result.size):
        result.unsafe_set(i, x_c.unsafe_get(i) if cond_c.unsafe_get(i) else y)
    return result^


def `where`[
    dtype: DType
](
    condition: NDArray[DType.bool],
    x: Scalar[dtype],
    y: NDArray[dtype],
) raises -> NDArray[dtype]:
    """Returns scalar `x` or elements of `y` depending on `condition`.

    Overload of ``where`` where the true-branch is a scalar broadcast.

    Parameters:
        dtype: DType of `x` and `y`.

    Args:
        condition: Boolean selector array.
        x: Scalar used where ``condition`` is True.
        y: Values used where ``condition`` is False.

    Returns:
        New array filled from `x` where True and `y` everywhere else.

    Raises:
        Error: If ``condition`` and `y` are not broadcast-compatible.

    Examples:
        ```mojo
        import numojo as nm

        var b = nm.array[nm.f32]("[10.0, 20.0, 30.0, 40.0]")
        var mask = nm.array[nm.boolean]("[True, False, True, False]")
        print(nm.where(mask, Scalar[nm.f32](0.0), b))
        # [0.0  20.0  0.0  40.0]
        ```
        .
    """
    var bc_shape = condition.shape.broadcast(y.shape)
    var cond_c = broadcast_to(condition, bc_shape).contiguous()
    var y_c = broadcast_to(y, bc_shape).contiguous()

    var result = NDArray[dtype](bc_shape)
    for i in range(result.size):
        result.unsafe_set(i, x if cond_c.unsafe_get(i) else y_c.unsafe_get(i))
    return result^


# ===----------------------------------------------------------------------=== #
# Indexing-like operations
# ===----------------------------------------------------------------------=== #


def fancy_index[
    dtype: DType,
    //,
](a: NDArray[dtype], index_arrays: List[NDArray[DType.int]]) raises -> NDArray[
    dtype
]:
    """Element-wise multi-axis fancy (advanced) indexing.

    Selects elements from `a` by supplying one integer-array index per axis
    as a ``List``.  All index arrays are broadcast against each other; the
    output shape equals that broadcast shape.  This allows the ``a[[row_arr, col_arr, ...]]``
    syntax (the outer ``[]`` is the list literal).

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Source N-D array.
        index_arrays: List of integer NDArrays — exactly `a.ndim` entries,
            one per axis.  Each array is broadcast to the common shape.

    Returns:
        Array of shape ``broadcast(index_arrays)`` whose element ``i`` equals
        ``a[index_arrays[0][i], index_arrays[1][i], ...]``.

    Raises:
        Error: If the number of index arrays does not equal `a.ndim`.
        Error: If the index arrays are not mutually broadcast-compatible.
        Error: If any index value is out of bounds for its axis.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.i32](12).reshape(nm.Shape(3, 4))
        var rows = nm.array[nm.int]("[0, 1]")
        var cols = nm.array[nm.int]("[2, 3]")
        var idx = List[nm.NDArray[DType.int]]()
        idx.append(rows^)
        idx.append(cols^)
        print(nm.fancy_index(a, idx))
        # [2  7]
        # or via __getitem__:
        print(a[idx])
        # [2  7]
        ```
    """
    var n_idx = len(index_arrays)
    if n_idx != a.ndim:
        raise Error(
            String(
                "\nError in `fancy_index`: expected {} index arrays (one per"
                " axis), got {}."
            ).format(a.ndim, n_idx)
        )

    # Broadcast all index arrays to a common shape.
    var out_shape = index_arrays[0].shape
    for k in range(1, n_idx):
        try:
            out_shape = out_shape.broadcast(index_arrays[k].shape)
        except e:
            raise Error(
                String(
                    "\nError in `fancy_index`: index arrays are not"
                    " broadcast-compatible: "
                )
                + String(e)
            )

    # Materialise each broadcast index array as a contiguous buffer.
    var bc_indices = List[NDArray[DType.int]](capacity=n_idx)
    for k in range(n_idx):
        bc_indices.append(broadcast_to(index_arrays[k], out_shape).contiguous())

    var result = NDArray[dtype](out_shape)
    var out_size = result.size

    for i in range(out_size):
        var coords = List[Int](capacity=n_idx)
        for k in range(n_idx):
            var raw = Int(bc_indices[k]._buf.ptr[i])
            var ax_size = a.shape[k]
            if raw < -ax_size or raw >= ax_size:
                raise Error(
                    String(
                        "\nError in `fancy_index`: index {} is out of bounds"
                        " for axis {} with size {}."
                    ).format(raw, k, ax_size)
                )
            if raw < 0:
                raw += ax_size
            coords.append(raw)
        result._buf.ptr[i] = a._getitem(coords)

    return result^


def fancy_index[
    dtype: DType,
    //,
](a: NDArray[dtype], *index_arrays: NDArray[DType.int]) raises -> NDArray[
    dtype
]:
    """Element-wise multi-axis fancy (advanced) indexing (variadic overload).

    Convenience overload that accepts index arrays as variadic positional
    arguments instead of a ``List``.  Delegates to the ``List`` overload.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Source N-D array.
        index_arrays: Exactly `a.ndim` integer index arrays, one per axis.

    Returns:
        Array of shape ``broadcast(index_arrays)``.

    Raises:
        Error: If the number of index arrays does not equal `a.ndim`.
        Error: If the index arrays are not mutually broadcast-compatible.
        Error: If any index value is out of bounds for its axis.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.i32](12).reshape(nm.Shape(3, 4))
        var rows = nm.array[nm.int]("[0, 1]")
        var cols = nm.array[nm.int]("[2, 3]")
        print(nm.fancy_index(a, rows, cols))
        # [2  7]
        ```
        .
    """
    var idx_list = List[NDArray[DType.int]](capacity=len(index_arrays))
    for k in range(len(index_arrays)):
        idx_list.append(index_arrays[k].copy())  # variadic refs can't be moved
    return fancy_index(a, idx_list)

def fancy_index[
    dtype: DType,
    //,
](a: NDArray[dtype], index_arrays: List[NDArray[DType.int]]) raises -> NDArray[
    dtype
]:
    """Element-wise multi-axis fancy (advanced) indexing.

    Selects elements from `a` by supplying one integer-array index per axis
    as a ``List``.  All index arrays are broadcast against each other; the
    output shape equals that broadcast shape.  This allows the ``a[[row_arr, col_arr, ...]]``
    syntax (the outer ``[]`` is the list literal).

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Source N-D array.
        index_arrays: List of integer NDArrays — exactly `a.ndim` entries,
            one per axis.  Each array is broadcast to the common shape.

    Returns:
        Array of shape ``broadcast(index_arrays)`` whose element ``i`` equals
        ``a[index_arrays[0][i], index_arrays[1][i], ...]``.

    Raises:
        Error: If the number of index arrays does not equal `a.ndim`.
        Error: If the index arrays are not mutually broadcast-compatible.
        Error: If any index value is out of bounds for its axis.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.i32](12).reshape(nm.Shape(3, 4))
        var rows = nm.array[nm.int]("[0, 1]")
        var cols = nm.array[nm.int]("[2, 3]")
        var idx = List[nm.NDArray[DType.int]]()
        idx.append(rows^)
        idx.append(cols^)
        print(nm.fancy_index(a, idx))
        # [2  7]
        # or via __getitem__:
        print(a[idx])
        # [2  7]
        ```
    """
    var n_idx = len(index_arrays)
    if n_idx != a.ndim:
        raise Error(
            String(
                "\nError in `fancy_index`: expected {} index arrays (one per"
                " axis), got {}."
            ).format(a.ndim, n_idx)
        )

    # Broadcast all index arrays to a common shape.
    var out_shape = index_arrays[0].shape
    for k in range(1, n_idx):
        try:
            out_shape = out_shape.broadcast(index_arrays[k].shape)
        except e:
            raise Error(
                String(
                    "\nError in `fancy_index`: index arrays are not"
                    " broadcast-compatible: "
                )
                + String(e)
            )

    # Materialise each broadcast index array as a contiguous buffer.
    var bc_indices = List[NDArray[DType.int]](capacity=n_idx)
    for k in range(n_idx):
        bc_indices.append(broadcast_to(index_arrays[k], out_shape).contiguous())

    var result = NDArray[dtype](out_shape)
    var out_size = result.size

    for i in range(out_size):
        var coords = List[Int](capacity=n_idx)
        for k in range(n_idx):
            var raw = Int(bc_indices[k].unsafe_get(i))
            var ax_size = a.shape[k]
            if raw < -ax_size or raw >= ax_size:
                raise Error(
                    String(
                        "\nError in `fancy_index`: index {} is out of bounds"
                        " for axis {} with size {}."
                    ).format(raw, k, ax_size)
                )
            if raw < 0:
                raw += ax_size
            coords.append(raw)
        result.unsafe_set(i, a._getitem(coords))

    return result^


def fancy_index[
    dtype: DType,
    //,
](a: NDArray[dtype], *index_arrays: NDArray[DType.int]) raises -> NDArray[
    dtype
]:
    """Element-wise multi-axis fancy (advanced) indexing (variadic overload).

    Convenience overload that accepts index arrays as variadic positional
    arguments instead of a ``List``.  Delegates to the ``List`` overload.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Source N-D array.
        index_arrays: Exactly `a.ndim` integer index arrays, one per axis.

    Returns:
        Array of shape ``broadcast(index_arrays)``.

    Raises:
        Error: If the number of index arrays does not equal `a.ndim`.
        Error: If the index arrays are not mutually broadcast-compatible.
        Error: If any index value is out of bounds for its axis.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.i32](12).reshape(nm.Shape(3, 4))
        var rows = nm.array[nm.int]("[0, 1]")
        var cols = nm.array[nm.int]("[2, 3]")
        print(nm.fancy_index(a, rows, cols))
        # [2  7]
        ```
        .
    """
    var idx_list = List[NDArray[DType.int]](capacity=len(index_arrays))
    for k in range(len(index_arrays)):
        idx_list.append(index_arrays[k].copy())  # variadic refs can't be moved
    return fancy_index(a, idx_list)


def compress[
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
    if not condition.is_c_contiguous():
        return compress(condition.contiguous(), a, axis)

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
        number_of_true += Int(condition.unsafe_get(i))

    var shape_of_res: NDArrayShape = a.shape
    shape_of_res[normalized_axis] = number_of_true

    var result: NDArray[dtype] = NDArray[dtype](Shape(shape_of_res))
    var res_strides: NDArrayStrides = NDArrayStrides(
        ndim=result.ndim, initialized=False
    )
    var temp: Scalar[DType.int] = 1
    for i in range(result.ndim - 1, -1, -1):
        if i != normalized_axis:
            (res_strides._buf.ptr.unsafe_offset(i)).unsafe_write(temp)
            temp *= Scalar[DType.int](result.shape[i])
    (res_strides._buf.ptr.unsafe_offset(normalized_axis)).unsafe_write(temp)

    var iterator = a.iter_over_dimension(normalized_axis)

    var count: Scalar[DType.int] = 0
    for i in range(len(condition)):
        if condition.item(i):
            var current_slice = iterator.ith(i)
            for offset in range(current_slice.size):
                var remainder: Scalar[DType.int] = count

                var item: Item = Item(ndim=result.ndim)

                # First along the axis
                var j = normalized_axis
                (item._buf.ptr.unsafe_offset(j)).unsafe_write(
                    remainder // res_strides.unsafe_load(j)
                )
                remainder %= res_strides.unsafe_load(j)

                # Then along other axes
                for j in range(result.ndim):
                    if j != normalized_axis:
                        (item._buf.ptr.unsafe_offset(j)).unsafe_write(
                            remainder // res_strides.unsafe_load(j)
                        )
                        remainder %= res_strides.unsafe_load(j)

                result.unsafe_set(
                    IndexMethods.get_1d_index(item, result.strides),
                    current_slice.unsafe_get(offset),
                )

                count += 1

    return result^


def compress[
    dtype: DType
](condition: NDArray[DType.bool], a: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Return selected slices of an array along given axis.
    If no axis is provided, the array is flattened before use.
    This is a function ***OVERLOAD***.

    Raises:
        Error: If the condition is not 1-D array.
        Error: If the condition length is out of bound for the given axis.

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


def take_along_axis[
    dtype: DType,
    //,
](
    arr: NDArray[dtype], indices: NDArray[DType.int], axis: Int = 0
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

    var broadcasted_indices: NDArray[
        DType.int
    ] = indices.copy()  # make this owned and don't copy

    if arr.shape != indices.shape:
        var arr_shape_new = arr.shape
        arr_shape_new[normalized_axis] = indices.shape[normalized_axis]

        try:
            broadcasted_indices = broadcast_to(indices, arr_shape_new)
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
            var arr_slice_after_applying_indices: NDArray[dtype] = arr_slice[
                indices_slice
            ]
            unsafe_memcpy(
                dest=result.unsafe_ptr().unsafe_offset(
                    i * result.shape[normalized_axis]
                ),
                src=arr_slice_after_applying_indices.unsafe_ptr(),
                count=result.shape[normalized_axis],
            )
    else:
        # If axis is not the last axis, the data is not contiguous.
        for i in range(length_of_iterator):
            var indices_slice_offsets: NDArray[DType.int]
            var indices_slice: NDArray[DType.int]
            var indices_slice_offsets_slice = indices_iterator.ith_with_offsets(
                i
            )
            indices_slice_offsets = indices_slice_offsets_slice[0].copy()
            indices_slice = indices_slice_offsets_slice[1].copy()
            var arr_slice = arr_iterator.ith(i)
            var arr_slice_after_applying_indices = arr_slice[indices_slice]
            for j in range(arr_slice_after_applying_indices.size):
                result.unsafe_set(
                    Int(indices_slice_offsets[j]),
                    arr_slice_after_applying_indices.unsafe_get(j),
                )

    return result^


# ===----------------------------------------------------------------------=== #
# take
# ===----------------------------------------------------------------------=== #


def take[
    dtype: DType,
    //,
](
    a: NDArray[dtype],
    indices: NDArray[DType.int],
    axis: Int,
) raises -> NDArray[
    dtype
]:
    """Takes elements from an array along an axis.

    Output shape is `a.shape[:axis] + indices.shape + a.shape[axis+1:]`.
    Negative indices into `a` along the axis are normalised. Negative `axis`
    values are also normalised.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Source array.
        indices: Indices of values to take along the axis.
        axis: Axis along which to select. Negative values count from the end.

    Returns:
        Array of shape `a.shape[:axis] + indices.shape + a.shape[axis+1:]`.

    Raises:
        Error: If `axis` is out of bounds.
        Error: If any index is out of bounds for the given axis.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.i32](12).reshape(nm.Shape(3, 4))

        print(nm.indexing.take(a, nm.array[nm.int]("[2, 0, 1]"), axis=0))
        # shape (3, 4): rows 2, 0, 1
        #
        print(nm.indexing.take(a, nm.array[nm.int]("[1, 3]"), axis=1))
        # shape (3, 2): cols 1, 3
        ```
        .
    """
    var norm_axis = axis
    if norm_axis < 0:
        norm_axis = a.ndim + norm_axis
    if norm_axis < 0 or norm_axis >= a.ndim:
        raise Error(
            String(
                "\nError in `take`: axis {} is out of bounds for array with"
                " {} dimensions."
            ).format(axis, a.ndim)
        )

    # a.shape[:axis] + indices.shape + a.shape[axis+1:]
    var out_shape_list = List[Int]()
    for d in range(norm_axis):
        out_shape_list.append(a.shape[d])
    for d in range(indices.ndim):
        out_shape_list.append(indices.shape[d])
    for d in range(norm_axis + 1, a.ndim):
        out_shape_list.append(a.shape[d])

    var result = NDArray[dtype](NDArrayShape(out_shape_list))

    # Sizes product(a.shape[:axis]) + indices.size + product(a.shape[axis+1:])
    var outer_size = 1
    for d in range(norm_axis):
        outer_size *= a.shape[d]
    var n_idx = indices.size
    var inner_size = 1
    for d in range(norm_axis + 1, a.ndim):
        inner_size *= a.shape[d]

    var axis_size = a.shape[norm_axis]
    var indices_c = indices.contiguous()
    var a_c = a.contiguous()

    for outer in range(outer_size):
        for i in range(n_idx):
            var raw = Int(indices_c.unsafe_get(i))
            if raw < -axis_size or raw >= axis_size:
                raise Error(
                    String(
                        "\nError in `take`: index {} is out of bounds for"
                        " axis {} with size {}."
                    ).format(raw, norm_axis, axis_size)
                )
            var norm_idx = raw
            if norm_idx < 0:
                norm_idx += axis_size

            var src_base = (
                outer * axis_size * inner_size + norm_idx * inner_size
            )
            var dst_base = outer * n_idx * inner_size + i * inner_size
            unsafe_memcpy(
                dest=result.unsafe_ptr().unsafe_offset(dst_base),
                src=a_c.unsafe_ptr().unsafe_offset(src_base),
                count=inner_size,
            )

    return result^


def take[
    dtype: DType,
    //,
](a: NDArray[dtype], indices: NDArray[DType.int],) raises -> NDArray[dtype]:
    """Takes elements from a flattened array by linear indices.

    Equivalent to `take(a.flatten(), indices, axis=0)`. The output shape
    matches `indices.shape`.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Source array (flattened before indexing).
        indices: Linear indices into the flattened source. May be any shape.

    Returns:
        Array with the same shape as `indices`.

    Raises:
        Error: If any index is out of bounds for the flattened array.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.i32](12).reshape(nm.Shape(3, 4))
        print(nm.indexing.take(a, nm.array[nm.int]("[0, 5, 11]")))
        # [0, 5, 11]
        ```
        .
    """
    var flat = manipulation.ravel(a)
    return take(flat, indices, axis=0)


# TODO: Add this after Scalar[DType.int] and Int are merged in Mojo. a
# def take[
#     dtype: DType,
#     //,
# ](a: NDArray[dtype], indices: List[Scalar[DType.int]],) raises -> NDArray[dtype]:
#     """Takes elements from a flattened array by linear indices.

#     Equivalent to `take(a.flatten(), indices, axis=0)`. The output shape
#     matches `indices.shape`.

#     Parameters:
#         dtype: Data type of the source array.

#     Args:
#         a: Source array (flattened before indexing).
#         indices: List of linear indices into the flattened source (1D).

#     Returns:
#         Array with the same shape as `indices`.

#     Raises:
#         Error: If any index is out of bounds for the flattened array.

#     Examples:
#         ```mojo
#         import numojo as nm

#         var a = nm.arange[nm.i32](12).reshape(nm.Shape(3, 4))
#         print(nm.indexing.take(a, nm.array[nm.int]("[0, 5, 11]")))
#         # [0, 5, 11]
#         ```
#         .
#     """
#     var shape: List[Scalar[DType.int]] = [Scalar[DType.int](len(indices))]
#     var arr = _array_creation_from_list[dtype](indices, shape)
#     return take(arr, indices, axis=0)


# ===----------------------------------------------------------------------=== #
# put
# ===----------------------------------------------------------------------=== #


def put[
    dtype: DType,
    //,
](
    mut a: NDArray[dtype],
    indices: NDArray[DType.int],
    values: NDArray[dtype],
) raises:
    """Replaces values at flat (linear) index positions of `a` in-place.

    Equivalent to `a.flatten()[indices] = values`, but writes directly into
    `a` (any array order). If `values` has fewer elements than `indices`, it
    is repeated (broadcast) cyclically over `indices`. `values` must not be empty
    unless `indices` is also empty.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Destination array to be modified in-place.
        indices: Linear (flat) indices into `a`. May be any shape. Negative
            indices are normalised (counted from the end).
        values: Values to write. Broadcast cyclically if shorter than
            `indices`.

    Raises:
        Error: If any index is out of bounds for the flattened array.
        Error: If `values` is empty while `indices` is not.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.i32](6)
        nm.indexing.put(a, nm.array[nm.int]("[0, 2]"), nm.array[nm.i32]("[10, 20]"))
        print(a)
        # [10, 1, 20, 3, 4, 5]
        ```
        .
    """
    if indices.size == 0:
        return

    if values.size == 0:
        raise Error(
            String(
                "\nError in `put`: values is empty but indices has {}"
                " element(s)."
            ).format(indices.size)
        )

    var indices_c = indices.contiguous()
    var values_c = values.contiguous()

    for i in range(indices_c.size):
        var raw = Int(indices_c.unsafe_get(i))
        if raw < -a.size or raw >= a.size:
            raise Error(
                String(
                    "\nError in `put`: index {} is out of bounds for array"
                    " of size {}."
                ).format(raw, a.size)
            )
        var norm_idx = raw
        if norm_idx < 0:
            norm_idx += a.size

        a.itemset(norm_idx, values_c.unsafe_get(i % values_c.size))


def put[
    dtype: DType,
    //,
](
    mut a: NDArray[dtype],
    indices: NDArray[DType.int],
    value: Scalar[dtype],
) raises:
    """Replaces values at flat (linear) index positions of `a` in-place with
    a single broadcast scalar.

    This is a function ***OVERLOAD*** of `put` for the scalar-`value` case.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Destination array to be modified in-place.
        indices: Linear (flat) indices into `a`. May be any shape. Negative
            indices are normalised (counted from the end).
        value: Scalar value written to every selected position.

    Raises:
        Error: If any index is out of bounds for the flattened array.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.i32](6)
        nm.indexing.put(a, nm.array[nm.int]("[0, 2]"), Scalar[nm.i32](99))
        print(a)
        # [99, 1, 99, 3, 4, 5]
        ```
        .
    """
    if indices.size == 0:
        return

    var indices_c = indices.contiguous()

    for i in range(indices_c.size):
        var raw = Int(indices_c.unsafe_get(i))
        if raw < -a.size or raw >= a.size:
            raise Error(
                String(
                    "\nError in `put`: index {} is out of bounds for array"
                    " of size {}."
                ).format(raw, a.size)
            )
        var norm_idx = raw
        if norm_idx < 0:
            norm_idx += a.size

        a.itemset(norm_idx, value)


# ===----------------------------------------------------------------------=== #
# nonzero
# ===----------------------------------------------------------------------=== #


def unravel_index(
    index: Int, shape: NDArrayShape, order: String = "C"
) raises -> List[Int]:
    """Converts a flat index into coordinates for `shape`.

    Args:
        index: Flat linear index.
        shape: Target shape.
        order: `"C"` for row-major order or `"F"` for column-major order.

    Returns:
        A list of coordinates, one per dimension.

    Raises:
        Error: If `index` is out of bounds for the flattened array.
        Error: If `order` is not `"C"` or `"F"`.
    """
    var size = shape.size()
    if order != "C" and order != "F":
        raise Error(
            String(
                "\nError in `unravel_index`: order must be 'C' or 'F', got"
                " '{}'."
            ).format(order)
        )
    if index < 0 or index >= size:
        raise Error(
            String(
                "\nError in `unravel_index`: index {} is out of bounds for"
                " array with size {}."
            ).format(index, size)
        )

    var result = List[Int](capacity=shape.ndim)
    for _ in range(shape.ndim):
        result.append(0)

    var rem = index
    if order == "C":
        for d in range(shape.ndim - 1, -1, -1):
            var dim = shape[d]
            result[d] = rem % dim
            rem //= dim
    elif order == "F":
        for d in range(shape.ndim):
            var dim = shape[d]
            result[d] = rem % dim
            rem //= dim

    return result^


def unravel_index(
    indices: NDArray[DType.int], shape: NDArrayShape, order: String = "C"
) raises -> List[NDArray[DType.int]]:
    """Converts flat indices into coordinate arrays for `shape`.

    Args:
        indices: Flat linear indices.
        shape: Target shape.
        order: `"C"` for row-major order or `"F"` for column-major order.

    Returns:
        A list of coordinate arrays, one per dimension.

    Raises:
        Error: If any index is out of bounds for the flattened array.
        Error: If `order` is not `"C"` or `"F"`.

    Notes:
        Each output coordinate array has the same shape as `indices`.
    """
    var size = shape.size()
    if order != "C" and order != "F":
        raise Error(
            String(
                "\nError in `unravel_index`: order must be 'C' or 'F', got"
                " '{}'."
            ).format(order)
        )
    var indices_c = indices.contiguous()

    var result = List[NDArray[DType.int]]()
    for _ in range(shape.ndim):
        result.append(NDArray[DType.int](indices.shape))

    for i in range(indices_c.size):
        var raw = Int(indices_c.unsafe_get(i))
        if raw < 0 or raw >= size:
            raise Error(
                String(
                    "\nError in `unravel_index`: index {} is out of bounds for"
                    " array with size {}."
                ).format(raw, size)
            )

        var rem = raw
        if order == "C":
            for d in range(shape.ndim - 1, -1, -1):
                var dim = shape[d]
                result[d].unsafe_set(i, Scalar[DType.int](rem % dim))
                rem //= dim
        elif order == "F":
            for d in range(shape.ndim):
                var dim = shape[d]
                result[d].unsafe_set(i, Scalar[DType.int](rem % dim))
                rem //= dim

    return result^


def unravel_index(
    index: Int, shape: List[Int], order: String = "C"
) raises -> List[Int]:
    """Overload of `unravel_index` accepting a shape list."""
    return unravel_index(index, NDArrayShape(shape), order)


def unravel_index(
    indices: NDArray[DType.int], shape: List[Int], order: String = "C"
) raises -> List[NDArray[DType.int]]:
    """Overload of `unravel_index` accepting a shape list."""
    return unravel_index(indices, NDArrayShape(shape), order)


def ravel_multi_index(
    multi_index: List[NDArray[DType.int]],
    shape: NDArrayShape,
    order: String = "C",
) raises -> NDArray[DType.int]:
    """Converts coordinate arrays into flat indices for `shape`.

    Coordinate arrays are broadcast against each other. The result shape is the
    broadcast shape of those coordinate arrays.

    Args:
        multi_index: List of integer coordinate arrays, one per dimension.
        shape: Target shape.
        order: `"C"` for row-major order or `"F"` for column-major order.

    Returns:
        Integer array of flat linear indices.

    Raises:
        Error: If the number of coordinate arrays does not equal `shape.ndim`.
        Error: If coordinate arrays are not broadcast-compatible.
        Error: If any coordinate is out of bounds for its dimension.
        Error: If `order` is not `"C"` or `"F"`.
    """
    var n_idx = len(multi_index)
    if n_idx != shape.ndim:
        raise Error(
            String(
                "\nError in `ravel_multi_index`: expected {} coordinate"
                " arrays, got {}."
            ).format(shape.ndim, n_idx)
        )

    if order != "C" and order != "F":
        raise Error(
            String(
                "\nError in `ravel_multi_index`: order must be 'C' or 'F', got"
                " '{}'."
            ).format(order)
        )
    if n_idx == 0:
        raise Error(
            "\nError in `ravel_multi_index`: expected at least one coordinate"
            " array."
        )

    var out_shape = multi_index[0].shape
    for k in range(1, n_idx):
        try:
            out_shape = out_shape.broadcast(multi_index[k].shape)
        except e:
            raise Error(
                String(
                    "\nError in `ravel_multi_index`: coordinate arrays are not"
                    " broadcast-compatible: "
                )
                + String(e)
            )

    var bc_indices = List[NDArray[DType.int]](capacity=n_idx)
    for k in range(n_idx):
        bc_indices.append(broadcast_to(multi_index[k], out_shape).contiguous())

    var result = NDArray[DType.int](out_shape)
    for i in range(result.size):
        var flat = 0
        if order == "C":
            for d in range(shape.ndim):
                var coord = Int(bc_indices[d].unsafe_get(i))
                var dim = shape[d]
                if coord < 0 or coord >= dim:
                    raise Error(
                        String(
                            "\nError in `ravel_multi_index`: coordinate {} is"
                            " out of bounds for axis {} with size {}."
                        ).format(coord, d, dim)
                    )
                flat = flat * dim + coord
        elif order == "F":
            var stride = 1
            for d in range(shape.ndim):
                var coord = Int(bc_indices[d].unsafe_get(i))
                var dim = shape[d]
                if coord < 0 or coord >= dim:
                    raise Error(
                        String(
                            "\nError in `ravel_multi_index`: coordinate {} is"
                            " out of bounds for axis {} with size {}."
                        ).format(coord, d, dim)
                    )
                flat += coord * stride
                stride *= dim

        result.unsafe_set(i, Scalar[DType.int](flat))

    return result^


def ravel_multi_index(
    multi_index: List[NDArray[DType.int]],
    shape: List[Int],
    order: String = "C",
) raises -> NDArray[DType.int]:
    """Overload of `ravel_multi_index` accepting a shape list."""
    return ravel_multi_index(multi_index, NDArrayShape(shape), order)


def flatnonzero[
    dtype: DType,
    //,
](a: NDArray[dtype]) raises -> NDArray[DType.int]:
    """Returns flat indices of non-zero elements.

    Args:
        a: Input array.

    Returns:
        A 1-D integer array of flat linear indices where `a` is non-zero.

    Notes:
        Indices are reported in C-order over the flattened array.
    """
    var a_c = a.contiguous()

    var count: Int = 0
    for i in range(a_c.size):
        if a_c.unsafe_get(i) != 0:
            count += 1

    var result = NDArray[DType.int](NDArrayShape(count))
    var out_idx = 0
    for i in range(a_c.size):
        if a_c.unsafe_get(i) != 0:
            result.unsafe_set(out_idx, Scalar[DType.int](i))
            out_idx += 1

    return result^


def nonzero[
    dtype: DType,
    //,
](a: NDArray[dtype]) raises -> List[NDArray[DType.int]]:
    """Returns the indices of elements that are non-zero.

    Returns a list of 1-D index arrays, one per dimension of `a`. Each array
    contains the coordinates of non-zero elements along that dimension.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Input array.

    Returns:
        A `List` of `ndim` 1-D integer arrays. The i-th array contains the
        indices along dimension `i` of all non-zero elements.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.array[nm.i32]("[3, 0, 5, 0, 2]")
        var idx = nm.nonzero(a)
        print(idx[0])  # [0, 2, 4]

        var b = nm.array[nm.i32]("[[1, 0], [0, 4]]")
        var idx2 = nm.nonzero(b)
        print(idx2[0])  # [0, 1]  (row indices)
        print(idx2[1])  # [0, 1]  (col indices)
        ```
        .
    """
    var a_c = a.contiguous()

    var count: Int = 0
    for i in range(a_c.size):
        if a_c.unsafe_get(i) != 0:
            count += 1

    var result = List[NDArray[DType.int]]()
    for _ in range(a.ndim):
        result.append(NDArray[DType.int](NDArrayShape(count)))

    var out_idx = 0
    for flat in range(a_c.size):
        if a_c.unsafe_get(flat) != 0:
            var rem = flat
            for d in range(a.ndim - 1, -1, -1):
                var coord = rem % a.shape[d]
                rem //= a.shape[d]
                result[d].unsafe_set(out_idx, Scalar[DType.int](coord))
            out_idx += 1

    return result^


# ===----------------------------------------------------------------------=== #
# searchsorted
# ===----------------------------------------------------------------------=== #


def searchsorted[
    dtype: DType,
    //,
](
    a: NDArray[dtype], v: NDArray[dtype], side: String = "left"
) raises -> NDArray[DType.int]:
    """Finds indices where elements of `v` should be inserted into sorted
    1-D array `a` to keep it sorted.

    Uses binary search. `a` must be a 1-D array, assumed (not verified)
    to be sorted in ascending order.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: 1-D sorted source array.
        v: Array of values to find insertion indices for.
        side: `"left"` (default) returns the leftmost valid insertion index;
            `"right"` returns the rightmost.

    Returns:
        Array of insertion indices, same shape as `v`.

    Raises:
        Error: If `a` is not 1-D.
        Error: If `side` is not `"left"` or `"right"`.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.array[nm.i32]("[1, 3, 5, 7]")
        print(nm.indexing.searchsorted(a, nm.array[nm.i32]("[2, 6]")))
        # [1, 3]
        ```
        .
    """
    if a.ndim != 1:
        raise Error(
            String(
                "\nError in `searchsorted`: `a` must be a 1-D array, got {}"
                " dimensions."
            ).format(a.ndim)
        )
    if side != "left" and side != "right":
        raise Error(
            String(
                "\nError in `searchsorted`: `side` must be 'left' or"
                " 'right', got '{}'."
            ).format(side)
        )

    var a_c = a.contiguous()
    var v_c = v.contiguous()
    var n = a_c.size

    var result = NDArray[DType.int](Shape(v_c.shape))

    for i in range(v_c.size):
        var target = v_c.unsafe_get(i)
        var lo = 0
        var hi = n
        if side == "left":
            while lo < hi:
                var mid = (lo + hi) // 2
                if a_c.unsafe_get(mid) < target:
                    lo = mid + 1
                else:
                    hi = mid
        else:
            while lo < hi:
                var mid = (lo + hi) // 2
                if a_c.unsafe_get(mid) <= target:
                    lo = mid + 1
                else:
                    hi = mid
        result.unsafe_set(i, Scalar[DType.int](lo))

    return result^


def searchsorted[
    dtype: DType,
    //,
](a: NDArray[dtype], v: Scalar[dtype], side: String = "left") raises -> Int:
    """Finds the index where scalar `v` should be inserted into sorted 1-D
    array `a` to keep it sorted.

    This is a function ***OVERLOAD*** of `searchsorted` for a scalar `v`.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: 1-D sorted source array.
        v: Scalar value to find the insertion index for.
        side: `"left"` (default) returns the leftmost valid insertion index;
            `"right"` returns the rightmost.

    Returns:
        Insertion index.

    Raises:
        Error: If `a` is not 1-D.
        Error: If `side` is not `"left"` or `"right"`.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.array[nm.i32]("[1, 3, 5, 7]")
        print(nm.indexing.searchsorted(a, Scalar[nm.i32](4)))
        # 2
        ```
        .
    """
    if a.ndim != 1:
        raise Error(
            String(
                "\nError in `searchsorted`: `a` must be a 1-D array, got {}"
                " dimensions."
            ).format(a.ndim)
        )
    if side != "left" and side != "right":
        raise Error(
            String(
                "\nError in `searchsorted`: `side` must be 'left' or"
                " 'right', got '{}'."
            ).format(side)
        )

    var a_c = a.contiguous()
    var n = a_c.size
    var lo = 0
    var hi = n
    if side == "left":
        while lo < hi:
            var mid = (lo + hi) // 2
            if a_c.unsafe_get(mid) < v:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            var mid = (lo + hi) // 2
            if a_c.unsafe_get(mid) <= v:
                lo = mid + 1
            else:
                hi = mid
    return lo
