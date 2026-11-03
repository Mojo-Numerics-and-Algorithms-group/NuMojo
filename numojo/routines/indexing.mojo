# ===----------------------------------------------------------------------=== #
# NuMojo: Indexing routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Indexing routines (numojo.routines.indexing).
==============================================
Advanced indexing operations for arrays.

Functions for generating index arrays, fancy indexing, selecting elements,
and inserting data into arrays.

Exports
-------
- `where`: Conditional element selection.
- `compress`: Extract elements based on condition.
- `take`, `take_along_axis`: Advanced indexing.
- `nonzero`, `flatnonzero`: Find non-zero elements.
- `fancy_index`: Apply fancy indexing.
- `unravel_index`, `ravel_multi_index`: Index conversion.
- `indices`, `ix_`: Index-grid generators.
- `diag_indices`, `diag_indices_from`: Diagonal index generators.
- `tril_indices`, `tril_indices_from`, `triu_indices`, `triu_indices_from`,
  `mask_indices`: Triangular index generators.
- `argwhere`: Indices of non-zero elements, grouped by element.
- `fill_diagonal`: In-place diagonal fill.
- `insert`, `place`, `extract`, `put_along_axis`, `putmask`, `choose`:
  Element insertion/selection helpers.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.algorithm import vectorize
from std.memory import unsafe_memcpy
from std.sys import simd_width_of

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
import numojo.routines.manipulation as manipulation
from numojo import broadcast_to
from numojo.core.error import NumojoError
from numojo.core.indexing import IndexMethods
from numojo.core.layout import NDArrayShape, NDArrayStrides
from numojo.core.ndarray import NDArray
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
        NumojoError: If the shapes of `x` and `y` do not match.

    Parameters:
        dtype: DType.

    Args:
        x: NDArray[dtype].
        y: NDArray[dtype].
        mask: NDArray[DType.bool].

    """
    if x.shape != y.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "Shape mismatch error: x and y must have the same shape"
                ),
                location="where",
            )
        )

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
        NumojoError: If ``condition``, ``x``, and ``y`` are not broadcast-compatible.

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
        NumojoError: If ``condition`` and `x` are not broadcast-compatible.

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
        NumojoError: If ``condition`` and `y` are not broadcast-compatible.

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
        NumojoError: If the number of index arrays does not equal `a.ndim`.
        NumojoError: If the index arrays are not mutually broadcast-compatible.
        NumojoError: If any index value is out of bounds for its axis.

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
            NumojoError(
                category="value",
                message=String(
                    "\nError in `fancy_index`: expected {} index arrays (one"
                    " per axis), got {}."
                ).format(a.ndim, n_idx),
                location="fancy_index",
            )
        )

    # Broadcast all index arrays to a common shape.
    var out_shape = index_arrays[0].shape
    for k in range(1, n_idx):
        try:
            out_shape = out_shape.broadcast(index_arrays[k].shape)
        except e:
            raise Error(
                NumojoError(
                    category="broadcast",
                    message=String(
                        "\nError in `fancy_index`: index arrays are not"
                        " broadcast-compatible: "
                    )
                    + String(e),
                    location="fancy_index",
                )
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
                    NumojoError(
                        category="index",
                        message=String(
                            "\nError in `fancy_index`: index {} is out of"
                            " bounds for axis {} with size {}."
                        ).format(raw, k, ax_size),
                        location="fancy_index",
                    )
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
        NumojoError: If the number of index arrays does not equal `a.ndim`.
        NumojoError: If the index arrays are not mutually broadcast-compatible.
        NumojoError: If any index value is out of bounds for its axis.

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
        NumojoError: If the axis is out of bound for the given array.
        NumojoError: If the condition is not 1-D array.
        NumojoError: If the condition length is out of bound for the given axis.

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
            NumojoError(
                category="index",
                message=String(
                    "\nError in `compress`: Axis {} is out of bound for array"
                    " with {} dimensions"
                ).format(axis, a.ndim),
                location="compress",
            )
        )

    if condition.ndim != 1:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "\nError in `compress`: Condition must be 1-D array, got {}"
                ).format(condition.ndim),
                location="compress",
            )
        )
    if condition.size > a.shape[normalized_axis]:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "\nError in `compress`: Condition length {} is out of bound"
                    " for axis {} with size {}"
                ).format(condition.size, axis, a.shape[normalized_axis]),
                location="compress",
            )
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
            res_strides.unsafe_set(i, temp)
            temp *= Scalar[DType.int](result.shape[i])
    res_strides.unsafe_set(normalized_axis, temp)

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
                item.unsafe_set(j, remainder // res_strides.unsafe_load(j))
                remainder %= res_strides.unsafe_load(j)

                # Then along other axes
                for j in range(result.ndim):
                    if j != normalized_axis:
                        item.unsafe_set(
                            j, remainder // res_strides.unsafe_load(j)
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
        NumojoError: If the condition is not 1-D array.
        NumojoError: If the condition length is out of bound for the given axis.

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
            NumojoError(
                category="value",
                message=String(
                    "\nError in `compress`: Condition must be 1-D array, got {}"
                ).format(condition.ndim),
                location="compress",
            )
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
        NumojoError: If the axis is out of bounds for the given array.
        NumojoError: If the ndim of arr and indices are not the same.
        NumojoError: If the shape of indices does not match the shape of the
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
            NumojoError(
                category="index",
                message=String(
                    "\nError in `take_along_axis`: Axis {} is out of bound for"
                    " array with {} dimensions"
                ).format(axis, arr.ndim),
                location="take_along_axis",
            )
        )

    # Check if the ndim of arr and indices are same
    if arr.ndim != indices.ndim:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "\nError in `take_along_axis`: The ndim of arr and indices"
                    " must be same. Got {} and {}."
                ).format(arr.ndim, indices.ndim),
                location="take_along_axis",
            )
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
                NumojoError(
                    category="broadcast",
                    message=String(
                        "\nError in `take_along_axis`: Shape of indices must"
                        " match shape of array except along the given axis. "
                        + String(e)
                    ),
                    location="take_along_axis",
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
        NumojoError: If `axis` is out of bounds.
        NumojoError: If any index is out of bounds for the given axis.

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
            NumojoError(
                category="index",
                message=String(
                    "\nError in `take`: axis {} is out of bounds for array with"
                    " {} dimensions."
                ).format(axis, a.ndim),
                location="take",
            )
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
                    NumojoError(
                        category="index",
                        message=String(
                            "\nError in `take`: index {} is out of bounds for"
                            " axis {} with size {}."
                        ).format(raw, norm_axis, axis_size),
                        location="take",
                    )
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
        NumojoError: If any index is out of bounds for the flattened array.

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
        NumojoError: If any index is out of bounds for the flattened array.
        NumojoError: If `values` is empty while `indices` is not.

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
            NumojoError(
                category="value",
                message=String(
                    "\nError in `put`: values is empty but indices has {}"
                    " element(s)."
                ).format(indices.size),
                location="put",
            )
        )

    var indices_c = indices.contiguous()
    var values_c = values.contiguous()

    for i in range(indices_c.size):
        var raw = Int(indices_c.unsafe_get(i))
        if raw < -a.size or raw >= a.size:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "\nError in `put`: index {} is out of bounds for array"
                        " of size {}."
                    ).format(raw, a.size),
                    location="put",
                )
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
        NumojoError: If any index is out of bounds for the flattened array.

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
                NumojoError(
                    category="index",
                    message=String(
                        "\nError in `put`: index {} is out of bounds for array"
                        " of size {}."
                    ).format(raw, a.size),
                    location="put",
                )
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
        NumojoError: If `index` is out of bounds for the flattened array.
        NumojoError: If `order` is not `"C"` or `"F"`.
    """
    var size = shape.size()
    if order != "C" and order != "F":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "\nError in `unravel_index`: order must be 'C' or 'F', got"
                    " '{}'."
                ).format(order),
                location="unravel_index",
            )
        )
    if index < 0 or index >= size:
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "\nError in `unravel_index`: index {} is out of bounds for"
                    " array with size {}."
                ).format(index, size),
                location="unravel_index",
            )
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
        NumojoError: If any index is out of bounds for the flattened array.
        NumojoError: If `order` is not `"C"` or `"F"`.

    Notes:
        Each output coordinate array has the same shape as `indices`.
    """
    var size = shape.size()
    if order != "C" and order != "F":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "\nError in `unravel_index`: order must be 'C' or 'F', got"
                    " '{}'."
                ).format(order),
                location="unravel_index",
            )
        )
    var indices_c = indices.contiguous()

    var result = List[NDArray[DType.int]]()
    for _ in range(shape.ndim):
        result.append(NDArray[DType.int](indices.shape))

    for i in range(indices_c.size):
        var raw = Int(indices_c.unsafe_get(i))
        if raw < 0 or raw >= size:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "\nError in `unravel_index`: index {} is out of bounds"
                        " for array with size {}."
                    ).format(raw, size),
                    location="unravel_index",
                )
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
        NumojoError: If the number of coordinate arrays does not equal `shape.ndim`.
        NumojoError: If coordinate arrays are not broadcast-compatible.
        NumojoError: If any coordinate is out of bounds for its dimension.
        NumojoError: If `order` is not `"C"` or `"F"`.
    """
    var n_idx = len(multi_index)
    if n_idx != shape.ndim:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "\nError in `ravel_multi_index`: expected {} coordinate"
                    " arrays, got {}."
                ).format(shape.ndim, n_idx),
                location="ravel_multi_index",
            )
        )

    if order != "C" and order != "F":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "\nError in `ravel_multi_index`: order must be 'C' or 'F',"
                    " got '{}'."
                ).format(order),
                location="ravel_multi_index",
            )
        )
    if n_idx == 0:
        raise Error(
            NumojoError(
                category="value",
                message=(
                    "\nError in `ravel_multi_index`: expected at least one"
                    " coordinate array."
                ),
                location="ravel_multi_index",
            )
        )

    var out_shape = multi_index[0].shape
    for k in range(1, n_idx):
        try:
            out_shape = out_shape.broadcast(multi_index[k].shape)
        except e:
            raise Error(
                NumojoError(
                    category="broadcast",
                    message=String(
                        "\nError in `ravel_multi_index`: coordinate arrays are"
                        " not broadcast-compatible: "
                    )
                    + String(e),
                    location="ravel_multi_index",
                )
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
                        NumojoError(
                            category="index",
                            message=String(
                                "\nError in `ravel_multi_index`: coordinate {}"
                                " is out of bounds for axis {} with size {}."
                            ).format(coord, d, dim),
                            location="ravel_multi_index",
                        )
                    )
                flat = flat * dim + coord
        elif order == "F":
            var stride = 1
            for d in range(shape.ndim):
                var coord = Int(bc_indices[d].unsafe_get(i))
                var dim = shape[d]
                if coord < 0 or coord >= dim:
                    raise Error(
                        NumojoError(
                            category="index",
                            message=String(
                                "\nError in `ravel_multi_index`: coordinate {}"
                                " is out of bounds for axis {} with size {}."
                            ).format(coord, d, dim),
                            location="ravel_multi_index",
                        )
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
        NumojoError: If `a` is not 1-D.
        NumojoError: If `side` is not `"left"` or `"right"`.

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
            NumojoError(
                category="shape",
                message=String(
                    "\nError in `searchsorted`: `a` must be a 1-D array, got {}"
                    " dimensions."
                ).format(a.ndim),
                location="searchsorted",
            )
        )
    if side != "left" and side != "right":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "\nError in `searchsorted`: `side` must be 'left' or"
                    " 'right', got '{}'."
                ).format(side),
                location="searchsorted",
            )
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
        NumojoError: If `a` is not 1-D.
        NumojoError: If `side` is not `"left"` or `"right"`.

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
            NumojoError(
                category="shape",
                message=String(
                    "\nError in `searchsorted`: `a` must be a 1-D array, got {}"
                    " dimensions."
                ).format(a.ndim),
                location="searchsorted",
            )
        )
    if side != "left" and side != "right":
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "\nError in `searchsorted`: `side` must be 'left' or"
                    " 'right', got '{}'."
                ).format(side),
                location="searchsorted",
            )
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


# ===----------------------------------------------------------------------=== #
# Index generators
# ===----------------------------------------------------------------------=== #


def indices(dimensions: List[Int]) raises -> NDArray[DType.int]:
    """Returns an array representing the indices of a grid.

    Args:
        dimensions: The shape of the grid.

    Returns:
        An integer array of shape `[len(dimensions)] + dimensions` whose
        `result[d]` sub-array contains the index along dimension `d` at
        every grid point.

    Examples:
        ```mojo
        import numojo as nm

        var grid = nm.indices(List[Int](2, 3))
        print(grid.shape)
        # (2, 2, 3)
        ```
        .
    """
    var ndim = len(dimensions)
    var inner_size = 1
    for d in range(ndim):
        inner_size *= dimensions[d]

    var out_shape_list = List[Int](capacity=ndim + 1)
    out_shape_list.append(ndim)
    for d in range(ndim):
        out_shape_list.append(dimensions[d])

    var result = NDArray[DType.int](NDArrayShape(out_shape_list))

    for dim_idx in range(ndim):
        for flat in range(inner_size):
            var rem = flat
            var coord = 0
            for d in range(ndim - 1, -1, -1):
                var dim_size = dimensions[d]
                var c = rem % dim_size
                rem //= dim_size
                if d == dim_idx:
                    coord = c
            result.unsafe_set(
                dim_idx * inner_size + flat, Scalar[DType.int](coord)
            )

    return result^


def ix_[
    dtype: DType,
    //,
](arrays: List[NDArray[dtype]]) raises -> List[NDArray[dtype]]:
    """Constructs an open mesh from 1-D arrays for cross-product indexing.

    Each input array is reshaped so that it has size along its own axis and
    size 1 along every other axis, enabling broadcasting into a dense grid.

    Parameters:
        dtype: Data type of the input arrays.

    Args:
        arrays: A `List` of 1-D arrays, one per dimension of the desired grid.

    Returns:
        A `List` of the same length, with array `k` reshaped to shape
        `(1, ..., len(arrays[k]), ..., 1)` (the non-1 entry at position `k`).

    Raises:
        NumojoError: If any input array is not 1-D.

    Examples:
        ```mojo
        import numojo as nm

        var rows = nm.array[nm.int]("[0, 1, 2]")
        var cols = nm.array[nm.int]("[0, 1]")
        var mesh = List[nm.NDArray[nm.int]]()
        mesh.append(rows^)
        mesh.append(cols^)
        var grid = nm.indexing.ix_(mesh)
        # grid[0] has shape (3, 1); grid[1] has shape (1, 2)
        ```
        .
    """
    var n = len(arrays)
    var result = List[NDArray[dtype]]()
    for k in range(n):
        if arrays[k].ndim != 1:
            raise Error(
                NumojoError(
                    category="shape",
                    message=String(
                        "\nError in `ix_`: array {} must be 1-D, got {}"
                        " dimensions."
                    ).format(k, arrays[k].ndim),
                    location="ix_",
                )
            )
        var shape_list = List[Int](capacity=n)
        for j in range(n):
            if j == k:
                shape_list.append(arrays[k].size)
            else:
                shape_list.append(1)
        result.append(
            manipulation.reshape(arrays[k], NDArrayShape(shape_list))
        )
    return result^


def diag_indices(n: Int, ndim: Int = 2) raises -> List[NDArray[DType.int]]:
    """Returns indices to access the main diagonal of an `ndim`-D array.

    Args:
        n: The length of each dimension.
        ndim: The number of dimensions. Default is 2.

    Returns:
        A `List` of `ndim` 1-D integer arrays, each containing `0, 1, ..., n-1`.

    Examples:
        ```mojo
        import numojo as nm

        var idx = nm.indexing.diag_indices(3)
        print(idx[0])  # [0, 1, 2]
        print(idx[1])  # [0, 1, 2]
        ```
        .
    """
    var result = List[NDArray[DType.int]]()
    for _ in range(ndim):
        var arr = NDArray[DType.int](NDArrayShape(n))
        for i in range(n):
            arr.unsafe_set(i, Scalar[DType.int](i))
        result.append(arr^)
    return result^


def diag_indices_from[
    dtype: DType,
    //,
](a: NDArray[dtype]) raises -> List[NDArray[DType.int]]:
    """Returns indices to access the main diagonal of an N-D array `a`.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Array whose diagonal indices are computed. Must have at least 2
            dimensions, all of the same length.

    Returns:
        A `List` of `a.ndim` 1-D integer arrays, each containing
        `0, 1, ..., n-1` where `n` is the common dimension length.

    Raises:
        NumojoError: If `a` has fewer than 2 dimensions.
        NumojoError: If the dimensions of `a` are not all equal.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.zeros[nm.i32](nm.Shape(3, 3))
        var idx = nm.indexing.diag_indices_from(a)
        print(idx[0])  # [0, 1, 2]
        ```
        .
    """
    if a.ndim < 2:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "\nError in `diag_indices_from`: array must be at least"
                    " 2-D."
                ),
                location="diag_indices_from",
            )
        )
    for d in range(1, a.ndim):
        if a.shape[d] != a.shape[0]:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "\nError in `diag_indices_from`: all dimensions of"
                        " the input must be of equal length."
                    ),
                    location="diag_indices_from",
                )
            )
    return diag_indices(a.shape[0], a.ndim)


def tril_indices(
    n: Int, k: Int = 0, m: Optional[Int] = None
) raises -> Tuple[NDArray[DType.int], NDArray[DType.int]]:
    """Returns the indices of the lower triangle of an `(n, m)` array.

    Args:
        n: Number of rows.
        k: Diagonal offset. `k=0` is the main diagonal; `k>0` is above it;
            `k<0` is below it.
        m: Number of columns. Defaults to `n`.

    Returns:
        A tuple `(rows, cols)` of 1-D integer arrays giving the coordinates
        of the lower-triangular entries, in row-major order.

    Examples:
        ```mojo
        import numojo as nm

        var rc = nm.indexing.tril_indices(3)
        print(rc[0])  # [0, 1, 1, 2, 2, 2]
        print(rc[1])  # [0, 0, 1, 0, 1, 2]
        ```
        .
    """
    var ncols = m.value() if m else n
    var rows = List[Int]()
    var cols = List[Int]()
    for i in range(n):
        for j in range(ncols):
            if j <= i + k:
                rows.append(i)
                cols.append(j)

    var row_arr = NDArray[DType.int](NDArrayShape(len(rows)))
    var col_arr = NDArray[DType.int](NDArrayShape(len(rows)))
    for idx in range(len(rows)):
        row_arr.unsafe_set(idx, Scalar[DType.int](rows[idx]))
        col_arr.unsafe_set(idx, Scalar[DType.int](cols[idx]))

    return Tuple(row_arr^, col_arr^)


def triu_indices(
    n: Int, k: Int = 0, m: Optional[Int] = None
) raises -> Tuple[NDArray[DType.int], NDArray[DType.int]]:
    """Returns the indices of the upper triangle of an `(n, m)` array.

    Args:
        n: Number of rows.
        k: Diagonal offset. `k=0` is the main diagonal; `k>0` is above it;
            `k<0` is below it.
        m: Number of columns. Defaults to `n`.

    Returns:
        A tuple `(rows, cols)` of 1-D integer arrays giving the coordinates
        of the upper-triangular entries, in row-major order.

    Examples:
        ```mojo
        import numojo as nm

        var rc = nm.indexing.triu_indices(3)
        print(rc[0])  # [0, 0, 0, 1, 1, 2]
        print(rc[1])  # [0, 1, 2, 1, 2, 2]
        ```
        .
    """
    var ncols = m.value() if m else n
    var rows = List[Int]()
    var cols = List[Int]()
    for i in range(n):
        for j in range(ncols):
            if j >= i + k:
                rows.append(i)
                cols.append(j)

    var row_arr = NDArray[DType.int](NDArrayShape(len(rows)))
    var col_arr = NDArray[DType.int](NDArrayShape(len(rows)))
    for idx in range(len(rows)):
        row_arr.unsafe_set(idx, Scalar[DType.int](rows[idx]))
        col_arr.unsafe_set(idx, Scalar[DType.int](cols[idx]))

    return Tuple(row_arr^, col_arr^)


def tril_indices_from[
    dtype: DType,
    //,
](a: NDArray[dtype], k: Int = 0) raises -> Tuple[
    NDArray[DType.int], NDArray[DType.int]
]:
    """Returns the indices of the lower triangle of 2-D array `a`.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: A 2-D array.
        k: Diagonal offset. `k=0` is the main diagonal.

    Returns:
        A tuple `(rows, cols)` of 1-D integer arrays.

    Raises:
        NumojoError: If `a` is not 2-D.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.zeros[nm.i32](nm.Shape(3, 4))
        var rc = nm.indexing.tril_indices_from(a)
        print(rc[0])  # [0, 1, 1, 2, 2, 2]
        ```
        .
    """
    if a.ndim != 2:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "\nError in `tril_indices_from`: array must be 2-D."
                ),
                location="tril_indices_from",
            )
        )
    return tril_indices(a.shape[0], k, a.shape[1])


def triu_indices_from[
    dtype: DType,
    //,
](a: NDArray[dtype], k: Int = 0) raises -> Tuple[
    NDArray[DType.int], NDArray[DType.int]
]:
    """Returns the indices of the upper triangle of 2-D array `a`.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: A 2-D array.
        k: Diagonal offset. `k=0` is the main diagonal.

    Returns:
        A tuple `(rows, cols)` of 1-D integer arrays.

    Raises:
        NumojoError: If `a` is not 2-D.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.zeros[nm.i32](nm.Shape(3, 4))
        var rc = nm.indexing.triu_indices_from(a)
        print(rc[0])  # [0, 0, 0, 0, 1, 1, 1, 2, 2]
        ```
        .
    """
    if a.ndim != 2:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "\nError in `triu_indices_from`: array must be 2-D."
                ),
                location="triu_indices_from",
            )
        )
    return triu_indices(a.shape[0], k, a.shape[1])


def mask_indices(
    n: Int, mask_func: String, k: Int = 0
) raises -> Tuple[NDArray[DType.int], NDArray[DType.int]]:
    """Returns indices to access `(n, n)` arrays, given a masking function.

    Args:
        n: The returned indices apply to an `(n, n)` array.
        mask_func: Either `"triu"` or `"tril"`.
        k: Diagonal offset forwarded to the masking function.

    Returns:
        A tuple `(rows, cols)` of 1-D integer arrays.

    Raises:
        NumojoError: If `mask_func` is not `"triu"` or `"tril"`.

    Notes:
        NumPy's `mask_indices` accepts an arbitrary callable `mask_func`.
        Mojo's type system cannot cleanly accept a bound Python-style
        callable here, so this is a deliberate compatibility deviation:
        `mask_func` is a `String`, either `"triu"` or `"tril"`, naming the
        masking function to apply.

    Examples:
        ```mojo
        import numojo as nm

        var rc = nm.indexing.mask_indices(4, "triu")
        print(rc[0])
        ```
        .
    """
    if mask_func == "triu":
        return triu_indices(n, k)
    elif mask_func == "tril":
        return tril_indices(n, k)
    else:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "\nError in `mask_indices`: mask_func must be 'triu' or"
                    " 'tril', got '{}'."
                ).format(mask_func),
                location="mask_indices",
            )
        )


def argwhere[
    dtype: DType,
    //,
](a: NDArray[dtype]) raises -> NDArray[DType.int]:
    """Returns the indices of non-zero elements, grouped by element.

    Parameters:
        dtype: Data type of the source array.

    Args:
        a: Input array.

    Returns:
        An integer array of shape `(N, a.ndim)` where `N` is the number of
        non-zero elements; row `i` gives the coordinates of the `i`-th
        non-zero element.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.array[nm.i32]("[[1, 0], [0, 4]]")
        print(nm.indexing.argwhere(a))
        # [[0 0]
        #  [1 1]]
        ```
        .
    """
    var idx = nonzero(a)
    var ndim = a.ndim
    var n = idx[0].size if ndim > 0 else 0

    var out_shape_list = List[Int]()
    out_shape_list.append(n)
    out_shape_list.append(ndim)
    var result = NDArray[DType.int](NDArrayShape(out_shape_list))
    for row in range(n):
        for d in range(ndim):
            result.unsafe_set(row * ndim + d, idx[d].unsafe_get(row))

    return result^


def fill_diagonal[
    dtype: DType,
    //,
](mut a: NDArray[dtype], val: Scalar[dtype], wrap: Bool = False) raises:
    """Fills the main diagonal of `a` in-place with `val`.

    For arrays with more than 2 dimensions, all dimensions must have equal
    length, and `wrap` has no effect (matching NumPy).

    Parameters:
        dtype: Data type of the array.

    Args:
        a: Array to modify in-place. Must have at least 2 dimensions.
        val: Value written to each diagonal entry.
        wrap: For tall (`rows > cols`) 2-D arrays only: if `True`, the
            diagonal "wraps" after `cols` columns and continues filling
            every `cols + 1`-th flat element through the end of the array.

    Raises:
        NumojoError: If `a` has fewer than 2 dimensions.
        NumojoError: If `a.ndim > 2` and its dimensions are not all equal.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.zeros[nm.i32](nm.Shape(3, 3))
        nm.indexing.fill_diagonal(a, Scalar[nm.i32](1))
        print(a)
        # [[1 0 0]
        #  [0 1 0]
        #  [0 0 1]]
        ```
        .
    """
    if a.ndim < 2:
        raise Error(
            NumojoError(
                category="shape",
                message="\nError in `fill_diagonal`: array must be at least 2-D.",
                location="fill_diagonal",
            )
        )

    var step: Int
    var end: Int
    if a.ndim == 2:
        var ncols = a.shape[1]
        step = ncols + 1
        if wrap:
            end = a.size
        else:
            end = ncols * ncols
    else:
        for d in range(1, a.ndim):
            if a.shape[d] != a.shape[0]:
                raise Error(
                    NumojoError(
                        category="shape",
                        message=(
                            "\nError in `fill_diagonal`: all dimensions must"
                            " be of equal length for ndim > 2."
                        ),
                        location="fill_diagonal",
                    )
                )
        var cumulative = 1
        step = 1
        for d in range(a.ndim - 1):
            cumulative *= a.shape[d]
            step += cumulative
        end = a.size

    if end > a.size:
        end = a.size

    var idx = 0
    while idx < end:
        a.itemset(idx, val)
        idx += step


# ===----------------------------------------------------------------------=== #
# Insertion / extraction helpers
# ===----------------------------------------------------------------------=== #


def insert[
    dtype: DType,
    //,
](
    a: NDArray[dtype],
    index: Int,
    values: NDArray[dtype],
    axis: Optional[Int] = None,
) raises -> NDArray[dtype]:
    """Inserts `values` into `a` before position `index`.

    Parameters:
        dtype: Data type of the array.

    Args:
        a: Source array.
        index: Insertion position along `axis` (or the flattened array, if
            `axis` is `None`). Negative values count from the end.
        values: Values to insert.
        axis: Axis along which to insert. If `None` (default), `a` is
            flattened before insertion and the result is 1-D.

    Returns:
        A new array with `values` inserted.

    Raises:
        NumojoError: If `axis` is out of bounds for `a`.

    Notes:
        When `axis` is given, `values` must either be broadcastable to a
        single slice (any array whose size along `axis` is 1, or which has
        fewer dimensions than `a`) or match `a`'s shape exactly along every
        axis other than `axis`. General NumPy-style broadcasting across the
        non-insertion axes is not supported.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.array[nm.i32]("[1, 2, 3]")
        print(nm.indexing.insert(a, 1, nm.array[nm.i32]("[9]")))
        # [1, 9, 2, 3]
        ```
        .
    """
    if not axis:
        var flat_a = ravel(a)
        var flat_v = ravel(values)
        var n = flat_a.size
        var v = flat_v.size

        var idx = index
        if idx < 0:
            idx += n
        if idx < 0:
            idx = 0
        if idx > n:
            idx = n

        var result = NDArray[dtype](NDArrayShape(n + v))
        for i in range(idx):
            result.unsafe_set(i, flat_a.unsafe_get(i))
        for i in range(v):
            result.unsafe_set(idx + i, flat_v.unsafe_get(i))
        for i in range(idx, n):
            result.unsafe_set(idx + v + (i - idx), flat_a.unsafe_get(i))
        return result^

    var norm_axis = axis.value()
    if norm_axis < 0:
        norm_axis += a.ndim
    if norm_axis < 0 or norm_axis >= a.ndim:
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "\nError in `insert`: axis {} is out of bounds for array"
                    " with {} dimensions."
                ).format(axis.value(), a.ndim),
                location="insert",
            )
        )

    var axis_size = a.shape[norm_axis]
    var vcount = values.shape[norm_axis] if values.ndim == a.ndim else 1

    var idx = index
    if idx < 0:
        idx += axis_size
    if idx < 0:
        idx = 0
    if idx > axis_size:
        idx = axis_size

    var out_shape_list = List[Int](capacity=a.ndim)
    for d in range(a.ndim):
        if d == norm_axis:
            out_shape_list.append(axis_size + vcount)
        else:
            out_shape_list.append(a.shape[d])

    var result = NDArray[dtype](NDArrayShape(out_shape_list))
    var a_c = a.contiguous()
    var values_c = values.contiguous()
    var values_is_full = values.ndim == a.ndim

    for flat in range(result.size):
        var rem = flat
        var coords = List[Int](capacity=a.ndim)
        for _ in range(a.ndim):
            coords.append(0)
        for d in range(result.ndim - 1, -1, -1):
            coords[d] = rem % result.shape[d]
            rem //= result.shape[d]

        var c = coords[norm_axis]
        if c < idx:
            result.unsafe_set(flat, a_c._getitem(coords))
        elif c < idx + vcount:
            if values_is_full:
                var vcoords = coords.copy()
                vcoords[norm_axis] = c - idx
                result.unsafe_set(flat, values_c._getitem(vcoords))
            else:
                result.unsafe_set(flat, values_c.unsafe_get(0))
        else:
            var acoords = coords.copy()
            acoords[norm_axis] = c - vcount
            result.unsafe_set(flat, a_c._getitem(acoords))

    return result^


def place[
    dtype: DType,
    //,
](mut a: NDArray[dtype], mask: NDArray[DType.bool], vals: NDArray[dtype]) raises:
    """Changes elements of `a` based on `mask`, cyclically using `vals`.

    Unlike `putmask`, `vals` is only consumed (and cycled) for positions
    actually selected by `mask`, matching `numpy.place`.

    Parameters:
        dtype: Data type of the array.

    Args:
        a: Array to modify in-place.
        mask: Boolean array, same shape as `a`.
        vals: Values cyclically used to fill selected positions.

    Raises:
        NumojoError: If `mask`'s shape does not match `a`'s shape.
        NumojoError: If `vals` is empty while `mask` selects at least one
            position.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.i32](6)
        var mask = nm.array[nm.boolean]("[True, False, True, False, True, False]")
        nm.indexing.place(a, mask, nm.array[nm.i32]("[10, 20]"))
        print(a)
        # [10, 1, 20, 3, 10, 5]
        ```
        .
    """
    if a.shape != mask.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "\nError in `place`: mask must have the same shape as a."
                ),
                location="place",
            )
        )

    var mask_c = mask.contiguous()

    if vals.size == 0:
        for i in range(a.size):
            if mask_c.unsafe_get(i):
                raise Error(
                    NumojoError(
                        category="value",
                        message=(
                            "\nError in `place`: vals is empty but mask"
                            " selects at least one position."
                        ),
                        location="place",
                    )
                )
        return

    var vals_c = vals.contiguous()
    var vi = 0
    for i in range(a.size):
        if mask_c.unsafe_get(i):
            a.itemset(i, vals_c.unsafe_get(vi % vals_c.size))
            vi += 1


def extract[
    dtype: DType,
    //,
](condition: NDArray[DType.bool], a: NDArray[dtype]) raises -> NDArray[dtype]:
    """Returns the elements of `a` where `condition` is `True`.

    Equivalent to `compress(ravel(condition), ravel(a))`.

    Parameters:
        dtype: Data type of the array.

    Args:
        condition: Boolean selector, same size as `a`.
        a: Source array.

    Returns:
        A 1-D array of the selected elements.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.i32](6)
        var mask = nm.array[nm.boolean]("[True, False, True, False, True, False]")
        print(nm.indexing.extract(mask, a))
        # [0, 2, 4]
        ```
        .
    """
    var cond_flat = ravel(condition)
    var a_flat = ravel(a)
    return compress(cond_flat, a_flat, axis=0)


def put_along_axis[
    dtype: DType,
    //,
](
    mut a: NDArray[dtype],
    indices: NDArray[DType.int],
    values: NDArray[dtype],
    axis: Int,
) raises:
    """Puts values into `a` at the given 1-D indices along `axis`.

    The inverse counterpart of `take_along_axis`.

    Parameters:
        dtype: Data type of the array.

    Args:
        a: Destination array to modify in-place.
        indices: Indices array, broadcastable to `a`'s shape with the size
            along `axis` replaced by `indices.shape[axis]`.
        values: Values array, broadcast to the same shape as `indices`.
        axis: Axis along which to place values.

    Raises:
        NumojoError: If `axis` is out of bounds for `a`.
        NumojoError: If `indices.ndim` does not equal `a.ndim`.
        NumojoError: If any index is out of bounds for the given axis.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.zeros[nm.i32](nm.Shape(2, 3))
        var ind = nm.array[nm.int]("[[0], [2]]")
        var vals = nm.array[nm.i32]("[[9], [9]]")
        nm.indexing.put_along_axis(a, ind, vals, axis=1)
        print(a)
        # [[9 0 0]
        #  [0 0 9]]
        ```
        .
    """
    var norm_axis = axis
    if norm_axis < 0:
        norm_axis += a.ndim
    if norm_axis < 0 or norm_axis >= a.ndim:
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "\nError in `put_along_axis`: axis {} is out of bounds"
                    " for array with {} dimensions."
                ).format(axis, a.ndim),
                location="put_along_axis",
            )
        )
    if indices.ndim != a.ndim:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "\nError in `put_along_axis`: the ndim of a and indices"
                    " must be same. Got {} and {}."
                ).format(a.ndim, indices.ndim),
                location="put_along_axis",
            )
        )

    var bc_shape = a.shape
    bc_shape[norm_axis] = indices.shape[norm_axis]
    var indices_bc = broadcast_to(indices, bc_shape).contiguous()
    var values_bc = broadcast_to(values, bc_shape).contiguous()

    var axis_size = a.shape[norm_axis]

    for flat in range(indices_bc.size):
        var rem = flat
        var coords = List[Int](capacity=a.ndim)
        for _ in range(a.ndim):
            coords.append(0)
        for d in range(indices_bc.ndim - 1, -1, -1):
            coords[d] = rem % indices_bc.shape[d]
            rem //= indices_bc.shape[d]

        var raw = Int(indices_bc.unsafe_get(flat))
        if raw < -axis_size or raw >= axis_size:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "\nError in `put_along_axis`: index {} is out of"
                        " bounds for axis {} with size {}."
                    ).format(raw, norm_axis, axis_size),
                    location="put_along_axis",
                )
            )
        if raw < 0:
            raw += axis_size
        coords[norm_axis] = raw

        a.itemset(coords.copy(), values_bc.unsafe_get(flat))


def putmask[
    dtype: DType,
    //,
](mut a: NDArray[dtype], mask: NDArray[DType.bool], values: NDArray[dtype]) raises:
    """Changes elements of `a` based on `mask`, tiling `values` over `a`'s shape.

    Unlike `place`, `values` is conceptually tiled to `a`'s full size first
    (indexed by position in `a`, not by selection order), matching
    `numpy.putmask`.

    Parameters:
        dtype: Data type of the array.

    Args:
        a: Array to modify in-place.
        mask: Boolean array, same shape as `a`.
        values: Values tiled (cyclically repeated) over `a`'s positions.

    Raises:
        NumojoError: If `mask`'s shape does not match `a`'s shape.
        NumojoError: If `values` is empty while `mask` selects at least one
            position.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.i32](6)
        var mask = nm.array[nm.boolean]("[True, False, True, False, True, False]")
        nm.indexing.putmask(a, mask, nm.array[nm.i32]("[10, 20]"))
        print(a)
        # [10, 1, 10, 3, 10, 5]
        ```
        .
    """
    if a.shape != mask.shape:
        raise Error(
            NumojoError(
                category="shape",
                message=(
                    "\nError in `putmask`: mask must have the same shape as"
                    " a."
                ),
                location="putmask",
            )
        )

    var mask_c = mask.contiguous()

    if values.size == 0:
        for i in range(a.size):
            if mask_c.unsafe_get(i):
                raise Error(
                    NumojoError(
                        category="value",
                        message=(
                            "\nError in `putmask`: values is empty but mask"
                            " selects at least one position."
                        ),
                        location="putmask",
                    )
                )
        return

    var values_c = values.contiguous()
    for i in range(a.size):
        if mask_c.unsafe_get(i):
            a.itemset(i, values_c.unsafe_get(i % values_c.size))


def choose[
    dtype: DType,
    //,
](a: NDArray[DType.int], choices: List[NDArray[dtype]]) raises -> NDArray[
    dtype
]:
    """Constructs an array from an index array and a list of choice arrays.

    `result[i] = choices[a[i]][i]`, with `a` and every entry of `choices`
    broadcast against each other.

    Parameters:
        dtype: Data type of the choice arrays.

    Args:
        a: Integer index array selecting, for each output position, which
            array in `choices` to draw from.
        choices: List of arrays to choose between.

    Returns:
        Array of the broadcast shape of `a` and every entry in `choices`.

    Raises:
        NumojoError: If `choices` is empty.
        NumojoError: If `a` and `choices` are not mutually broadcast-compatible.
        NumojoError: If any value in `a` is out of bounds for `choices`.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.array[nm.int]("[0, 1, 0]")
        var c0 = nm.array[nm.i32]("[1, 1, 1]")
        var c1 = nm.array[nm.i32]("[9, 9, 9]")
        var options = List[nm.NDArray[nm.i32]]()
        options.append(c0^)
        options.append(c1^)
        print(nm.indexing.choose(a, options))
        # [1, 9, 1]
        ```
        .
    """
    var n = len(choices)
    if n == 0:
        raise Error(
            NumojoError(
                category="value",
                message="\nError in `choose`: choices must not be empty.",
                location="choose",
            )
        )

    var out_shape = a.shape
    for k in range(n):
        try:
            out_shape = out_shape.broadcast(choices[k].shape)
        except e:
            raise Error(
                NumojoError(
                    category="broadcast",
                    message=String(
                        "\nError in `choose`: a and choices are not"
                        " broadcast-compatible: "
                    )
                    + String(e),
                    location="choose",
                )
            )

    var a_bc = broadcast_to(a, out_shape).contiguous()
    var bc_choices = List[NDArray[dtype]](capacity=n)
    for k in range(n):
        bc_choices.append(broadcast_to(choices[k], out_shape).contiguous())

    var result = NDArray[dtype](out_shape)
    for i in range(result.size):
        var sel = Int(a_bc.unsafe_get(i))
        if sel < 0 or sel >= n:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "\nError in `choose`: index {} is out of bounds for"
                        " {} choices."
                    ).format(sel, n),
                    location="choose",
                )
            )
        result.unsafe_set(i, bc_choices[sel].unsafe_get(i))

    return result^
