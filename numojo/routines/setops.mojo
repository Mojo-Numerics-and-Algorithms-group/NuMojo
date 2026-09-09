# ===----------------------------------------------------------------------=== #
# NuMojo: Set routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Set routines (numojo.routines.setops).
======================================
Operations on arrays treated as sets of elements.

Exports
-------
- `unique`, `unique_counts`, `unique_index`, `unique_inverse`, `unique_all`:
  Sorted unique elements of an array (optionally along an axis), together
  with counts, first-occurrence indices, and/or reconstruction indices.

Notes
-----
- Each function has an axis-less overload (flattens the array first) and
  an `axis` overload (treats slices along that axis as elements). NumPy's
  own `unique_all`/`unique_counts`/`unique_inverse`/`unique_values`
  (the Array API-compatible entry points) do not accept `axis`; the axis
  overloads here match the capability of NumPy's classic
  `numpy.unique(ar, axis=..., return_index=..., return_inverse=...,
  return_counts=...)` instead.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.layout import NDArrayShape
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.manipulation import moveaxis, ravel
from numojo.routines.sorting import argsort

# ===----------------------------------------------------------------------=== #
# Shared helpers
# ===----------------------------------------------------------------------=== #


def _unique_core[
    dtype: DType
](flat: NDArray[dtype]) raises -> Tuple[
    List[Scalar[dtype]], List[Int], List[Int], List[Int]
]:
    """
    Internal: core of the flattened-array `unique*` family.

    Returns `(values, first_index, inverse, counts)`:

    - `values`: the sorted unique values of `flat`.
    - `first_index`: for each entry of `values`, the index (into `flat`) of
      its first occurrence.
    - `inverse`: for each entry of `flat`, the index into `values` that
      reconstructs it (`values[inverse[i]] == flat[i]`).
    - `counts`: for each entry of `values`, the number of occurrences in
      `flat`.
    """
    var n = flat.size
    var order = argsort(flat)

    var values = List[Scalar[dtype]]()
    var first_index = List[Int]()
    var counts = List[Int]()
    var group_of = List[Int]()  # group id of order[k], in sorted order.

    for k in range(n):
        var idx = Int(order.unsafe_get(k))
        var val = flat.unsafe_get(idx)
        if (k == 0) or (val != values[len(values) - 1]):
            values.append(val)
            first_index.append(idx)
            counts.append(1)
        else:
            counts[len(counts) - 1] += 1
            if idx < first_index[len(first_index) - 1]:
                first_index[len(first_index) - 1] = idx
        group_of.append(len(values) - 1)

    var inverse = List[Int]()
    for _ in range(n):
        inverse.append(0)
    for k in range(n):
        var idx = Int(order.unsafe_get(k))
        inverse[idx] = group_of[k]

    return (values^, first_index^, inverse^, counts^)


def _row_compare[
    dtype: DType
](B: NDArray[dtype], i: Int, j: Int, row_size: Int) -> Int:
    """Internal: lexicographically compares row `i` and row `j` of a
    C-contiguous array `B` of row length `row_size`. Returns -1, 0, or 1.
    """
    for k in range(row_size):
        var a = B.unsafe_get(i * row_size + k)
        var b = B.unsafe_get(j * row_size + k)
        if a < b:
            return -1
        if a > b:
            return 1
    return 0


def _sorted_row_order[
    dtype: DType
](B: NDArray[dtype], n: Int, row_size: Int) -> List[Int]:
    """Internal: returns row indices `[0, n)` of a C-contiguous array `B`
    sorted lexicographically (stable insertion sort)."""
    var order = List[Int]()
    for i in range(n):
        order.append(i)
    for i in range(1, n):
        var key = order[i]
        var j = i - 1
        while (j >= 0) and (_row_compare(B, order[j], key, row_size) > 0):
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key
    return order^


def _unique_rows_core[
    dtype: DType
](B: NDArray[dtype], n: Int, row_size: Int) -> Tuple[
    List[Int], List[Int], List[Int], List[Int]
]:
    """
    Internal: core of the axis-based `unique*` family, for a C-contiguous
    array `B` whose axis of interest has been moved to position 0 (`n`
    rows of `row_size` elements each).

    Returns `(unique_rows, first_row_index, inverse, counts)`:

    - `unique_rows`: for each unique group, the index (into `B`, i.e. the
      original axis) of a representative row.
    - `first_row_index`: for each unique group, the index of its first
      occurrence along the original axis.
    - `inverse`: for each of the `n` original rows, the group index that
      reconstructs it (`unique_rows[inverse[i]]` is equal to row `i`).
    - `counts`: for each unique group, the number of rows in it.
    """
    var order = _sorted_row_order(B, n, row_size)
    var unique_rows = List[Int]()
    var first_row_index = List[Int]()
    var counts = List[Int]()
    var inverse = List[Int]()
    for _ in range(n):
        inverse.append(0)

    for k in range(n):
        var row = order[k]
        if (k == 0) or (_row_compare(B, order[k - 1], row, row_size) != 0):
            unique_rows.append(row)
            first_row_index.append(row)
            counts.append(1)
        else:
            counts[len(counts) - 1] += 1
            if row < first_row_index[len(first_row_index) - 1]:
                first_row_index[len(first_row_index) - 1] = row
        inverse[row] = len(unique_rows) - 1

    return (unique_rows^, first_row_index^, inverse^, counts^)


def _normalize_axis(axis: Int, ndim: Int, location: String) raises -> Int:
    """Internal: normalizes a (possibly negative) axis and checks bounds."""
    var ax = axis
    if ax < 0:
        ax += ndim
    if (ax < 0) or (ax >= ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Axis out of range: got {}, expected {} <= axis < {}."
                ).format(axis, -ndim, ndim),
                location=location,
            )
        )
    return ax


def _gather_rows[
    dtype: DType
](B: NDArray[dtype], rows: List[Int], row_size: Int) raises -> NDArray[dtype]:
    """Internal: builds a new array with shape `(len(rows), *B.shape[1:])`
    by gathering the given rows of a C-contiguous array `B`."""
    var m = len(rows)
    var out_shape_list = List[Int]()
    out_shape_list.append(m)
    for d in range(1, B.ndim):
        out_shape_list.append(B.shape[d])
    var out = NDArray[dtype](NDArrayShape(out_shape_list))
    for i in range(m):
        var src_row = rows[i]
        for j in range(row_size):
            out.unsafe_set(
                i * row_size + j, B.unsafe_get(src_row * row_size + j)
            )
    return out^


# ===----------------------------------------------------------------------=== #
# unique
# ===----------------------------------------------------------------------=== #


def unique[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Returns the sorted unique elements of an array.

    The input array is flattened before the unique elements are computed.

    Parameters:
        dtype: DType.

    Args:
        A: A NDArray.

    Returns:
        A 1-d array of the sorted unique values of `A`.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.fromstring[nm.i32]("[3, 1, 2, 1, 3, 3]")
        print(nm.unique(a))  # [1, 2, 3]
        ```
    """
    if A.size == 0:
        return NDArray[dtype](Shape(0))

    var core = _unique_core(ravel(A, order="C"))
    var values = core[0].copy()
    var result = NDArray[dtype](Shape(len(values)))
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


def unique[dtype: DType](A: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    (overload) Returns the sorted unique slices of an array along `axis`.

    Each slice obtained by indexing `A` along `axis` is treated as a
    single element; two slices are equal if all their values match.

    Parameters:
        dtype: DType.

    Args:
        A: A NDArray.
        axis: The axis whose slices are compared. Supports negative
            indices.

    Returns:
        An array with the same shape as `A` except along `axis`, which is
        reduced to the number of unique slices, lexicographically sorted.

    Raises:
        NumojoError: If `axis` is out of bound.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.fromstring[nm.i32]("[[1,2][3,4][1,2]]")
        print(nm.unique(a, axis=0))  # [[1, 2], [3, 4]]
        ```
    """
    var ax = _normalize_axis(axis, A.ndim, "unique")
    var B = moveaxis(A, ax, 0).contiguous()
    var n = B.shape[0]
    if n == 0:
        return moveaxis(B, 0, ax)
    var row_size = B.size // n

    var core = _unique_rows_core(B, n, row_size)
    var unique_rows = core[0].copy()

    return moveaxis(_gather_rows(B, unique_rows, row_size), 0, ax)


# ===----------------------------------------------------------------------=== #
# unique_counts
# ===----------------------------------------------------------------------=== #


def unique_counts[
    dtype: DType
](A: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[DType.int]]:
    """
    Returns the sorted unique elements of an array together with the
    number of times each unique value occurs.

    The input array is flattened before the unique elements are computed.

    Parameters:
        dtype: DType.

    Args:
        A: A NDArray.

    Returns:
        A tuple `(values, counts)`: `values` is a 1-d array of the sorted
        unique values of `A`, and `counts` is a 1-d array of the same
        length giving the number of occurrences of each value.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.fromstring[nm.i32]("[3, 1, 2, 1, 3, 3]")
        var result = nm.unique_counts(a)
        print(result[0])  # [1, 2, 3]
        print(result[1])  # [2, 1, 3]
        ```
    """
    if A.size == 0:
        return (NDArray[dtype](Shape(0)), NDArray[DType.int](Shape(0)))

    var core = _unique_core(ravel(A, order="C"))
    var values = core[0].copy()
    var counts = core[3].copy()
    var n = len(values)
    var out_values = NDArray[dtype](Shape(n))
    var out_counts = NDArray[DType.int](Shape(n))
    for i in range(n):
        out_values.unsafe_set(i, values[i])
        out_counts.unsafe_set(i, counts[i])
    return (out_values^, out_counts^)


def unique_counts[
    dtype: DType
](A: NDArray[dtype], axis: Int) raises -> Tuple[
    NDArray[dtype], NDArray[DType.int]
]:
    """
    (overload) Returns the sorted unique slices of an array along `axis`
    together with the number of times each one occurs.

    Parameters:
        dtype: DType.

    Args:
        A: A NDArray.
        axis: The axis whose slices are compared. Supports negative
            indices.

    Returns:
        A tuple `(values, counts)`: `values` has the same shape as `A`
        except along `axis`, which is reduced to the number of unique
        slices, lexicographically sorted; `counts` is a 1-d array giving
        the number of occurrences of each slice.

    Raises:
        NumojoError: If `axis` is out of bound.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.fromstring[nm.i32]("[[1,2][3,4][1,2]]")
        var result = nm.unique_counts(a, axis=0)
        print(result[0])  # [[1, 2], [3, 4]]
        print(result[1])  # [2, 1]
        ```
    """
    var ax = _normalize_axis(axis, A.ndim, "unique_counts")
    var B = moveaxis(A, ax, 0).contiguous()
    var n = B.shape[0]
    if n == 0:
        return (moveaxis(B, 0, ax), NDArray[DType.int](Shape(0)))
    var row_size = B.size // n

    var core = _unique_rows_core(B, n, row_size)
    var unique_rows = core[0].copy()
    var counts = core[3].copy()

    var m = len(unique_rows)
    var out_counts = NDArray[DType.int](Shape(m))
    for i in range(m):
        out_counts.unsafe_set(i, counts[i])

    return (
        moveaxis(_gather_rows(B, unique_rows, row_size), 0, ax),
        out_counts^,
    )


# ===----------------------------------------------------------------------=== #
# unique_index, unique_inverse, unique_all
# ===----------------------------------------------------------------------=== #


def unique_index[
    dtype: DType
](A: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[DType.int]]:
    """
    Returns the sorted unique elements of a flattened array together with
    the index of the first occurrence of each one.

    Parameters:
        dtype: DType.

    Args:
        A: A NDArray.

    Returns:
        A tuple `(values, index)`: `values` is a 1-d array of the sorted
        unique values of the flattened `A`, and `index` gives, for each
        entry of `values`, the index of its first occurrence in the
        flattened `A`.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.fromstring[nm.i32]("[3, 1, 2, 1, 3, 3]")
        var result = nm.unique_index(a)
        print(result[0])  # [1, 2, 3]
        print(result[1])  # [1, 2, 0]
        ```
    """
    if A.size == 0:
        return (NDArray[dtype](Shape(0)), NDArray[DType.int](Shape(0)))

    var core = _unique_core(ravel(A, order="C"))
    var values = core[0].copy()
    var first_index = core[1].copy()
    var n = len(values)
    var out_values = NDArray[dtype](Shape(n))
    var out_index = NDArray[DType.int](Shape(n))
    for i in range(n):
        out_values.unsafe_set(i, values[i])
        out_index.unsafe_set(i, first_index[i])
    return (out_values^, out_index^)


def unique_index[
    dtype: DType
](A: NDArray[dtype], axis: Int) raises -> Tuple[
    NDArray[dtype], NDArray[DType.int]
]:
    """
    (overload) Returns the sorted unique slices of an array along `axis`
    together with the index of the first occurrence of each one.

    Parameters:
        dtype: DType.

    Args:
        A: A NDArray.
        axis: The axis whose slices are compared. Supports negative
            indices.

    Returns:
        A tuple `(values, index)`: `values` has the same shape as `A`
        except along `axis`, which is reduced to the number of unique
        slices, lexicographically sorted; `index` gives, for each entry of
        `values`, the position along `axis` of its first occurrence.

    Raises:
        NumojoError: If `axis` is out of bound.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.fromstring[nm.i32]("[[1,2][3,4][1,2]]")
        var result = nm.unique_index(a, axis=0)
        print(result[0])  # [[1, 2], [3, 4]]
        print(result[1])  # [0, 1]
        ```
    """
    var ax = _normalize_axis(axis, A.ndim, "unique_index")
    var B = moveaxis(A, ax, 0).contiguous()
    var n = B.shape[0]
    if n == 0:
        return (moveaxis(B, 0, ax), NDArray[DType.int](Shape(0)))
    var row_size = B.size // n

    var core = _unique_rows_core(B, n, row_size)
    var unique_rows = core[0].copy()
    var first_row_index = core[1].copy()

    var m = len(unique_rows)
    var out_index = NDArray[DType.int](Shape(m))
    for i in range(m):
        out_index.unsafe_set(i, first_row_index[i])

    return (moveaxis(_gather_rows(B, unique_rows, row_size), 0, ax), out_index^)


def unique_inverse[
    dtype: DType
](A: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[DType.int]]:
    """
    Returns the sorted unique elements of a flattened array together with
    the indices that reconstruct it from those unique elements.

    Parameters:
        dtype: DType.

    Args:
        A: A NDArray.

    Returns:
        A tuple `(values, inverse)`: `values` is a 1-d array of the sorted
        unique values of the flattened `A`, and `inverse` has the same
        length as the flattened `A`, such that
        `values[inverse[i]] == flatten(A)[i]` for every `i`.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.fromstring[nm.i32]("[3, 1, 2, 1, 3, 3]")
        var result = nm.unique_inverse(a)
        print(result[0])  # [1, 2, 3]
        print(result[1])  # [2, 0, 1, 0, 2, 2]
        ```
    """
    if A.size == 0:
        return (NDArray[dtype](Shape(0)), NDArray[DType.int](Shape(0)))

    var core = _unique_core(ravel(A, order="C"))
    var values = core[0].copy()
    var inverse = core[2].copy()
    var n_values = len(values)
    var out_values = NDArray[dtype](Shape(n_values))
    for i in range(n_values):
        out_values.unsafe_set(i, values[i])
    var out_inverse = NDArray[DType.int](Shape(len(inverse)))
    for i in range(len(inverse)):
        out_inverse.unsafe_set(i, inverse[i])
    return (out_values^, out_inverse^)


def unique_inverse[
    dtype: DType
](A: NDArray[dtype], axis: Int) raises -> Tuple[
    NDArray[dtype], NDArray[DType.int]
]:
    """
    (overload) Returns the sorted unique slices of an array along `axis`
    together with the indices that reconstruct it from those slices.

    Parameters:
        dtype: DType.

    Args:
        A: A NDArray.
        axis: The axis whose slices are compared. Supports negative
            indices.

    Returns:
        A tuple `(values, inverse)`: `values` has the same shape as `A`
        except along `axis`, which is reduced to the number of unique
        slices, lexicographically sorted; `inverse` has length
        `A.shape[axis]`, such that gathering `values` along axis 0 at
        `inverse` reconstructs `A` along `axis`.

    Raises:
        NumojoError: If `axis` is out of bound.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.fromstring[nm.i32]("[[1,2][3,4][1,2]]")
        var result = nm.unique_inverse(a, axis=0)
        print(result[0])  # [[1, 2], [3, 4]]
        print(result[1])  # [0, 1, 0]
        ```
    """
    var ax = _normalize_axis(axis, A.ndim, "unique_inverse")
    var B = moveaxis(A, ax, 0).contiguous()
    var n = B.shape[0]
    if n == 0:
        return (moveaxis(B, 0, ax), NDArray[DType.int](Shape(0)))
    var row_size = B.size // n

    var core = _unique_rows_core(B, n, row_size)
    var unique_rows = core[0].copy()
    var inverse = core[2].copy()

    var out_inverse = NDArray[DType.int](Shape(n))
    for i in range(n):
        out_inverse.unsafe_set(i, inverse[i])

    return (
        moveaxis(_gather_rows(B, unique_rows, row_size), 0, ax),
        out_inverse^,
    )


def unique_all[
    dtype: DType
](A: NDArray[dtype]) raises -> Tuple[
    NDArray[dtype], NDArray[DType.int], NDArray[DType.int], NDArray[DType.int]
]:
    """
    Returns the sorted unique elements of a flattened array together with
    first-occurrence indices, inverse indices, and counts, all in one
    call.

    Parameters:
        dtype: DType.

    Args:
        A: A NDArray.

    Returns:
        A tuple `(values, index, inverse, counts)`. See `unique_index`,
        `unique_inverse`, and `unique_counts` for the meaning of each
        component.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.fromstring[nm.i32]("[3, 1, 2, 1, 3, 3]")
        var result = nm.unique_all(a)
        print(result[0])  # values:  [1, 2, 3]
        print(result[1])  # index:   [1, 2, 0]
        print(result[2])  # inverse: [2, 0, 1, 0, 2, 2]
        print(result[3])  # counts:  [2, 1, 3]
        ```
    """
    if A.size == 0:
        return (
            NDArray[dtype](Shape(0)),
            NDArray[DType.int](Shape(0)),
            NDArray[DType.int](Shape(0)),
            NDArray[DType.int](Shape(0)),
        )

    var core = _unique_core(ravel(A, order="C"))
    var values = core[0].copy()
    var first_index = core[1].copy()
    var inverse = core[2].copy()
    var counts = core[3].copy()

    var n = len(values)
    var out_values = NDArray[dtype](Shape(n))
    var out_index = NDArray[DType.int](Shape(n))
    var out_counts = NDArray[DType.int](Shape(n))
    for i in range(n):
        out_values.unsafe_set(i, values[i])
        out_index.unsafe_set(i, first_index[i])
        out_counts.unsafe_set(i, counts[i])

    var out_inverse = NDArray[DType.int](Shape(len(inverse)))
    for i in range(len(inverse)):
        out_inverse.unsafe_set(i, inverse[i])

    return (out_values^, out_index^, out_inverse^, out_counts^)


def unique_all[
    dtype: DType
](A: NDArray[dtype], axis: Int) raises -> Tuple[
    NDArray[dtype], NDArray[DType.int], NDArray[DType.int], NDArray[DType.int]
]:
    """
    (overload) Returns the sorted unique slices of an array along `axis`
    together with first-occurrence indices, inverse indices, and counts,
    all in one call.

    Parameters:
        dtype: DType.

    Args:
        A: A NDArray.
        axis: The axis whose slices are compared. Supports negative
            indices.

    Returns:
        A tuple `(values, index, inverse, counts)`. See `unique_index`,
        `unique_inverse`, and `unique_counts` (axis overloads) for the
        meaning of each component.

    Raises:
        NumojoError: If `axis` is out of bound.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.fromstring[nm.i32]("[[1,2][3,4][1,2]]")
        var result = nm.unique_all(a, axis=0)
        print(result[0])  # values:  [[1, 2], [3, 4]]
        print(result[1])  # index:   [0, 1]
        print(result[2])  # inverse: [0, 1, 0]
        print(result[3])  # counts:  [2, 1]
        ```
    """
    var ax = _normalize_axis(axis, A.ndim, "unique_all")
    var B = moveaxis(A, ax, 0).contiguous()
    var n = B.shape[0]
    if n == 0:
        return (
            moveaxis(B, 0, ax),
            NDArray[DType.int](Shape(0)),
            NDArray[DType.int](Shape(0)),
            NDArray[DType.int](Shape(0)),
        )
    var row_size = B.size // n

    var core = _unique_rows_core(B, n, row_size)
    var unique_rows = core[0].copy()
    var first_row_index = core[1].copy()
    var inverse = core[2].copy()
    var counts = core[3].copy()

    var m = len(unique_rows)
    var out_index = NDArray[DType.int](Shape(m))
    var out_counts = NDArray[DType.int](Shape(m))
    for i in range(m):
        out_index.unsafe_set(i, first_row_index[i])
        out_counts.unsafe_set(i, counts[i])

    var out_inverse = NDArray[DType.int](Shape(n))
    for i in range(n):
        out_inverse.unsafe_set(i, inverse[i])

    return (
        moveaxis(_gather_rows(B, unique_rows, row_size), 0, ax),
        out_index^,
        out_inverse^,
        out_counts^,
    )
