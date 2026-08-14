# ===----------------------------------------------------------------------=== #
# NuMojo: Misc
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Miscellaneous Linear Algebra Routines (numojo.routines.linalg.misc)
-------------------------------------------------------------------
This module provides miscellaneous linear algebra routines, such as extracting diagonals and checking for symmetry.
"""
# ===----------------------------------------------------------------------===#
# numojo
# ===----------------------------------------------------------------------===#
from numojo.core.layout import NDArrayShape
from numojo.core.matrix import Matrix
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape


def diagonal[
    dtype: DType
](
    a: NDArray[dtype], offset: Int = 0, axis1: Int = 0, axis2: Int = 1
) raises -> NDArray[dtype]:
    """
    Returns specific diagonals.

    For 2-D arrays (the default `axis1=0, axis2=1` case), returns the 1-D
    diagonal at the given `offset`. For N-D arrays, `axis1` and `axis2` are
    treated as the two axes that define the 2-D sub-arrays whose diagonals
    are extracted; the result has the two diagonalized axes removed and
    replaced by a new last axis holding the diagonal values. The result shape is
    `a.shape[axes not in {axis1, axis2}] + (diagonal_length,)`, where the
    surviving axes keep their original relative order.

    Raises:
        Error: If the array has fewer than 2 dimensions.
        Error: If `axis1` or `axis2` is out of bounds, or `axis1 == axis2`.
        Error: If the offset is beyond the shape of the array.

    Parameters:
        dtype: Data type of the array.

    Args:
        a: An NDArray.
        offset: Offset of the diagonal from the main diagonal.
        axis1: First axis of the 2-D sub-arrays from which the diagonals
            should be taken. Defaults to 0.
        axis2: Second axis of the 2-D sub-arrays from which the diagonals
            should be taken. Defaults to 1.

    Returns:
        The diagonal(s) of the NDArray.

    Examples:
        ```mojo
        import numojo as nm

        var a = nm.arange[nm.i32](60).reshape(nm.Shape(3, 4, 5))
        # axis1=0, axis2=1 (default): result shape (5, 3) -- min(3,4)=3
        print(nm.linalg.diagonal(a, axis1=0, axis2=1))
        ```
        .
    """
    if a.ndim < 2:
        raise Error(
            "\nError in `diagonal`: Array must have at least 2 dimensions, got "
            + String(a.ndim)
        )

    var norm_axis1 = axis1
    if norm_axis1 < 0:
        norm_axis1 = a.ndim + norm_axis1
    var norm_axis2 = axis2
    if norm_axis2 < 0:
        norm_axis2 = a.ndim + norm_axis2

    if (
        norm_axis1 < 0
        or norm_axis1 >= a.ndim
        or norm_axis2 < 0
        or norm_axis2 >= a.ndim
    ):
        raise Error(
            String(
                "\nError in `diagonal`: axis1 {} and axis2 {} must be valid"
                " axes for an array with {} dimensions."
            ).format(axis1, axis2, a.ndim)
        )
    if norm_axis1 == norm_axis2:
        raise Error("\nError in `diagonal`: axis1 and axis2 must be different.")

    if not a.is_c_contiguous():
        return diagonal(a.contiguous(), offset, axis1, axis2)

    var m: Int = a.shape[norm_axis1]
    var n: Int = a.shape[norm_axis2]

    if offset > n - 1 or offset < -(m - 1):
        raise Error(
            "\nError in `diagonal`: Offset "
            + String(offset)
            + " is outside the valid range for axes with shape ("
            + String(m)
            + ", "
            + String(n)
            + ")"
        )

    var diag_len: Int
    var start_row: Int
    var start_col: Int
    if offset >= 0:
        diag_len = min(m, n - offset)
        start_row = 0
        start_col = offset
    else:
        diag_len = min(m + offset, n)
        start_row = -offset
        start_col = 0

    # Fast path: simple 2-D case with default axes.
    if a.ndim == 2 and norm_axis1 == 0 and norm_axis2 == 1:
        var result2d = NDArray[dtype](Shape(diag_len))
        for i in range(diag_len):
            result2d.unsafe_set(
                i, a.unsafe_get((i + start_row) * n + (i + start_col))
            )
        return result2d^

    # General N-D case: surviving axes (in original order) come first,
    # the diagonal axis (length `diag_len`) is appended last.
    var surviving_axes = List[Int]()
    for d in range(a.ndim):
        if d != norm_axis1 and d != norm_axis2:
            surviving_axes.append(d)

    var out_shape_list = List[Int]()
    for d in surviving_axes:
        out_shape_list.append(a.shape[d])
    out_shape_list.append(diag_len)

    var result = NDArray[dtype](NDArrayShape(out_shape_list))

    var surviving_size = 1
    for d in surviving_axes:
        surviving_size *= a.shape[d]

    # strides of `a` (C-contiguous, since we ensured this above)
    var a_strides = List[Int]()
    for _ in range(a.ndim):
        a_strides.append(0)
    var stride_acc = 1
    for d in range(a.ndim - 1, -1, -1):
        a_strides[d] = stride_acc
        stride_acc *= a.shape[d]

    for outer in range(surviving_size):
        # Decode `outer` into coordinates along `surviving_axes`.
        var rem = outer
        var base_offset = 0
        for k in range(len(surviving_axes) - 1, -1, -1):
            var axis_d = surviving_axes[k]
            var dim_size = a.shape[axis_d]
            var coord = rem % dim_size
            rem //= dim_size
            base_offset += coord * a_strides[axis_d]

        for i in range(diag_len):
            var src_offset = (
                base_offset
                + (i + start_row) * a_strides[norm_axis1]
                + (i + start_col) * a_strides[norm_axis2]
            )
            result.unsafe_set(outer * diag_len + i, a.unsafe_get(src_offset))

    return result^


def issymmetric[
    dtype: DType
](
    A: Matrix[dtype],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
) -> Bool:
    """
    Returns True if A is symmetric, False otherwise.

    Parameters:
        dtype: Data type of the Matrix Elements.

    Args:
        A: A Matrix.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if the array is symmetric, False otherwise.
    """

    if A.shape[0] != A.shape[1]:
        return False

    var n = A.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            var a_ij = A._load(i, j)
            var a_ji = A._load(j, i)
            var diff = abs(a_ij - a_ji)
            var allowed_error = atol + rtol * max(abs(a_ij), abs(a_ji))
            if diff > allowed_error:
                return False

    return True
