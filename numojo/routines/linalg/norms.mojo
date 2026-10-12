# ===----------------------------------------------------------------------=== #
# NuMojo: Norms
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Norms (numojo.routines.linalg.norms).
=====================================
Determinant, trace, and vector/matrix norm computation.

Exports
-------
- `det`: Determinant via LUP decomposition.
- `trace`: Sum of the diagonal elements.
- `norm`: Vector or matrix norm.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import sqrt
import std.math.math as stdlib_math
from std.utils.numerics import inf, neg_inf

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.linalg.decompositions import (
    lu_decomposition,
    partial_pivoting,
)


def det[dtype: DType](A: NDArray[dtype]) raises -> Scalar[dtype]:
    """
    Find the determinant of A using LUP decomposition.
    """

    if A.ndim != 2:
        raise Error(
            NumojoError(
                category="shape",
                message=String("Array must be 2d."),
                location="det",
            )
        )
    if A.shape[0] != A.shape[1]:
        raise Error(
            NumojoError(
                category="shape",
                message=String("Array is not square."),
                location="det",
            )
        )

    var det_L: Scalar[dtype] = 1
    var det_U: Scalar[dtype] = 1
    var n = A.shape[0]  # Dimension of the matrix

    var A_pivoted: NDArray[dtype]
    var U: NDArray[dtype]
    var L: NDArray[dtype]
    var s: Int
    var A_pivoted_s = partial_pivoting(A.copy())
    A_pivoted = A_pivoted_s[0].copy()
    s = A_pivoted_s[2].copy()

    var L_U: Tuple[NDArray[dtype], NDArray[dtype]] = lu_decomposition[dtype](
        A_pivoted
    )
    L = L_U[0].copy()
    U = L_U[1].copy()

    for i in range(n):
        det_L = det_L * L.item(i, i)
        det_U = det_U * U.item(i, i)

    if s % 2 == 0:
        return det_L * det_U
    else:
        return -det_L * det_U


def trace[
    dtype: DType
](
    array: NDArray[dtype], offset: Int = 0, axis1: Int = 0, axis2: Int = 1
) raises -> NDArray[dtype]:
    """
    Computes the trace of a ndarray.

    Parameters:
        dtype: Data type of the array.

    Args:
        array: A NDArray.
        offset: Offset of the diagonal from the main diagonal.
        axis1: First axis.
        axis2: Second axis.

    Returns:
        The trace of the NDArray.
    """
    if not array.is_c_contiguous():
        return trace(array.contiguous(), offset, axis1, axis2)

    if array.ndim != 2:
        raise Error(
            NumojoError(
                category="shape",
                message="Trace is currently only supported for 2D arrays",
                location="trace",
            )
        )
    if axis1 > array.ndim - 1 or axis2 > array.ndim - 1:
        raise Error(
            NumojoError(
                category="index",
                message="axis cannot be greater than the rank of the array",
                location="trace",
            )
        )
    var result: NDArray[dtype] = NDArray[dtype](Shape(1))
    var rows = array.shape[0]
    var cols = array.shape[1]
    var diag_length = min(rows, cols - offset) if offset >= 0 else min(
        rows + offset, cols
    )

    for i in range(diag_length):
        var row = i if offset >= 0 else i - offset
        var col = i + offset if offset >= 0 else i
        result.unsafe_store[width=1](
            0,
            result.unsafe_load[width=1](0) + array.unsafe_get(row * cols + col),
        )

    return result^


def norm[
    dtype: DType
](x: NDArray[dtype]) raises -> Scalar[dtype] where dtype.is_floating_point():
    """
    Computes the 2-norm (for vectors) or the Frobenius norm (for
    matrices) of `x`, i.e. `sqrt(sum(x**2))` over all elements.

    This matches `numpy.linalg.norm(x)` (no `ord` given).

    Parameters:
        dtype: The element type. Must be a floating-point type.

    Args:
        x: A 1-D or 2-D NDArray.

    Returns:
        The norm of `x`, as a scalar.
    """

    if not x.is_c_contiguous():
        return norm(x.contiguous())

    var acc: Scalar[dtype] = 0
    for i in range(x.size):
        var v = x.unsafe_get(i)
        acc += v * v
    return sqrt(acc)


def _vector_norm[
    dtype: DType
](x: NDArray[dtype], ord: Scalar[dtype]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    if ord == inf[dtype]():
        var result: Scalar[dtype] = 0
        for i in range(x.size):
            var v = abs(x.unsafe_get(i))
            if v > result:
                result = v
        return result

    if ord == neg_inf[dtype]():
        var result = abs(x.unsafe_get(0))
        for i in range(1, x.size):
            var v = abs(x.unsafe_get(i))
            if v < result:
                result = v
        return result

    if ord == 0:
        var result: Scalar[dtype] = 0
        for i in range(x.size):
            if x.unsafe_get(i) != 0:
                result += 1
        return result

    var acc: Scalar[dtype] = 0
    for i in range(x.size):
        acc += stdlib_math.pow(abs(x.unsafe_get(i)), ord)
    return stdlib_math.pow(acc, 1 / ord)


def _matrix_norm[
    dtype: DType
](x: NDArray[dtype], ord: Scalar[dtype]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    var m = x.shape[0]
    var n = x.shape[1]

    if ord == 1 or ord == -1:
        var best: Scalar[dtype] = 0
        for j in range(n):
            var col_sum: Scalar[dtype] = 0
            for i in range(m):
                col_sum += abs(x.item(i, j))
            if j == 0:
                best = col_sum
            elif (ord == 1 and col_sum > best) or (
                ord == -1 and col_sum < best
            ):
                best = col_sum
        return best

    if ord == inf[dtype]() or ord == neg_inf[dtype]():
        var best: Scalar[dtype] = 0
        for i in range(m):
            var row_sum: Scalar[dtype] = 0
            for j in range(n):
                row_sum += abs(x.item(i, j))
            if i == 0:
                best = row_sum
            elif (ord == inf[dtype]() and row_sum > best) or (
                ord == neg_inf[dtype]() and row_sum < best
            ):
                best = row_sum
        return best

    raise Error(
        NumojoError(
            category="value",
            message=String(
                "Unsupported matrix `ord`. Supported values are 1, -1,"
                " inf, -inf, or no `ord` (Frobenius norm). The 2-norm"
                " (largest singular value) requires SVD, which is not"
                " yet supported."
            ),
            location="norm",
        )
    )


def norm[
    dtype: DType
](x: NDArray[dtype], ord: Scalar[dtype]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """
    Computes a vector or matrix norm of `x`, matching a subset of
    `numpy.linalg.norm(x, ord=ord)`.

    For 1-D `x`, `ord` is the order of the generalized p-norm
    `sum(|x|**ord)**(1/ord)`, with `ord = inf` and `ord = -inf` special
    cased to the max and min absolute value, and `ord = 0` counting the
    number of non-zero elements.

    For 2-D `x`, `ord` must be one of `1` (max absolute column sum),
    `-1` (min absolute column sum), `inf` (max absolute row sum), or
    `-inf` (min absolute row sum). The matrix 2-norm (largest singular
    value) is not yet supported, since it requires SVD.

    Parameters:
        dtype: The element type. Must be a floating-point type.

    Args:
        x: A 1-D or 2-D NDArray.
        ord: The order of the norm.

    Returns:
        The norm of `x`, as a scalar.
    """

    if x.ndim == 1:
        return _vector_norm(x, ord)
    elif x.ndim == 2:
        return _matrix_norm(x, ord)
    else:
        raise Error(
            NumojoError(
                category="shape",
                message=String(
                    "`norm` is only supported for 1-D and 2-D arrays."
                ),
                location="norm",
            )
        )
