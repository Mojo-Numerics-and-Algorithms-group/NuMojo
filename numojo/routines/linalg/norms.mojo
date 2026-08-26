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
Determinant and trace computation for 2-D arrays.

Exports
-------
- `det`: Determinant via LUP decomposition.
- `trace`: Sum of the diagonal elements.
"""

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
