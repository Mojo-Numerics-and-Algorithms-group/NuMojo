# ===----------------------------------------------------------------------=== #
# Norms and other numbers
# ===----------------------------------------------------------------------=== #

from numojo.core.ndarray import NDArray
from numojo.routines.linalg.decompositions import partial_pivoting


fn det[dtype: DType](A: NDArray[dtype]) raises -> Scalar[dtype]:
    """
    Find the determinant of A using LUP decomposition.
    """

    if A.ndim != 2:
        raise Error(String("Array must be 2d."))
    if A.shape[0] != A.shape[1]:
        raise Error(String("Matrix is not square."))

    var det_L: Scalar[dtype] = 1
    var det_U: Scalar[dtype] = 1
    var n = A.shape[0]  # Dimension of the matrix

    var A_pivoted: NDArray[dtype]
    var U: NDArray[dtype]
    var L: NDArray[dtype]
    var s: Int
    A_pivoted, _, s = partial_pivoting(A)
    L, U = lu_decomposition[dtype](A_pivoted)

    for i in range(n):
        det_L = det_L * L.item(i, i)
        det_U = det_U * U.item(i, i)

    if s % 2 == 0:
        return det_L * det_U
    else:
        return -det_L * det_U


# TODO: implement for arbitrary axis
fn trace[
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
    if array.ndim != 2:
        raise Error("Trace is currently only supported for 2D arrays")
    if axis1 > array.ndim - 1 or axis2 > array.ndim - 1:
        raise Error("axis cannot be greater than the rank of the array")
    var result: NDArray[dtype] = NDArray[dtype](Shape(1))
    var rows = array.shape[0]
    var cols = array.shape[1]
    var diag_length = min(rows, cols - offset) if offset >= 0 else min(
        rows + offset, cols
    )

    for i in range(diag_length):
        var row = i if offset >= 0 else i - offset
        var col = i + offset if offset >= 0 else i
        result._buf.ptr.store(
            0, result._buf.ptr.load(0) + array._buf.ptr[row * cols + col]
        )

    return result
