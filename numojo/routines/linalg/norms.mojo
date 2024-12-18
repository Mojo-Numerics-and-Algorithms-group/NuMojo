# ===----------------------------------------------------------------------=== #
# Norms and other numbers
# ===----------------------------------------------------------------------=== #


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
        result._buf.store(0, result._buf.load(0) + array._buf[row * cols + col])

    return result
