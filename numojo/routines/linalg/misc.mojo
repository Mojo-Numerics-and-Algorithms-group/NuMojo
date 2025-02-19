# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
# Miscellaneous Linear Algebra Routines
# ===----------------------------------------------------------------------=== #

from numojo.core.ndarray import NDArray


fn diagonal[
    dtype: DType
](a: NDArray[dtype], offset: Int = 0) raises -> NDArray[dtype]:
    """
    Returns specific diagonals.
    Currently supports only 2D arrays.

    Raises:
        Error: If the array is not 2D.
        Error: If the offset is beyond the shape of the array.

    Parameters:
        dtype: Data type of the array.

    Args:
        a: An NDArray.
        offset: Offset of the diagonal from the main diagonal.

    Returns:
        The diagonal of the NDArray.
    """

    if a.ndim != 2:
        raise Error("\nError in `diagonal`: Only supports 2D arrays")

    var m = a.shape[0]
    var n = a.shape[1]

    if offset >= max(m, n):  # Offset beyond the shape of the array
        raise Error(
            "\nError in `diagonal`: Offset beyond the shape of the array"
        )

    var res: NDArray[dtype]

    if offset >= 0:
        var size_of_res = min(n - offset, m)
        res = NDArray[dtype](Shape(size_of_res))
        for i in range(size_of_res):
            res.item(i) = a.item(i, i + offset)
    else:
        var size_of_res = min(m + offset, m)
        res = NDArray[dtype](Shape(size_of_res))
        for i in range(size_of_res):
            res.item(i) = a.item(i - offset, i)

    return res
