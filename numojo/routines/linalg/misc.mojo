# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
# Miscellaneous Linear Algebra Routines
# ===----------------------------------------------------------------------=== #

from sys import simd_width_of
from algorithm import parallelize, vectorize

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

    var m: Int = a.shape[0]
    var n: Int = a.shape[1]

    if offset >= max(m, n):  # Offset beyond the shape of the array
        raise Error(
            "\nError in `diagonal`: Offset beyond the shape of the array"
        )

    var result: NDArray[dtype]

    if offset >= 0:
        var size_of_result = min(n - offset, m)
        result = NDArray[dtype](Shape(size_of_result))
        for i in range(size_of_result):
            result.item(i) = a.item(i, i + offset)
    else:
        var size_of_result = min(m + offset, m)
        result = NDArray[dtype](Shape(size_of_result))
        for i in range(size_of_result):
            result.item(i) = a.item(i - offset, i)

    return result^


fn issymmetric[
    dtype: DType
](
    A: Matrix[dtype], rtol: Scalar[dtype] = 1e-5, atol: Scalar[dtype] = 1e-8
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
