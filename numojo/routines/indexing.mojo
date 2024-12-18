"""
Indexing routines.
"""

from sys import simdwidthof
from algorithm import vectorize
from numojo.core.ndarray import NDArray
from numojo.core.ndstrides import NDArrayStrides

# ===----------------------------------------------------------------------=== #
# Generating index arrays
# ===----------------------------------------------------------------------=== #


fn where[
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
    for i in range(x.size):
        if mask._buf[i] == True:
            x._buf.store(i, scalar)


# TODO: do it with vectorization
fn where[
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
    for i in range(x.size):
        if mask._buf[i] == True:
            x._buf.store(i, y._buf[i])
