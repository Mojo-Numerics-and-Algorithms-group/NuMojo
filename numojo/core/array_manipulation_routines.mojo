"""
# ===----------------------------------------------------------------------=== #
# ARRAY MANIPULATION ROUTINES
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #
"""


fn copyto():
    pass


fn shape():
    pass


fn reshape():
    pass


fn ravel():
    pass


fn where[dtype: DType](inout array: NDArray[dtype], mask:NDArray[DType.bool], scalar: SIMD[dtype, 1]) raises:
    """
    Return elements chosen from `x` or `y` depending on `condition`.

    Parameters:
        dtype: DType.
    
    Args:
        array: NDArray[dtype].
        mask: NDArray[DType.bool].
        scalar: SIMD[dtype, 1].

    """

    for i in range(array.ndshape._size):
        if mask.data[i] == True:
            array.data.store(i, scalar)

fn where[dtype: DType](inout array: NDArray[dtype], mask:NDArray[DType.bool], array1: NDArray[dtype]) raises:
    """
    Return elements chosen from `x` or `y` depending on `condition`.

    Parameters:
        dtype: DType.
    
    Args:
        array: NDArray[dtype].
        mask: NDArray[DType.bool].
        array1: NDArray[dtype].

    """
    for i in range(array.ndshape._size):
        if mask.data[i] == True:
            array.data.store(i, array1.data[i])




