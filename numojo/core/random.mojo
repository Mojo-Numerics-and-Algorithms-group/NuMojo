"""
Random values array generation.
"""
# ===----------------------------------------------------------------------=== #
# Implements RANDOM 
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #


import random
from .ndarray import NDArray


fn rand[dtype: DType](*shape: Int) raises -> NDArray[dtype]:
    """
    Generate a random NDArray of the given shape and dtype.

    Example:
        ```py
        var arr = numojo.core.random.rand[numojo.i16](3,2,4)
        print(arr)
        ```

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        shape: The shape of the NDArray.

    Returns:
        The generated NDArray of type `dtype` filled with random values.
    """
    return NDArray[dtype](shape, random=True)
