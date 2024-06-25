"""
# ===----------------------------------------------------------------------=== #
# Implements RANDOM SAMPLING
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #
"""

import random
from .ndarray import NDArray


fn rand[dtype: DType](*shape: Int) raises -> NDArray[dtype]:
    """
    Example:
        numojo.core.random.rand[numojo.i8](3,2,4)
        Returns an random array with shape 3 x 2 x 4.
    """
    return NDArray[dtype](shape, random=True)
