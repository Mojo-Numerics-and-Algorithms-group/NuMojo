"""
`numojo.core.mat.stats` module provides statistical functions for Matrix type.

- Order statistics
- Averages and variances
- Correlating
- Histograms

"""

from .mat import Matrix
from .creation import full, zeros
from .math import sum

# ===-----------------------------------------------------------------------===#
# Averages and variances
# ===-----------------------------------------------------------------------===#


fn mean[dtype: DType](A: Matrix[dtype]) -> Scalar[dtype]:
    """
    Calculate the arithmetic average of all items in the Matrix.

    Args:
        A: Matrix.
    """

    return sum(A) / A.size


fn mean[dtype: DType](A: Matrix[dtype], axis: Int) raises -> Matrix[dtype]:
    """
    Calculate the arithmetic average of a Matrix along the axis.

    Args:
        A: Matrix.
        axis: 0 or 1.
    """

    if axis == 0:
        return sum(A, axis=0) / A.shape[0]
    elif axis == 1:
        return sum(A, axis=1) / A.shape[1]
    else:
        raise Error(String("The axis can either be 1 or 0!"))
