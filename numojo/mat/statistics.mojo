"""
`numojo.core.mat.statistics` module provides statistical functions for Matrix type.

- Order statistics
- Averages and variances
- Correlating
- Histograms

"""

from .matrix import Matrix
from .creation import full, zeros
from .mathematics import sum

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


fn variance[
    dtype: DType
](A: Matrix[dtype], ddof: Int = 0) raises -> Scalar[dtype]:
    """
    Compute the variance.

    Args:
        A: Matrix.
        ddof: Delta degree of freedom.
    """

    if ddof >= A.size:
        raise Error(String("ddof {ddof} should be smaller than size {A.size}"))

    return sum((A - mean(A)) * (A - mean(A))) / (A.size - ddof)


fn variance[
    dtype: DType
](A: Matrix[dtype], axis: Int, ddof: Int = 0) raises -> Matrix[dtype]:
    """
    Compute the variance along axis.

    Args:
        A: Matrix.
        axis: 0 or 1.
        ddof: Delta degree of freedom.
    """

    if (ddof >= A.shape[0]) or (ddof >= A.shape[1]):
        raise Error(
            String(
                "ddof {ddof} should be smaller than size"
                " {A.shape[0]}x{A.shape[1]}"
            )
        )

    if axis == 0:
        return sum((A - mean(A, axis=0)) * (A - mean(A, axis=0)), axis=0) / (
            A.shape[0] - ddof
        )
    elif axis == 1:
        return sum((A - mean(A, axis=1)) * (A - mean(A, axis=1)), axis=1) / (
            A.shape[1] - ddof
        )
    else:
        raise Error(String("The axis can either be 1 or 0!"))


fn std[dtype: DType](A: Matrix[dtype], ddof: Int = 0) raises -> Scalar[dtype]:
    """
    Compute the standard deviation.

    Args:
        A: Matrix.
        ddof: Delta degree of freedom.
    """

    if ddof >= A.size:
        raise Error(String("ddof {ddof} should be smaller than size {A.size}"))

    return variance(A, ddof=ddof) ** 0.5


fn std[
    dtype: DType
](A: Matrix[dtype], axis: Int, ddof: Int = 0) raises -> Matrix[dtype]:
    """
    Compute the standard deviation along axis.

    Args:
        A: Matrix.
        axis: 0 or 1.
        ddof: Delta degree of freedom.
    """

    return variance(A, axis, ddof=ddof) ** 0.5
