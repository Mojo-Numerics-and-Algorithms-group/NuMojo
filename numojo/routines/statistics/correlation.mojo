# ===----------------------------------------------------------------------=== #
# NuMojo: Covariance and correlation
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Correlation (numojo.routines.statistics.correlation).
=====================================================
Covariance and correlation-coefficient matrices for 2-D arrays.

Exports
-------
- `cov`: Covariance matrix of a set of variables.
- `corrcoef`: Pearson correlation-coefficient matrix of a set of variables.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from std.math import sqrt

from numojo.core.error import NumojoError
from numojo.core.indexing.item import Item
from numojo.core.ndarray import NDArray
from numojo.routines.linalg.products import matmul
from numojo.routines.manipulation import _broadcast_back_to, transpose
from numojo.routines.statistics.averages import mean


def cov[
    dtype: DType, //, returned_dtype: DType = DType.float64
](m: NDArray[dtype], rowvar: Bool = True, ddof: Int = 1) raises -> NDArray[
    returned_dtype
]:
    """
    Compute the covariance matrix of a set of variables.

    Matches `numpy.cov` for a 2-D input: each row (or, if `rowvar` is
    `False`, each column) is treated as a variable, with observations
    in the other dimension.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        m: A 2-D array of variables and observations.
        rowvar: If `True` (the default), each row represents a
            variable. If `False`, each column represents a variable.
        ddof: Delta degrees of freedom used in the normalizer
            `n_obs - ddof`. Defaults to `1` (matches numpy's default
            `bias=False`).

    Returns:
        The covariance matrix, of shape `(n_vars, n_vars)`.

    Raises:
        NumojoError: If `m` is not 2-dimensional.
        NumojoError: If `ddof` is not smaller than the number of
            observations.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var m = nm.fromstring("[[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]")
        print(nm.cov(m, rowvar=False))
    ```
    """

    if m.ndim != 2:
        raise Error(
            NumojoError(
                category="shape",
                message="`cov` currently only supports 2-dimensional arrays.",
                location="cov",
            )
        )

    var X = m.copy() if rowvar else transpose(m)
    var n_obs = X.shape[1]
    if ddof >= n_obs:
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "ddof ({}) should be smaller than the number of"
                    " observations ({})"
                ).format(ddof, n_obs),
                location="cov",
            )
        )

    var Xf = X.astype[returned_dtype]()
    var row_means = mean[returned_dtype](Xf, axis=1)
    var demeaned = Xf - _broadcast_back_to(row_means, Xf.shape, axis=1)

    return matmul(demeaned, transpose(demeaned)) / Scalar[returned_dtype](
        n_obs - ddof
    )


def corrcoef[
    dtype: DType, //, returned_dtype: DType = DType.float64
](m: NDArray[dtype], rowvar: Bool = True) raises -> NDArray[returned_dtype]:
    """
    Compute the Pearson correlation-coefficient matrix of a set of
    variables.

    Matches `numpy.corrcoef` for a 2-D input: the covariance matrix
    (see `cov`) is normalized so that the diagonal is all `1`s.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        m: A 2-D array of variables and observations.
        rowvar: If `True` (the default), each row represents a
            variable. If `False`, each column represents a variable.

    Returns:
        The correlation-coefficient matrix, of shape
        `(n_vars, n_vars)`.

    Raises:
        NumojoError: If `m` is not 2-dimensional.
        NumojoError: If there are fewer than 2 observations.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var m = nm.fromstring("[[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]")
        print(nm.corrcoef(m, rowvar=False))
    ```
    """

    var C = cov[returned_dtype](m, rowvar=rowvar, ddof=1)
    var n = C.shape[0]

    var stddevs = List[Scalar[returned_dtype]](capacity=n)
    for i in range(n):
        stddevs.append(sqrt(C.item(i, i)))

    var R = NDArray[returned_dtype](C.shape)
    for i in range(n):
        for j in range(n):
            R[Item(i, j)] = C.item(i, j) / (stddevs[i] * stddevs[j])

    return R^
