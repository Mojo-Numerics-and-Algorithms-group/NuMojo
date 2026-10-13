# ===----------------------------------------------------------------------=== #
# NuMojo: Percentiles and quantiles
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Percentiles (numojo.routines.statistics.percentiles).
=====================================================
Order statistics based on linear interpolation between sorted elements.

Exports
-------
- `quantile`: Value at a given fraction of sorted array elements.
- `percentile`: Value at a given percentage of sorted array elements.
- `quantile_1d`: Backend function for `quantile`, with or without `axis`.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.ndarray import NDArray
from numojo.routines.creation import _0darray
from numojo.routines.sorting import sort


def quantile_1d[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], q: Scalar[returned_dtype]) raises -> Scalar[
    returned_dtype
]:
    """
    Quantile of all items of an array using linear interpolation.
    Regardless of the shape of input, it is treated as a 1-d array.
    It is the backend function for `quantile`, with or without `axis`.

    Matches `numpy.quantile` with the default (`"linear"`) interpolation
    method.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: A 1-d array.
        q: Quantile to compute, in the inclusive range `[0, 1]`.

    Returns:
        The `q`-th quantile of the array as a scalar of `returned_dtype`.

    Raises:
        NumojoError: If `q` is not in the range `[0, 1]`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[1.0, 2.0, 3.0, 4.0]")
        print(nm.quantile(a, Float64(0.5)))  # 2.5
    ```
    """

    if (q < 0) or (q > 1):
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "`q` must be in the range [0, 1], got {}"
                ).format(q),
                location="quantile",
            )
        )

    var sorted_array = sort(a)
    var virtual_index = q * Scalar[returned_dtype](a.size - 1)
    var lo = Int(virtual_index)
    var hi = min(lo + 1, a.size - 1)
    var frac = virtual_index - Scalar[returned_dtype](lo)

    var lo_val = sorted_array.item(lo).cast[returned_dtype]()
    var hi_val = sorted_array.item(hi).cast[returned_dtype]()
    return lo_val + (hi_val - lo_val) * frac


def quantile[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], q: Scalar[returned_dtype]) raises -> Scalar[
    returned_dtype
]:
    """
    Quantile of all items of an array using linear interpolation.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: An array.
        q: Quantile to compute, in the inclusive range `[0, 1]`.

    Returns:
        The `q`-th quantile of the array as a scalar of `returned_dtype`.

    Raises:
        NumojoError: If `q` is not in the range `[0, 1]`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[1.0, 2.0, 3.0, 4.0]")
        print(nm.quantile(a, Float64(0.5)))  # 2.5
    ```
    """

    return quantile_1d[returned_dtype](a, q)


def quantile[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], q: Scalar[returned_dtype], axis: Int) raises -> NDArray[
    returned_dtype
]:
    """
    Quantile of the array elements along the given axis.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: An array.
        q: Quantile to compute, in the inclusive range `[0, 1]`.
        axis: The axis along which the quantile is computed.

    Returns:
        An array of quantiles with `axis` removed.

    Raises:
        NumojoError: If `axis` is out of bounds.
        NumojoError: If `q` is not in the range `[0, 1]`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[[1.0, 2.0], [3.0, 4.0]]")
        print(nm.quantile(a, Float64(0.5), axis=0))  # [2.0, 3.0]
    ```
    """

    var normalized_axis = axis
    if normalized_axis < 0:
        normalized_axis += a.ndim
    if (normalized_axis < 0) or (normalized_axis >= a.ndim):
        raise Error(
            NumojoError(
                category="index",
                message=String(
                    "Error in `quantile`: Axis {} not in bound [-{}, {})"
                ).format(axis, a.ndim, a.ndim),
                location="quantile",
            )
        )

    var iterator = a.iter_along_axis(axis=normalized_axis)
    var res: NDArray[returned_dtype]

    if a.ndim == 1:
        res = _0darray[returned_dtype](0)
        res.unsafe_set(0, quantile_1d[returned_dtype](a, q))
    else:
        res = NDArray[returned_dtype](a.shape.pop(axis=normalized_axis))
        for i in range(a.size // a.shape[normalized_axis]):
            res.unsafe_set(i, quantile_1d[returned_dtype](iterator.ith(i), q))

    return res^


def percentile[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], p: Scalar[returned_dtype]) raises -> Scalar[
    returned_dtype
]:
    """
    Percentile of all items of an array using linear interpolation.

    Equivalent to `quantile(a, p / 100)`.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: An array.
        p: Percentile to compute, in the inclusive range `[0, 100]`.

    Returns:
        The `p`-th percentile of the array as a scalar of `returned_dtype`.

    Raises:
        NumojoError: If `p` is not in the range `[0, 100]`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[1.0, 2.0, 3.0, 4.0]")
        print(nm.percentile(a, Float64(50)))  # 2.5
    ```
    """

    if (p < 0) or (p > 100):
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "`p` must be in the range [0, 100], got {}"
                ).format(p),
                location="percentile",
            )
        )

    return quantile_1d[returned_dtype](a, p / 100)


def percentile[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], p: Scalar[returned_dtype], axis: Int) raises -> NDArray[
    returned_dtype
]:
    """
    Percentile of the array elements along the given axis.

    Equivalent to `quantile(a, p / 100, axis=axis)`.

    Parameters:
        dtype: The element type.
        returned_dtype: The returned data type, defaulting to float64.

    Args:
        a: An array.
        p: Percentile to compute, in the inclusive range `[0, 100]`.
        axis: The axis along which the percentile is computed.

    Returns:
        An array of percentiles with `axis` removed.

    Raises:
        NumojoError: If `axis` is out of bounds.
        NumojoError: If `p` is not in the range `[0, 100]`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[[1.0, 2.0], [3.0, 4.0]]")
        print(nm.percentile(a, Float64(50), axis=0))  # [2.0, 3.0]
    ```
    """

    if (p < 0) or (p > 100):
        raise Error(
            NumojoError(
                category="value",
                message=String(
                    "`p` must be in the range [0, 100], got {}"
                ).format(p),
                location="percentile",
            )
        )

    return quantile[returned_dtype](a, p / 100, axis)
