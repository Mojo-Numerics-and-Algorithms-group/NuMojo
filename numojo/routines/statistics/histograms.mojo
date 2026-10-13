# ===----------------------------------------------------------------------=== #
# NuMojo: Histograms
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Histograms (numojo.routines.statistics.histograms).
===================================================
Binning and counting of array elements.

Exports
-------
- `bincount`: Count of occurrences of each non-negative integer.
- `histogram`: Count of elements falling into equal-width bins.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.error import NumojoError
from numojo.core.indexing.item import Item
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import Shape
from numojo.routines.creation import zeros
from numojo.routines.math.extrema import max as array_max
from numojo.routines.math.extrema import min as array_min


def bincount[
    dtype: DType
](x: NDArray[dtype], minlength: Int = 0) raises -> NDArray[
    DType.int
] where dtype.is_integral():
    """
    Count the occurrences of each non-negative integer in an array.

    Matches `numpy.bincount` without weights.

    Parameters:
        dtype: The element type. Must be an integer type.

    Args:
        x: A 1-d array of non-negative integers.
        minlength: The minimum number of bins for the output array.

    Returns:
        An array `counts` such that `counts[i]` is the number of
        occurrences of `i` in `x`, of length
        `max(x.max() + 1, minlength)`.

    Raises:
        NumojoError: If `x` is not 1-dimensional.
        NumojoError: If `x` contains a negative value.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var x = nm.fromstring[nm.i64]("[1, 1, 2, 4]")
        print(nm.bincount(x))  # [0, 2, 1, 0, 1]
    ```
    """

    if x.ndim != 1:
        raise Error(
            NumojoError(
                category="shape",
                message="`bincount` only supports 1-dimensional arrays.",
                location="bincount",
            )
        )

    var n = minlength
    if x.size > 0:
        var x_max = Int(array_max(x))
        if x_max < 0:
            raise Error(
                NumojoError(
                    category="value",
                    message="`bincount` does not support negative values.",
                    location="bincount",
                )
            )
        n = max(n, x_max + 1)

    var counts = zeros[DType.int](Shape(n))
    for i in range(x.size):
        var val = Int(x.item(i))
        if val < 0:
            raise Error(
                NumojoError(
                    category="value",
                    message="`bincount` does not support negative values.",
                    location="bincount",
                )
            )
        counts[Item(val)] = counts.item(val) + 1

    return counts^


def bincount[
    dtype: DType, wdtype: DType
](
    x: NDArray[dtype], weights: NDArray[wdtype], minlength: Int = 0
) raises -> NDArray[wdtype] where dtype.is_integral():
    """
    Sum the weights of each non-negative integer's occurrences in an
    array.

    Matches `numpy.bincount` with weights.

    Parameters:
        dtype: The element type of `x`. Must be an integer type.
        wdtype: The element type of `weights`.

    Args:
        x: A 1-d array of non-negative integers.
        weights: A 1-d array of weights, the same length as `x`.
        minlength: The minimum number of bins for the output array.

    Returns:
        An array `out` such that `out[i]` is the sum of `weights` at
        the positions where `x` equals `i`, of length
        `max(x.max() + 1, minlength)`.

    Raises:
        NumojoError: If `x` is not 1-dimensional, or if `weights` is
            not the same length as `x`.
        NumojoError: If `x` contains a negative value.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var x = nm.fromstring[nm.i64]("[1, 1, 2]")
        var w = nm.fromstring("[0.5, 0.5, 2.0]")
        print(nm.bincount(x, w))  # [0.0, 1.0, 2.0]
    ```
    """

    if x.ndim != 1:
        raise Error(
            NumojoError(
                category="shape",
                message="`bincount` only supports 1-dimensional arrays.",
                location="bincount",
            )
        )
    if weights.size != x.size:
        raise Error(
            NumojoError(
                category="shape",
                message="`weights` must be the same length as `x`.",
                location="bincount",
            )
        )

    var n = minlength
    if x.size > 0:
        var x_max = Int(array_max(x))
        if x_max < 0:
            raise Error(
                NumojoError(
                    category="value",
                    message="`bincount` does not support negative values.",
                    location="bincount",
                )
            )
        n = max(n, x_max + 1)

    var out = zeros[wdtype](Shape(n))
    for i in range(x.size):
        var val = Int(x.item(i))
        if val < 0:
            raise Error(
                NumojoError(
                    category="value",
                    message="`bincount` does not support negative values.",
                    location="bincount",
                )
            )
        out[Item(val)] = out.item(val) + weights.item(i)

    return out^


def histogram[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], bins: Int = 10) raises -> Tuple[
    NDArray[DType.int], NDArray[returned_dtype]
] where dtype.is_floating_point():
    """
    Compute a histogram of equal-width bins spanning the data range.

    Matches `numpy.histogram` with the default `range` (the data's own
    min/max).

    Parameters:
        dtype: The element type.
        returned_dtype: The data type of the returned bin edges,
            defaulting to float64.

    Args:
        a: An array of values to bin. It is treated as flat
            regardless of its shape.
        bins: The number of equal-width bins.

    Returns:
        A tuple `(counts, bin_edges)`: `counts` has length `bins` and
        `bin_edges` has length `bins + 1`.

    Raises:
        NumojoError: If `bins` is not positive.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[0.0, 0.5, 1.5, 2.5]")
        var counts_edges = nm.histogram(a, bins=3)
        print(counts_edges[0])  # counts per bin
        print(counts_edges[1])  # bin edges
    ```
    """

    if bins <= 0:
        raise Error(
            NumojoError(
                category="value",
                message="`bins` must be a positive integer.",
                location="histogram",
            )
        )

    var lo: Scalar[returned_dtype]
    var hi: Scalar[returned_dtype]
    if a.size == 0:
        lo = 0
        hi = 1
    else:
        lo = array_min(a).cast[returned_dtype]()
        hi = array_max(a).cast[returned_dtype]()
        if lo == hi:
            lo -= 0.5
            hi += 0.5

    return histogram[returned_dtype=returned_dtype](a, bins, lo, hi)


def histogram[
    dtype: DType, //, returned_dtype: DType = DType.float64
](
    a: NDArray[dtype],
    bins: Int,
    range_lo: Scalar[returned_dtype],
    range_hi: Scalar[returned_dtype],
) raises -> Tuple[
    NDArray[DType.int], NDArray[returned_dtype]
] where dtype.is_floating_point():
    """
    Compute a histogram of equal-width bins spanning an explicit
    range.

    Matches `numpy.histogram` with an explicit `range`. Values outside
    `[range_lo, range_hi]` are ignored, as in numpy.

    Parameters:
        dtype: The element type.
        returned_dtype: The data type of the returned bin edges,
            defaulting to float64.

    Args:
        a: An array of values to bin. It is treated as flat
            regardless of its shape.
        bins: The number of equal-width bins.
        range_lo: The lower edge of the first bin.
        range_hi: The upper edge of the last bin.

    Returns:
        A tuple `(counts, bin_edges)`: `counts` has length `bins` and
        `bin_edges` has length `bins + 1`.

    Raises:
        NumojoError: If `bins` is not positive, or if `range_lo` is
            not smaller than `range_hi`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[0.0, 0.5, 1.5, 2.5]")
        var counts_edges = nm.histogram(a, bins=3, range_lo=0.0, range_hi=3.0)
        print(counts_edges[0])  # [1, 1, 2]
        print(counts_edges[1])  # [0.0, 1.0, 2.0, 3.0]
    ```
    """

    if bins <= 0:
        raise Error(
            NumojoError(
                category="value",
                message="`bins` must be a positive integer.",
                location="histogram",
            )
        )
    if range_lo >= range_hi:
        raise Error(
            NumojoError(
                category="value",
                message="`range_lo` must be smaller than `range_hi`.",
                location="histogram",
            )
        )

    var width = (range_hi - range_lo) / Scalar[returned_dtype](bins)

    var edges = zeros[returned_dtype](Shape(bins + 1))
    for i in range(bins + 1):
        edges[Item(i)] = range_lo + Scalar[returned_dtype](i) * width
    edges[Item(bins)] = range_hi

    var counts = zeros[DType.int](Shape(bins))
    for i in range(a.size):
        var val = a.item(i).cast[returned_dtype]()
        if (val < range_lo) or (val > range_hi):
            continue
        var idx = Int((val - range_lo) / width)
        if idx >= bins:
            idx = bins - 1
        counts[Item(idx)] = counts.item(idx) + 1

    return Tuple(counts^, edges^)
