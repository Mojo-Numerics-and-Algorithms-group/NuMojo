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
- `histogram_bin_edges`: Compute only the bin edges for `histogram`.
- `histogram2d`: Bivariate histogram over equal-width bins.
- `histogramdd`: Multivariate histogram over equal-width bins.
- `digitize`: Indices of the bins to which each value belongs.
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


def histogram_bin_edges[
    dtype: DType, //, returned_dtype: DType = DType.float64
](a: NDArray[dtype], bins: Int = 10) raises -> NDArray[
    returned_dtype
] where dtype.is_floating_point():
    """
    Compute the bin edges for `histogram`, without the counts.

    Matches `numpy.histogram_bin_edges` with the default `range` (the
    data's own min/max).

    Parameters:
        dtype: The element type.
        returned_dtype: The data type of the returned bin edges,
            defaulting to float64.

    Args:
        a: An array of values the edges would bin. It is treated as
            flat regardless of its shape.
        bins: The number of equal-width bins.

    Returns:
        An array of `bins + 1` bin edges.

    Raises:
        NumojoError: If `bins` is not positive.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[0.0, 0.5, 1.5, 2.5]")
        print(nm.histogram_bin_edges(a, bins=3))  # [0.0, 0.833.., 1.666.., 2.5]
    ```
    """

    if bins <= 0:
        raise Error(
            NumojoError(
                category="value",
                message="`bins` must be a positive integer.",
                location="histogram_bin_edges",
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

    return histogram_bin_edges[returned_dtype=returned_dtype](a, bins, lo, hi)


def histogram_bin_edges[
    dtype: DType, //, returned_dtype: DType = DType.float64
](
    a: NDArray[dtype],
    bins: Int,
    range_lo: Scalar[returned_dtype],
    range_hi: Scalar[returned_dtype],
) raises -> NDArray[returned_dtype] where dtype.is_floating_point():
    """
    Compute the bin edges for `histogram` over an explicit range.

    Matches `numpy.histogram_bin_edges` with an explicit `range`.

    Parameters:
        dtype: The element type.
        returned_dtype: The data type of the returned bin edges,
            defaulting to float64.

    Args:
        a: Unused except for dtype inference; present to match
            `numpy.histogram_bin_edges`'s signature.
        bins: The number of equal-width bins.
        range_lo: The lower edge of the first bin.
        range_hi: The upper edge of the last bin.

    Returns:
        An array of `bins + 1` bin edges.

    Raises:
        NumojoError: If `bins` is not positive, or if `range_lo` is
            not smaller than `range_hi`.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var a = nm.fromstring("[0.0, 0.5, 1.5, 2.5]")
        print(nm.histogram_bin_edges(a, bins=3, range_lo=0.0, range_hi=3.0))
        # [0.0, 1.0, 2.0, 3.0]
    ```
    """

    if bins <= 0:
        raise Error(
            NumojoError(
                category="value",
                message="`bins` must be a positive integer.",
                location="histogram_bin_edges",
            )
        )
    if range_lo >= range_hi:
        raise Error(
            NumojoError(
                category="value",
                message="`range_lo` must be smaller than `range_hi`.",
                location="histogram_bin_edges",
            )
        )

    var width = (range_hi - range_lo) / Scalar[returned_dtype](bins)
    var edges = zeros[returned_dtype](Shape(bins + 1))
    for i in range(bins + 1):
        edges[Item(i)] = range_lo + Scalar[returned_dtype](i) * width
    edges[Item(bins)] = range_hi

    return edges^


def histogram2d[
    dtype: DType, //, returned_dtype: DType = DType.float64
](x: NDArray[dtype], y: NDArray[dtype], bins: Int = 10) raises -> Tuple[
    NDArray[DType.int], NDArray[returned_dtype], NDArray[returned_dtype]
] where dtype.is_floating_point():
    """
    Compute a bivariate histogram of two data samples.

    Matches `numpy.histogram2d` with the default `range` (each sample's
    own min/max) and an integer `bins` shared by both dimensions.

    Parameters:
        dtype: The element type.
        returned_dtype: The data type of the returned bin edges,
            defaulting to float64.

    Args:
        x: Sample of x-values. Treated as flat regardless of shape.
        y: Sample of y-values, the same size as `x`.
        bins: The number of equal-width bins along each axis.

    Returns:
        A tuple `(counts, xedges, yedges)`: `counts` has shape
        `(bins, bins)`, and `xedges`/`yedges` each have length
        `bins + 1`.

    Raises:
        NumojoError: If `x` and `y` are not the same size, or if `bins`
            is not positive.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var x = nm.fromstring("[0.0, 0.5, 1.5, 2.5]")
        var y = nm.fromstring("[0.0, 1.0, 1.0, 2.0]")
        var result = nm.histogram2d(x, y, bins=2)
        print(result[0])  # counts
    ```
    """

    if x.size != y.size:
        raise Error(
            NumojoError(
                category="shape",
                message="`x` and `y` must be the same size.",
                location="histogram2d",
            )
        )
    if bins <= 0:
        raise Error(
            NumojoError(
                category="value",
                message="`bins` must be a positive integer.",
                location="histogram2d",
            )
        )

    var xlo: Scalar[returned_dtype]
    var xhi: Scalar[returned_dtype]
    var ylo: Scalar[returned_dtype]
    var yhi: Scalar[returned_dtype]
    if x.size == 0:
        xlo = 0
        xhi = 1
        ylo = 0
        yhi = 1
    else:
        xlo = array_min(x).cast[returned_dtype]()
        xhi = array_max(x).cast[returned_dtype]()
        if xlo == xhi:
            xlo -= 0.5
            xhi += 0.5
        ylo = array_min(y).cast[returned_dtype]()
        yhi = array_max(y).cast[returned_dtype]()
        if ylo == yhi:
            ylo -= 0.5
            yhi += 0.5

    var xwidth = (xhi - xlo) / Scalar[returned_dtype](bins)
    var ywidth = (yhi - ylo) / Scalar[returned_dtype](bins)

    var xedges = zeros[returned_dtype](Shape(bins + 1))
    var yedges = zeros[returned_dtype](Shape(bins + 1))
    for i in range(bins + 1):
        xedges[Item(i)] = xlo + Scalar[returned_dtype](i) * xwidth
        yedges[Item(i)] = ylo + Scalar[returned_dtype](i) * ywidth
    xedges[Item(bins)] = xhi
    yedges[Item(bins)] = yhi

    var counts = zeros[DType.int](Shape(bins, bins))
    for k in range(x.size):
        var xv = x.item(k).cast[returned_dtype]()
        var yv = y.item(k).cast[returned_dtype]()
        if (xv < xlo) or (xv > xhi) or (yv < ylo) or (yv > yhi):
            continue
        var ix = Int((xv - xlo) / xwidth)
        if ix >= bins:
            ix = bins - 1
        var iy = Int((yv - ylo) / ywidth)
        if iy >= bins:
            iy = bins - 1
        counts[Item(ix, iy)] = counts.item(ix, iy) + 1

    return Tuple(counts^, xedges^, yedges^)


def histogramdd[
    dtype: DType, //, returned_dtype: DType = DType.float64
](sample: NDArray[dtype], bins: Int = 10) raises -> Tuple[
    NDArray[DType.int], List[NDArray[returned_dtype]]
] where dtype.is_floating_point():
    """
    Compute a multivariate histogram of a data sample.

    Matches `numpy.histogramdd` with the default `range` (each column's
    own min/max) and an integer `bins` shared by every dimension.

    Parameters:
        dtype: The element type.
        returned_dtype: The data type of the returned bin edges,
            defaulting to float64.

    Args:
        sample: A 2-d array of shape `(n_points, n_dims)`.
        bins: The number of equal-width bins along each axis.

    Returns:
        A tuple `(counts, edges)`: `counts` has shape
        `(bins, bins, ..., bins)` with one `bins` per column of
        `sample`, and `edges` is a list of one bin-edges array
        (length `bins + 1`) per column.

    Raises:
        NumojoError: If `sample` is not 2-dimensional, or if `bins` is
            not positive.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var sample = nm.fromstring("[[0.0, 0.0], [0.5, 1.0], [2.5, 2.0]]")
        var result = nm.histogramdd(sample, bins=2)
        print(result[0])  # counts
    ```
    """

    if sample.ndim != 2:
        raise Error(
            NumojoError(
                category="shape",
                message="`sample` must be a 2-dimensional array of shape"
                " (n_points, n_dims).",
                location="histogramdd",
            )
        )
    if bins <= 0:
        raise Error(
            NumojoError(
                category="value",
                message="`bins` must be a positive integer.",
                location="histogramdd",
            )
        )

    var n = sample.shape[0]
    var d = sample.shape[1]

    var los = List[Scalar[returned_dtype]]()
    var his = List[Scalar[returned_dtype]]()
    var widths = List[Scalar[returned_dtype]]()
    var edges_list = List[NDArray[returned_dtype]]()

    for col in range(d):
        var lo: Scalar[returned_dtype]
        var hi: Scalar[returned_dtype]
        if n == 0:
            lo = 0
            hi = 1
        else:
            lo = sample.item(0, col).cast[returned_dtype]()
            hi = lo
            for row in range(1, n):
                var v = sample.item(row, col).cast[returned_dtype]()
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v
            if lo == hi:
                lo -= 0.5
                hi += 0.5
        var width = (hi - lo) / Scalar[returned_dtype](bins)
        los.append(lo)
        his.append(hi)
        widths.append(width)

        var edges = zeros[returned_dtype](Shape(bins + 1))
        for i in range(bins + 1):
            edges[Item(i)] = lo + Scalar[returned_dtype](i) * width
        edges[Item(bins)] = hi
        edges_list.append(edges^)

    var counts_shape = List[Int]()
    for _ in range(d):
        counts_shape.append(bins)
    var counts = zeros[DType.int](Shape(counts_shape))

    for row in range(n):
        var idx = List[Int]()
        var out_of_range = False
        for col in range(d):
            var v = sample.item(row, col).cast[returned_dtype]()
            if (v < los[col]) or (v > his[col]):
                out_of_range = True
                break
            var bin_idx = Int((v - los[col]) / widths[col])
            if bin_idx >= bins:
                bin_idx = bins - 1
            idx.append(bin_idx)
        if out_of_range:
            continue
        var cur = counts[Item(idx.copy())]
        counts[Item(idx)] = cur + 1

    return Tuple(counts^, edges_list^)


def digitize[
    dtype: DType
](
    x: NDArray[dtype], bins: NDArray[dtype], right: Bool = False
) raises -> NDArray[DType.int] where dtype.is_numeric():
    """
    Return the indices of the bins to which each value in `x` belongs.

    Matches `numpy.digitize`. `bins` must be monotonically increasing or
    decreasing; monotonicity is determined from its endpoints and is not
    otherwise verified.

    Parameters:
        dtype: The element type.

    Args:
        x: Array of values to bin. Treated as flat regardless of shape.
        bins: A 1-d array of bin edges.
        right: If `True`, intervals include the right edge instead of
            the left.

    Returns:
        An array of bin indices, same shape as `x`.

    Raises:
        NumojoError: If `bins` is not 1-dimensional.

    Examples:
    ```mojo
    import numojo as nm
    def main() raises:
        var x = nm.fromstring("[0.2, 6.4, 3.0, 1.6]")
        var bins = nm.fromstring("[0.0, 1.0, 2.5, 4.0, 10.0]")
        print(nm.digitize(x, bins))  # [1, 4, 3, 2]
    ```
    """

    if bins.ndim != 1:
        raise Error(
            NumojoError(
                category="shape",
                message="`digitize` requires `bins` to be 1-dimensional.",
                location="digitize",
            )
        )

    var bins_c = bins.contiguous()
    var n = bins_c.size
    var x_c = x.contiguous()
    var result = NDArray[DType.int](Shape(x_c.shape))
    var side_right = not right

    var decreasing = n >= 2 and bins_c.unsafe_get(0) > bins_c.unsafe_get(
        n - 1
    )
    var search_bins: NDArray[dtype]
    if decreasing:
        search_bins = NDArray[dtype](Shape(n))
        for i in range(n):
            search_bins.unsafe_set(i, bins_c.unsafe_get(n - 1 - i))
    else:
        search_bins = bins_c.copy()

    for i in range(x_c.size):
        var target = x_c.unsafe_get(i)
        var lo = 0
        var hi = n
        if not side_right:
            while lo < hi:
                var mid = (lo + hi) // 2
                if search_bins.unsafe_get(mid) < target:
                    lo = mid + 1
                else:
                    hi = mid
        else:
            while lo < hi:
                var mid = (lo + hi) // 2
                if search_bins.unsafe_get(mid) <= target:
                    lo = mid + 1
                else:
                    hi = mid
        var idx = lo
        if decreasing:
            idx = n - idx
        result.unsafe_set(i, Scalar[DType.int](idx))

    return result^
