from std.python import Python, PythonObject
from std.testing.testing import assert_raises, assert_true
from std.utils.numerics import inf, nan, neg_inf
from utils_for_test import check, check_is_close, check_values_close
from std.testing import TestSuite

import numojo as nm
from numojo.prelude import *

# ===-----------------------------------------------------------------------===#
# Statistics
# ===-----------------------------------------------------------------------===#


def test_mean_median_var_std() raises:
    var np = Python.import_module("numpy")
    var sp = Python.import_module("scipy")
    var A = nm.random.randn(3, 4, 5)
    var Anp = A.to_numpy()

    assert_true(
        np.all(np.isclose(nm.mean(A), np.mean(Anp), atol=PythonObject(0.001))),
        "`mean` is broken",
    )
    for axis in range(3):
        check_is_close(
            nm.mean(A, axis=axis),
            np.mean(Anp, axis=axis),
            String("`mean` is broken for axis {}").format(axis),
        )

    assert_true(
        np.all(
            np.isclose(nm.median(A), np.median(Anp), atol=PythonObject(0.001))
        ),
        "`median` is broken",
    )
    for axis in range(3):
        check_is_close(
            nm.median(A, axis),
            np.median(Anp, axis),
            String("`median` is broken for axis {}").format(axis),
        )

    assert_true(
        np.all(
            np.isclose(
                nm.mode(A),
                sp.stats.mode(Anp, axis=PythonObject(None)).mode,
                atol=PythonObject(0.001),
            )
        ),
        "`mode` is broken",
    )
    for axis in range(3):
        check_is_close(
            nm.mode(A, axis),
            sp.stats.mode(Anp, axis).mode,
            String("`mode` is broken for axis {}").format(axis),
        )

    assert_true(
        np.all(
            np.isclose(nm.variance(A), np.`var`(Anp), atol=PythonObject(0.001))
        ),
        "`variance` is broken",
    )
    for axis in range(3):
        check_is_close(
            nm.variance(A, axis),
            np.`var`(Anp, axis),
            String("`variance` is broken for axis {}").format(axis),
        )

    assert_true(
        np.all(np.isclose(nm.stddev(A), np.std(Anp), atol=PythonObject(0.001))),
        "`std` is broken",
    )
    for axis in range(3):
        check_is_close(
            nm.stddev(A, axis),
            np.std(Anp, axis),
            String("`std` is broken for axis {}").format(axis),
        )


def test_quantile_percentile() raises:
    var np = Python.import_module("numpy")
    var A = nm.random.randn(6, 4)
    var Anp = A.to_numpy()

    for q in [0.0, 0.01, 0.25, 0.5, 0.503, 0.99, 1.0]:
        assert_true(
            np.isclose(
                nm.quantile(A, Float64(q)),
                np.quantile(Anp, q),
                atol=PythonObject(0.001),
            ),
            String("`quantile` is broken for q={}").format(q),
        )
        assert_true(
            np.isclose(
                nm.percentile(A, Float64(q * 100)),
                np.percentile(Anp, q * 100),
                atol=PythonObject(0.001),
            ),
            String("`percentile` is broken for p={}").format(q * 100),
        )

    for axis in range(2):
        check_is_close(
            nm.quantile(A, Float64(0.75), axis=axis),
            np.quantile(Anp, 0.75, axis=axis),
            String("`quantile` is broken for axis {}").format(axis),
        )
        check_is_close(
            nm.percentile(A, Float64(75), axis=axis),
            np.percentile(Anp, 75, axis=axis),
            String("`percentile` is broken for axis {}").format(axis),
        )

    with assert_raises():
        _ = nm.quantile(A, Float64(1.5))

    with assert_raises():
        _ = nm.percentile(A, Float64(150))


def test_cov_corrcoef() raises:
    var np = Python.import_module("numpy")
    var A = nm.random.randn(3, 6)
    var Anp = A.to_numpy()

    check_is_close(nm.cov(A), np.cov(Anp), "`cov` is broken for rowvar=True")
    check_is_close(
        nm.cov(A, rowvar=False),
        np.cov(Anp, rowvar=False),
        "`cov` is broken for rowvar=False",
    )
    check_is_close(
        nm.corrcoef(A),
        np.corrcoef(Anp),
        "`corrcoef` is broken for rowvar=True",
    )
    check_is_close(
        nm.corrcoef(A, rowvar=False),
        np.corrcoef(Anp, rowvar=False),
        "`corrcoef` is broken for rowvar=False",
    )

    with assert_raises():
        # ddof not smaller than the number of observations
        _ = nm.cov(A, ddof=6)


def test_bincount() raises:
    var np = Python.import_module("numpy")
    var x = nm.fromstring[nm.i64]("[1, 1, 2, 4, 1, 0]")
    var xnp = x.to_numpy()

    check_is_close(nm.bincount(x), np.bincount(xnp), "`bincount` is broken")
    check_is_close(
        nm.bincount(x, minlength=8),
        np.bincount(xnp, minlength=8),
        "`bincount` is broken for minlength",
    )

    var w = nm.fromstring("[0.5, 1.5, 2.0, 3.0, 1.0, 0.25]")
    check_is_close(
        nm.bincount(x, w),
        np.bincount(xnp, weights=w.to_numpy()),
        "`bincount` is broken for weights",
    )

    with assert_raises():
        var neg = nm.fromstring[nm.i64]("[1, -1, 2]")
        _ = nm.bincount(neg)


def test_histogram() raises:
    var np = Python.import_module("numpy")
    var a = nm.random.randn(200)
    var anp = a.to_numpy()

    var ce = nm.histogram(a, bins=12)
    var np_hist = np.histogram(anp, bins=12)
    check_is_close(ce[0], np_hist[0], "`histogram` counts broken")
    check_is_close(ce[1], np_hist[1], "`histogram` edges broken")

    var ce2 = nm.histogram(a, 10, Float64(-1.0), Float64(1.0))
    var np_hist2 = np.histogram(anp, bins=10, range=Python.tuple(-1.0, 1.0))
    check_is_close(
        ce2[0], np_hist2[0], "`histogram` counts broken for explicit range"
    )
    check_is_close(
        ce2[1], np_hist2[1], "`histogram` edges broken for explicit range"
    )

    # Degenerate case: all-equal data.
    var same = nm.fromstring("[3.0, 3.0, 3.0]")
    var ce3 = nm.histogram(same, bins=4)
    var np_hist3 = np.histogram(same.to_numpy(), bins=4)
    check_is_close(
        ce3[0], np_hist3[0], "`histogram` counts broken for degenerate data"
    )
    check_is_close(
        ce3[1], np_hist3[1], "`histogram` edges broken for degenerate data"
    )

    with assert_raises():
        _ = nm.histogram(a, bins=0)


def test_histogram_bin_edges() raises:
    var np = Python.import_module("numpy")
    var a = nm.random.randn(200)
    var anp = a.to_numpy()

    check_is_close(
        nm.histogram_bin_edges(a, bins=12),
        np.histogram_bin_edges(anp, bins=12),
        "`histogram_bin_edges` broken for default range",
    )
    check_is_close(
        nm.histogram_bin_edges(a, 10, Float64(-1.0), Float64(1.0)),
        np.histogram_bin_edges(anp, bins=10, range=Python.tuple(-1.0, 1.0)),
        "`histogram_bin_edges` broken for explicit range",
    )

    with assert_raises():
        _ = nm.histogram_bin_edges(a, bins=0)


def test_histogram2d() raises:
    var np = Python.import_module("numpy")
    var x = nm.random.randn(150)
    var y = nm.random.randn(150)
    var xnp = x.to_numpy()
    var ynp = y.to_numpy()

    var result = nm.histogram2d(x, y, bins=6)
    var np_result = np.histogram2d(xnp, ynp, bins=6)
    check_is_close(result[0], np_result[0], "`histogram2d` counts broken")
    check_is_close(result[1], np_result[1], "`histogram2d` xedges broken")
    check_is_close(result[2], np_result[2], "`histogram2d` yedges broken")

    with assert_raises():
        _ = nm.histogram2d(x, nm.random.randn(5), bins=6)

    with assert_raises():
        _ = nm.histogram2d(x, y, bins=0)


def test_histogramdd() raises:
    var np = Python.import_module("numpy")
    var sample = nm.random.randn(100, 3)
    var samplenp = sample.to_numpy()

    var result = nm.histogramdd(sample, bins=4)
    var np_result = np.histogramdd(samplenp, bins=4)
    check_is_close(result[0], np_result[0], "`histogramdd` counts broken")
    for d in range(3):
        check_is_close(
            result[1][d],
            np_result[1][d],
            String("`histogramdd` edges broken for axis {}").format(d),
        )

    with assert_raises():
        _ = nm.histogramdd(sample, bins=0)

    with assert_raises():
        _ = nm.histogramdd(nm.random.randn(10), bins=4)


def test_digitize() raises:
    var np = Python.import_module("numpy")
    var x = nm.fromstring[nm.f64]("[0.2, 6.4, 3.0, 1.6, -1.0, 11.0]")
    var xnp = x.to_numpy()
    var bins_inc = nm.fromstring[nm.f64]("[0.0, 1.0, 2.5, 4.0, 10.0]")
    var bins_inc_np = bins_inc.to_numpy()

    check_is_close(
        nm.digitize(x, bins_inc),
        np.digitize(xnp, bins_inc_np),
        "`digitize` broken for increasing bins, right=False",
    )
    check_is_close(
        nm.digitize(x, bins_inc, right=True),
        np.digitize(xnp, bins_inc_np, right=True),
        "`digitize` broken for increasing bins, right=True",
    )

    var bins_dec = nm.fromstring[nm.f64]("[10.0, 4.0, 2.5, 1.0, 0.0]")
    var bins_dec_np = bins_dec.to_numpy()

    check_is_close(
        nm.digitize(x, bins_dec),
        np.digitize(xnp, bins_dec_np),
        "`digitize` broken for decreasing bins, right=False",
    )
    check_is_close(
        nm.digitize(x, bins_dec, right=True),
        np.digitize(xnp, bins_dec_np, right=True),
        "`digitize` broken for decreasing bins, right=True",
    )

    with assert_raises():
        var bad_bins = nm.random.randn(3, 2)
        _ = nm.digitize(x, bad_bins)


# ===-----------------------------------------------------------------------===#
# NaN-aware reductions
# ===-----------------------------------------------------------------------===#


def _make_nan_array() raises -> NDArray[f64]:
    return nm.array[f64](
        data=[
            Scalar[f64](1.0),
            nan[f64](),
            3.0,
            4.0,
            5.0,
            nan[f64](),
        ],
        shape=[2, 3],
    )


def test_nansum() raises:
    var np = Python.import_module("numpy")
    var A = _make_nan_array()
    var Anp = A.to_numpy()

    assert_true(
        np.isclose(nm.nansum(A), np.nansum(Anp), atol=PythonObject(0.001)),
        "`nansum` is broken",
    )
    for axis in range(2):
        check_is_close(
            nm.nansum(A, axis=axis),
            np.nansum(Anp, axis=axis),
            String("`nansum` is broken for axis {}").format(axis),
        )


def test_nanmean() raises:
    var np = Python.import_module("numpy")
    var A = _make_nan_array()
    var Anp = A.to_numpy()

    assert_true(
        np.isclose(nm.nanmean(A), np.nanmean(Anp), atol=PythonObject(0.001)),
        "`nanmean` is broken",
    )
    for axis in range(2):
        check_is_close(
            nm.nanmean(A, axis=axis),
            np.nanmean(Anp, axis=axis),
            String("`nanmean` is broken for axis {}").format(axis),
        )

    with assert_raises():
        var all_nan = nm.array[f64](data=[nan[f64](), nan[f64]()], shape=[2])
        _ = nm.nanmean(all_nan)


def test_nanmax_nanmin() raises:
    var np = Python.import_module("numpy")
    var A = _make_nan_array()
    var Anp = A.to_numpy()

    assert_true(
        np.isclose(nm.nanmax(A), np.nanmax(Anp), atol=PythonObject(0.001)),
        "`nanmax` is broken",
    )
    assert_true(
        np.isclose(nm.nanmin(A), np.nanmin(Anp), atol=PythonObject(0.001)),
        "`nanmin` is broken",
    )
    for axis in range(2):
        check_is_close(
            nm.nanmax(A, axis=axis),
            np.nanmax(Anp, axis=axis),
            String("`nanmax` is broken for axis {}").format(axis),
        )
        check_is_close(
            nm.nanmin(A, axis=axis),
            np.nanmin(Anp, axis=axis),
            String("`nanmin` is broken for axis {}").format(axis),
        )

    with assert_raises():
        var all_nan = nm.array[f64](data=[nan[f64](), nan[f64]()], shape=[2])
        _ = nm.nanmax(all_nan)

    with assert_raises():
        var all_nan = nm.array[f64](data=[nan[f64](), nan[f64]()], shape=[2])
        _ = nm.nanmin(all_nan)


def test_nanvar_nanstd() raises:
    var np = Python.import_module("numpy")
    var A = _make_nan_array()
    var Anp = A.to_numpy()

    assert_true(
        np.isclose(nm.nanvar(A), np.nanvar(Anp), atol=PythonObject(0.001)),
        "`nanvar` is broken",
    )
    assert_true(
        np.isclose(nm.nanstd(A), np.nanstd(Anp), atol=PythonObject(0.001)),
        "`nanstd` is broken",
    )
    assert_true(
        np.isclose(
            nm.nanvar(A, ddof=1),
            np.nanvar(Anp, ddof=PythonObject(1)),
            atol=PythonObject(0.001),
        ),
        "`nanvar` with ddof is broken",
    )
    for axis in range(2):
        check_is_close(
            nm.nanvar(A, axis=axis),
            np.nanvar(Anp, axis=axis),
            String("`nanvar` is broken for axis {}").format(axis),
        )
        check_is_close(
            nm.nanstd(A, axis=axis),
            np.nanstd(Anp, axis=axis),
            String("`nanstd` is broken for axis {}").format(axis),
        )

    with assert_raises():
        var all_nan = nm.array[f64](data=[nan[f64](), nan[f64]()], shape=[2])
        _ = nm.nanvar(all_nan)

    with assert_raises():
        # ddof not smaller than the number of non-NaN elements
        var a = nm.array[f64](data=[1.0, nan[f64]()], shape=[2])
        _ = nm.nanvar(a, ddof=1)


def test_nanmedian() raises:
    var np = Python.import_module("numpy")
    var A = _make_nan_array()
    var Anp = A.to_numpy()

    assert_true(
        np.isclose(
            nm.nanmedian(A), np.nanmedian(Anp), atol=PythonObject(0.001)
        ),
        "`nanmedian` is broken",
    )
    for axis in range(2):
        check_is_close(
            nm.nanmedian(A, axis=axis),
            np.nanmedian(Anp, axis=axis),
            String("`nanmedian` is broken for axis {}").format(axis),
        )

    with assert_raises():
        var all_nan = nm.array[f64](data=[nan[f64](), nan[f64]()], shape=[2])
        _ = nm.nanmedian(all_nan)


def test_nanargmax_nanargmin() raises:
    var np = Python.import_module("numpy")
    var A = _make_nan_array()
    var Anp = A.to_numpy()

    check_values_close(
        nm.nanargmax(A), np.nanargmax(Anp), "`nanargmax` is broken"
    )
    check_values_close(
        nm.nanargmin(A), np.nanargmin(Anp), "`nanargmin` is broken"
    )
    for axis in range(2):
        check_is_close(
            nm.nanargmax(A, axis=axis),
            np.nanargmax(Anp, axis=axis),
            String("`nanargmax` is broken for axis {}").format(axis),
        )
        check_is_close(
            nm.nanargmin(A, axis=axis),
            np.nanargmin(Anp, axis=axis),
            String("`nanargmin` is broken for axis {}").format(axis),
        )

    with assert_raises():
        var all_nan = nm.array[f64](data=[nan[f64](), nan[f64]()], shape=[2])
        _ = nm.nanargmax(all_nan)

    with assert_raises():
        var all_nan = nm.array[f64](data=[nan[f64](), nan[f64]()], shape=[2])
        _ = nm.nanargmin(all_nan)


def test_nanprod() raises:
    var np = Python.import_module("numpy")
    var A = _make_nan_array()
    var Anp = A.to_numpy()

    assert_true(
        np.isclose(nm.nanprod(A), np.nanprod(Anp), atol=PythonObject(0.001)),
        "`nanprod` is broken",
    )
    for axis in range(2):
        check_is_close(
            nm.nanprod(A, axis=axis),
            np.nanprod(Anp, axis=axis),
            String("`nanprod` is broken for axis {}").format(axis),
        )


def test_nancumsum_nancumprod() raises:
    var np = Python.import_module("numpy")
    var A = _make_nan_array()
    var Anp = A.to_numpy()

    check_is_close(nm.nancumsum(A), np.nancumsum(Anp), "`nancumsum` is broken")
    check_is_close(
        nm.nancumprod(A), np.nancumprod(Anp), "`nancumprod` is broken"
    )
    for axis in range(2):
        check_is_close(
            nm.nancumsum(A, axis=axis),
            np.nancumsum(Anp, axis=axis),
            String("`nancumsum` is broken for axis {}").format(axis),
        )
        check_is_close(
            nm.nancumprod(A, axis=axis),
            np.nancumprod(Anp, axis=axis),
            String("`nancumprod` is broken for axis {}").format(axis),
        )


def test_nanquantile_nanpercentile() raises:
    var np = Python.import_module("numpy")
    var A = _make_nan_array()
    var Anp = A.to_numpy()

    assert_true(
        np.isclose(
            nm.nanquantile(A, Float64(0.5)),
            np.nanquantile(Anp, 0.5),
            atol=PythonObject(0.001),
        ),
        "`nanquantile` is broken",
    )
    assert_true(
        np.isclose(
            nm.nanpercentile(A, Float64(50)),
            np.nanpercentile(Anp, 50),
            atol=PythonObject(0.001),
        ),
        "`nanpercentile` is broken",
    )
    for axis in range(2):
        check_is_close(
            nm.nanquantile(A, Float64(0.5), axis=axis),
            np.nanquantile(Anp, 0.5, axis=axis),
            String("`nanquantile` is broken for axis {}").format(axis),
        )
        check_is_close(
            nm.nanpercentile(A, Float64(50), axis=axis),
            np.nanpercentile(Anp, 50, axis=axis),
            String("`nanpercentile` is broken for axis {}").format(axis),
        )

    with assert_raises():
        var all_nan = nm.array[f64](data=[nan[f64](), nan[f64]()], shape=[2])
        _ = nm.nanquantile(all_nan, Float64(0.5))


def test_nan_to_num() raises:
    var np = Python.import_module("numpy")
    var A = nm.array[f64](
        data=[Scalar[f64](1.0), nan[f64](), inf[f64](), neg_inf[f64]()],
        shape=[4],
    )
    var Anp = A.to_numpy()

    check_is_close(
        nm.nan_to_num(A), np.nan_to_num(Anp), "`nan_to_num` is broken"
    )
    check_is_close(
        nm.nan_to_num(A, nan=Float64(-1.0)),
        np.nan_to_num(Anp, nan=-1.0),
        "`nan_to_num` with `nan` override is broken",
    )
    check_is_close(
        nm.nan_to_num(A, posinf=Float64(100.0), neginf=Float64(-100.0)),
        np.nan_to_num(Anp, posinf=100.0, neginf=-100.0),
        "`nan_to_num` with `posinf`/`neginf` override is broken",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
