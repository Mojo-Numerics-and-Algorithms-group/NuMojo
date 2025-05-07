import numojo as nm
from numojo.prelude import *
from numojo.core.matrix import Matrix
from python import Python, PythonObject
from testing.testing import assert_raises, assert_true
from utils_for_test import check, check_is_close

# ===-----------------------------------------------------------------------===#
# Statistics
# ===-----------------------------------------------------------------------===#


def test_mean_median_var_std():
    var np = Python.import_module("numpy")
    var sp = Python.import_module("scipy")
    var A = nm.random.randn(3, 4, 5)
    var Anp = A.to_numpy()

    assert_true(
        np.all(np.isclose(nm.mean(A), np.mean(Anp), atol=0.001)),
        "`mean` is broken",
    )
    for axis in range(3):
        check_is_close(
            nm.mean(A, axis=axis),
            np.mean(Anp, axis=axis),
            String("`mean` is broken for axis {}").format(axis),
        )

    assert_true(
        np.all(np.isclose(nm.median(A), np.median(Anp), atol=0.001)),
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
                nm.mode(A), sp.stats.mode(Anp, axis=None).mode, atol=0.001
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
        np.all(np.isclose(nm.variance(A), np.`var`(Anp), atol=0.001)),
        "`variance` is broken",
    )
    for axis in range(3):
        check_is_close(
            nm.variance(A, axis),
            np.`var`(Anp, axis),
            String("`variance` is broken for axis {}").format(axis),
        )

    assert_true(
        np.all(np.isclose(nm.std(A), np.std(Anp), atol=0.001)),
        "`std` is broken",
    )
    for axis in range(3):
        check_is_close(
            nm.std(A, axis),
            np.std(Anp, axis),
            String("`std` is broken for axis {}").format(axis),
        )
