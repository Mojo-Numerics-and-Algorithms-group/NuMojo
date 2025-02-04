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
    var A = nm.random.randn(3, 4, 5)
    var Anp = A.to_numpy()

    assert_true(
        np.all(np.isclose(nm.mean(A), np.mean(Anp), atol=0.1)),
        "`mean` is broken",
    )
    for i in range(3):
        check_is_close(
            nm.mean(A, i),
            np.mean(Anp, i),
            String("`mean` is broken for {}-dimension".format(i)),
        )

    assert_true(
        np.all(np.isclose(nm.median(A), np.median(Anp), atol=0.1)),
        "`median` is broken",
    )

    assert_true(
        np.all(np.isclose(nm.variance(A), np.`var`(Anp), atol=0.1)),
        "`variance` is broken",
    )
    for i in range(3):
        check_is_close(
            nm.variance(A, i),
            np.`var`(Anp, i),
            String("`variance` is broken for {}-dimension".format(i)),
        )

    assert_true(
        np.all(np.isclose(nm.std(A), np.std(Anp), atol=0.1)),
        "`std` is broken",
    )
    for i in range(3):
        check_is_close(
            nm.std(A, i),
            np.std(Anp, i),
            String("`std` is broken for {}-dimension".format(i)),
        )
