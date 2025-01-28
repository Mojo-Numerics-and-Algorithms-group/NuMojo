import numojo as nm
from numojo.prelude import *
from numojo.core.matrix import Matrix
from python import Python, PythonObject
from testing.testing import assert_raises, assert_true
from utils_for_test import check, check_is_close

# ===-----------------------------------------------------------------------===#
# Statistics
# ===-----------------------------------------------------------------------===#


def test_mean():
    var np = Python.import_module("numpy")
    var A = nm.NDArray[f64](Shape(10, 10))
    var Anp = np.matrix(A.to_numpy())

    assert_true(
        np.all(np.isclose(nm.mean(A), np.mean(Anp), atol=0.1)),
        "`mean` is broken",
    )
    for i in range(2):
        check_is_close(
            nm.mean(A, i),
            np.mean(Anp, i),
            String("`mean` is broken for {i}-dimension"),
        )
