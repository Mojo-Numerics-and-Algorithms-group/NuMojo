import numojo as nm
from numojo.prelude import *
from python import Python, PythonObject
from utils_for_test import check, check_is_close, check_values_close

# ===-----------------------------------------------------------------------===#
# Sums, products, differences
# ===-----------------------------------------------------------------------===#


def test_sum_prod():
    var np = Python.import_module("numpy")
    var A = nm.random.randn(4, 4, 4)
    var Anp = A.to_numpy()

    check_values_close(
        nm.sum(A),
        np.sum(Anp, axis=None),
        String("`sum` fails."),
    )
    for i in range(3):
        check_is_close(
            nm.sum(A, axis=i),
            np.sum(Anp, axis=i),
            String("`sum` by axis {} fails.".format(i)),
        )

    check_is_close(
        nm.cumsum(A),
        np.cumsum(Anp, axis=None),
        String("`cumsum` fails."),
    )
    for i in range(3):
        check_is_close(
            nm.cumsum(A, axis=i),
            np.cumsum(Anp, axis=i),
            String("`cumsum` by axis {} fails.".format(i)),
        )

    check_values_close(
        nm.prod(A),
        np.prod(Anp, axis=None),
        String("`prod` fails."),
    )
    for i in range(3):
        check_is_close(
            nm.prod(A, axis=i),
            np.prod(Anp, axis=i),
            String("`prod` by axis {} fails.".format(i)),
        )


def test_add_array():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check(nm.add[nm.f64](arr, 5.0), np.arange(0, 15) + 5, "Add array + scalar")
    check(
        nm.add[nm.f64](arr, arr),
        np.arange(0, 15) + np.arange(0, 15),
        "Add array + array",
    )


def test_add_array_par():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 500)

    check(
        nm.add[nm.f64, backend = nm.core._math_funcs.Vectorized](arr, 5.0),
        np.arange(0, 500) + 5,
        "Add array + scalar",
    )
    check(
        nm.add[nm.f64, backend = nm.core._math_funcs.Vectorized](arr, arr),
        np.arange(0, 500) + np.arange(0, 500),
        "Add array + array",
    )


def test_sin():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check_is_close(
        nm.sin[nm.f64](arr), np.sin(np.arange(0, 15)), "Add array + scalar"
    )


def test_sin_par():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check_is_close(
        nm.sin[
            nm.f64,
            backend = nm.core._math_funcs.Vectorized,
        ](arr),
        np.sin(np.arange(0, 15)),
        "Add array + scalar",
    )
