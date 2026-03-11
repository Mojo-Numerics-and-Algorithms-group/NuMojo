import numojo as nm
from numojo.prelude import *
from std.python import Python, PythonObject
from std.testing.testing import assert_raises, assert_true, assert_equal
from std.testing import TestSuite
from utils_for_test import check
from numojo.routines.logic.comparison import allclose, isclose, array_equal
from numojo.routines.logic.contents import isposinf, isneginf


def test_comparison_array_array() raises:
    var np = Python.import_module("numpy")

    var a = nm.array[nm.f64]("[[-1.5, 0.0, 2.0], [3.5, -4.0, 8.0]]")
    var b = nm.array[nm.f64]("[[-2.0, 0.0, 1.5], [3.5, -3.0, 9.0]]")

    var anp = a.to_numpy()
    var bnp = b.to_numpy()

    check(nm.greater(a, b), np.greater(anp, bnp), "greater(array, array)")
    check(
        nm.greater_equal(a, b),
        np.greater_equal(anp, bnp),
        "greater_equal(array, array)",
    )
    check(nm.less(a, b), np.less(anp, bnp), "less(array, array)")
    check(
        nm.less_equal(a, b),
        np.less_equal(anp, bnp),
        "less_equal(array, array)",
    )
    check(nm.equal(a, b), np.equal(anp, bnp), "equal(array, array)")
    check(
        nm.not_equal(a, b),
        np.not_equal(anp, bnp),
        "not_equal(array, array)",
    )


def test_comparison_array_scalar() raises:
    var np = Python.import_module("numpy")

    var a = nm.array[nm.f64]("[[-2.0, -0.0, 1.0], [2.0, 3.0, -5.0]]")
    var anp = a.to_numpy()
    var s: SIMD[nm.f64, 1] = 1.0

    check(nm.greater(a, s), np.greater(anp, 1.0), "greater(array, scalar)")
    check(
        nm.greater_equal(a, s),
        np.greater_equal(anp, 1.0),
        "greater_equal(array, scalar)",
    )
    check(nm.less(a, s), np.less(anp, 1.0), "less(array, scalar)")
    check(
        nm.less_equal(a, s),
        np.less_equal(anp, 1.0),
        "less_equal(array, scalar)",
    )
    check(nm.equal(a, s), np.equal(anp, 1.0), "equal(array, scalar)")
    check(
        nm.not_equal(a, s),
        np.not_equal(anp, 1.0),
        "not_equal(array, scalar)",
    )


def test_allclose_and_isclose_and_array_equal() raises:
    var np = Python.import_module("numpy")

    var a = nm.array[nm.f64]("[1.0, 2.0, 3.0, 4.0]")
    var b = nm.array[nm.f64]("[1.0, 2.000001, 2.999999, 4.0]")

    var a_allclose = allclose(a, b, rtol=1e-5, atol=1e-8)
    var np_allclose = np.allclose(
        a.to_numpy(),
        b.to_numpy(),
        rtol=PythonObject(1e-5),
        atol=PythonObject(1e-8),
    )
    if a_allclose != Bool(py=np_allclose):
        print("Error: allclose results differ")

    check(
        isclose(a, b, rtol=1e-5, atol=1e-8),
        np.isclose(
            a.to_numpy(),
            b.to_numpy(),
            rtol=PythonObject(1e-5),
            atol=PythonObject(1e-8),
        ),
        "isclose basic",
    )

    var same1 = nm.arange[nm.i32](0, 10)
    var same2 = nm.arange[nm.i32](0, 10)
    var diff = nm.arange[nm.i32](1, 11)

    assert_true(array_equal(same1, same2), "array_equal true")
    assert_true(not array_equal(same1, diff), "array_equal false")


def test_contents_isinf_isfinite_isnan() raises:
    var np = Python.import_module("numpy")

    var nan64 = Python.float("nan").__float__()
    var inf64 = Python.float("inf").__float__()
    var ninf64 = Python.float("-inf").__float__()

    var p_nan = nan64
    var p_inf = inf64
    var p_ninf = ninf64

    var data = Python.list(1.0, p_inf, p_ninf, p_nan, -3.25, 0.0)
    var p_arr = np.array(data, dtype=np.float64)

    var a = nm.array[nm.f64](p_arr)

    check(nm.isinf(a), np.isinf(p_arr), "isinf")
    check(nm.isfinite(a), np.isfinite(p_arr), "isfinite")
    check(nm.isnan(a), np.isnan(p_arr), "isnan")
    check(isposinf(a), np.isposinf(p_arr), "isposinf")
    check(isneginf(a), np.isneginf(p_arr), "isneginf")


def test_truth_all_any() raises:
    var np = Python.import_module("numpy")

    var p1 = np.array(Python.list(True, True, True, False), dtype=np.bool_)
    var p2 = np.array(Python.list(False, False, False, False), dtype=np.bool_)
    var p3 = np.array(Python.list(True, True, True, True), dtype=np.bool_)

    var a1 = nm.array[nm.boolean](p1)
    var a2 = nm.array[nm.boolean](p2)
    var a3 = nm.array[nm.boolean](p3)

    assert_true(
        Bool(nm.all(a1)) == Bool(np.all(p1)),
        "all with one false",
    )
    assert_true(
        Bool(nm.any(a1)) == Bool(np.any(p1)),
        "any with one false",
    )

    assert_true(
        Bool(nm.all(a2)) == Bool(np.all(p2)),
        "all all-false",
    )
    assert_true(
        Bool(nm.any(a2)) == Bool(np.any(p2)),
        "any all-false",
    )

    assert_true(
        Bool(nm.all(a3)) == Bool(np.all(p3)),
        "all all-true",
    )
    assert_true(
        Bool(nm.any(a3)) == Bool(np.any(p3)),
        "any all-true",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
