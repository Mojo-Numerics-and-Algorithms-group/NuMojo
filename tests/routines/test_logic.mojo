from std.python import Python, PythonObject
from std.testing.testing import assert_raises, assert_true, assert_equal
from std.testing import TestSuite
from utils_for_test import check

import numojo as nm
from numojo.prelude import *


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

    var a_allclose = nm.allclose(a, b, rtol=1e-5, atol=1e-8)
    var np_allclose = np.allclose(
        a.to_numpy(),
        b.to_numpy(),
        rtol=PythonObject(1e-5),
        atol=PythonObject(1e-8),
    )
    if a_allclose != Bool(py=np_allclose):
        print("Error: allclose results differ")

    check(
        nm.isclose(a, b, rtol=1e-5, atol=1e-8),
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

    assert_true(nm.array_equal(same1, same2), "array_equal true")
    assert_true(not nm.array_equal(same1, diff), "array_equal false")


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
    check(nm.isposinf(a), np.isposinf(p_arr), "isposinf")
    check(nm.isneginf(a), np.isneginf(p_arr), "isneginf")

    # The scalar overloads must agree with the array results checked above,
    # element by element, and so with NumPy.
    for i in range(a.size):
        var x = a.item(i)
        assert_equal(
            Bool(nm.isinf(x)), Bool(np.isinf(p_arr[i])), "isinf(scalar)"
        )
        assert_equal(
            Bool(nm.isfinite(x)),
            Bool(np.isfinite(p_arr[i])),
            "isfinite(scalar)",
        )
        assert_equal(
            Bool(nm.isnan(x)), Bool(np.isnan(p_arr[i])), "isnan(scalar)"
        )
        assert_equal(
            Bool(nm.isposinf(x)),
            Bool(np.isposinf(p_arr[i])),
            "isposinf(scalar)",
        )
        assert_equal(
            Bool(nm.isneginf(x)),
            Bool(np.isneginf(p_arr[i])),
            "isneginf(scalar)",
        )


def test_logical_operations() raises:
    var np = Python.import_module("numpy")
    var a = nm.array[nm.i32]("[[0, 1, 2], [0, 3, 4]]")
    var b = nm.array[nm.i32]("[[0, 0, 2], [5, 0, 4]]")
    var a_np = a.to_numpy()
    var b_np = b.to_numpy()

    check(nm.logical_and(a, b), np.logical_and(a_np, b_np), "logical_and")
    check(nm.logical_or(a, b), np.logical_or(a_np, b_np), "logical_or")
    check(nm.logical_xor(a, b), np.logical_xor(a_np, b_np), "logical_xor")
    check(nm.logical_not(a), np.logical_not(a_np), "logical_not")

    # Exercise both the full SIMD-width path and scalar tail with signed
    # integral values. Logical NOT must test equality with zero rather than
    # applying a bitwise complement.
    var signed = nm.arange[nm.i32](-19, 22)
    check(
        nm.logical_not(signed),
        np.logical_not(signed.to_numpy()),
        "logical_not signed SIMD and tail",
    )


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
