"""
Backend (HostExecutor) regression tests.

`HostExecutor` picks between a serial path and a chunked-parallel path
(`parallelize`) based on array size (see `_num_tasks_for` in
`numojo/routines/operations/backend.mojo`).

Every test below is run at both
a SMALL size (forces the serial fallback) and a LARGE size (forces the
parallel path) so a regression in either path is caught.
"""

import numojo as nm
from numojo.prelude import *
from numojo.routines.math.arithmetic import fma
from numojo.routines.logic.contents import isnan
from std.python import Python, PythonObject
from std.testing.testing import assert_true
from std.testing import TestSuite

comptime SMALL_N = 5
comptime LARGE_N = 200_000


def check_close[
    dtype: DType
](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(
        np.all(np.isclose(array.to_numpy(), np_sol, atol=PythonObject(1e-6))),
        st,
    )


def check_equal[
    dtype: DType
](array: nm.NDArray[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(array.to_numpy(), np_sol)), st)


def test_ndarray_multiplication_commutes() raises:
    var A = nm.ones[nm.f64](nm.Shape(2, 2))
    var B = nm.ones[nm.f64](nm.Shape(2, 2))
    var L = 2.0 * (A @ B)
    var R = (A @ B) * 2.0
    assert_true((L == R).all())


# ===-----------------------------------------------------------------------===#
# apply_unary: NDArray -> NDArray  (e.g. exp, sin, sqrt)
# ===-----------------------------------------------------------------------===#


def test_apply_unary_small() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, SMALL_N + 1)
    var anp = a.to_numpy()
    check_close(nm.exp(a), np.exp(anp), "apply_unary (serial): exp mismatch")


def test_apply_unary_large() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, LARGE_N + 1)
    var anp = a.to_numpy()
    check_close(nm.exp(a), np.exp(anp), "apply_unary (parallel): exp mismatch")


# ===-----------------------------------------------------------------------===#
# apply_binary: NDArray, NDArray -> NDArray  (e.g. add, mul)
# ===-----------------------------------------------------------------------===#


def test_apply_binary_array_array_small() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, SMALL_N + 1)
    var b = nm.arange[nm.f64](1, SMALL_N + 1)
    var anp = a.to_numpy()
    var bnp = b.to_numpy()
    check_close(a + b, anp + bnp, "apply_binary (serial): add mismatch")
    check_close(a * b, anp * bnp, "apply_binary (serial): mul mismatch")


def test_apply_binary_array_array_large() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, LARGE_N + 1)
    var b = nm.arange[nm.f64](1, LARGE_N + 1)
    var anp = a.to_numpy()
    var bnp = b.to_numpy()
    check_close(a + b, anp + bnp, "apply_binary (parallel): add mismatch")
    check_close(a * b, anp * bnp, "apply_binary (parallel): mul mismatch")


# ===-----------------------------------------------------------------------===#
# apply_binary: NDArray, Scalar -> NDArray  (and Scalar, NDArray -> NDArray)
# ===-----------------------------------------------------------------------===#


def test_apply_binary_array_scalar_small() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, SMALL_N + 1)
    var anp = a.to_numpy()
    check_close(
        a + 5.0, anp + 5.0, "apply_binary (serial): array+scalar mismatch"
    )
    check_close(
        5.0 + a, 5.0 + anp, "apply_binary (serial): scalar+array mismatch"
    )


def test_apply_binary_array_scalar_large() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, LARGE_N + 1)
    var anp = a.to_numpy()
    check_close(
        a + 5.0, anp + 5.0, "apply_binary (parallel): array+scalar mismatch"
    )
    check_close(
        5.0 + a, 5.0 + anp, "apply_binary (parallel): scalar+array mismatch"
    )


# ===-----------------------------------------------------------------------===#
# apply_binary_predicate: NDArray, NDArray -> NDArray[bool]
# ===-----------------------------------------------------------------------===#


def test_apply_binary_predicate_array_array_small() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, SMALL_N + 1)
    var b = nm.arange[nm.f64](SMALL_N, 0, -1)
    check_equal(
        a > b,
        a.to_numpy() > b.to_numpy(),
        "apply_binary_predicate (serial): array>array mismatch",
    )


def test_apply_binary_predicate_array_array_large() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, LARGE_N + 1)
    var b = nm.arange[nm.f64](LARGE_N, 0, -1)
    check_equal(
        a > b,
        a.to_numpy() > b.to_numpy(),
        "apply_binary_predicate (parallel): array>array mismatch",
    )


# ===-----------------------------------------------------------------------===#
# apply_binary_predicate: NDArray, Scalar -> NDArray[bool]
# ===-----------------------------------------------------------------------===#


def test_apply_binary_predicate_array_scalar_small() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, SMALL_N + 1)
    var anp = a.to_numpy()
    check_equal(
        a > 2.0,
        anp > 2.0,
        "apply_binary_predicate (serial): array>scalar mismatch",
    )


def test_apply_binary_predicate_array_scalar_large() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, LARGE_N + 1)
    var anp = a.to_numpy()
    check_equal(
        a > 2.0,
        anp > 2.0,
        "apply_binary_predicate (parallel): array>scalar mismatch",
    )


# ===-----------------------------------------------------------------------===#
# apply_unary_predicate: NDArray -> NDArray[bool]  (e.g. isnan)
# ===-----------------------------------------------------------------------===#


def test_apply_unary_predicate_small() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, SMALL_N + 1)
    var anp = a.to_numpy()
    check_equal(
        isnan(a),
        np.isnan(anp),
        "apply_unary_predicate (serial): isnan mismatch",
    )


def test_apply_unary_predicate_large() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, LARGE_N + 1)
    var anp = a.to_numpy()
    check_equal(
        isnan(a),
        np.isnan(anp),
        "apply_unary_predicate (parallel): isnan mismatch",
    )


# ===-----------------------------------------------------------------------===#
# apply_ternary: NDArray, NDArray, NDArray -> NDArray  (fma)
# and NDArray, NDArray, Scalar -> NDArray
# ===-----------------------------------------------------------------------===#


def test_apply_ternary_array_array_array_small() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, SMALL_N + 1)
    var b = nm.arange[nm.f64](1, SMALL_N + 1)
    var anp = a.to_numpy()
    var bnp = b.to_numpy()
    check_close(
        fma(a, b, a),
        anp * bnp + anp,
        "apply_ternary (serial): fma(a,b,a) mismatch",
    )


def test_apply_ternary_array_array_array_large() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, LARGE_N + 1)
    var b = nm.arange[nm.f64](1, LARGE_N + 1)
    var anp = a.to_numpy()
    var bnp = b.to_numpy()
    check_close(
        fma(a, b, a),
        anp * bnp + anp,
        "apply_ternary (parallel): fma(a,b,a) mismatch",
    )


def test_apply_ternary_array_array_scalar_small() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, SMALL_N + 1)
    var b = nm.arange[nm.f64](1, SMALL_N + 1)
    var anp = a.to_numpy()
    var bnp = b.to_numpy()
    check_close(
        fma(a, b, 2.0),
        anp * bnp + 2.0,
        "apply_ternary (serial): fma(a,b,2.0) mismatch",
    )


def test_apply_ternary_array_array_scalar_large() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, LARGE_N + 1)
    var b = nm.arange[nm.f64](1, LARGE_N + 1)
    var anp = a.to_numpy()
    var bnp = b.to_numpy()
    check_close(
        fma(a, b, 2.0),
        anp * bnp + 2.0,
        "apply_ternary (parallel): fma(a,b,2.0) mismatch",
    )


# ===-----------------------------------------------------------------------===#
# Automatic broadcasting through the same backend (Phase 1 regression net)
# ===-----------------------------------------------------------------------===#


def test_broadcast_through_backend_small() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, 4).reshape(Shape(3, 1))
    var b = nm.arange[nm.f64](1, 5).reshape(Shape(1, 4))
    check_close(
        a + b,
        a.to_numpy() + b.to_numpy(),
        "broadcast (serial): (3,1)+(1,4) mismatch",
    )


def test_broadcast_through_backend_large() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](1, 1001).reshape(Shape(1000, 1))
    var b = nm.arange[nm.f64](1, 1001).reshape(Shape(1, 1000))
    check_close(
        a + b,
        a.to_numpy() + b.to_numpy(),
        "broadcast (parallel): (1000,1)+(1,1000) mismatch",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
