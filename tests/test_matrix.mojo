import numojo as nm
from numojo.prelude import *
from time import now
from python import Python, PythonObject
from testing.testing import assert_raises, assert_true

# ===-----------------------------------------------------------------------===#
# Main functions
# ===-----------------------------------------------------------------------===#


fn check[
    dtype: DType
](matrix: nm.mat.Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(matrix.to_numpy(), np_sol)), st)


fn check_is_close[
    dtype: DType
](matrix: nm.mat.Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.isclose(matrix.to_numpy(), np_sol, atol=0.1)), st)


# ===-----------------------------------------------------------------------===#
# Creation
# ===-----------------------------------------------------------------------===#


def test_full():
    var np = Python.import_module("numpy")
    check(
        nm.mat.full[f64]((10, 10), 10),
        np.full((10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


# ===-----------------------------------------------------------------------===#
# Arithmetic
# ===-----------------------------------------------------------------------===#


def test_arithmetic():
    var np = Python.import_module("numpy")
    var A = nm.mat.rand[f64]((100, 100))
    var B = nm.mat.rand[f64]((100, 100))
    var Ap = A.to_numpy()
    var Bp = B.to_numpy()
    check_is_close(A + B, Ap + Bp, "Add is broken")
    check_is_close(A - B, Ap - Bp, "Sub is broken")
    check_is_close(A * B, Ap * Bp, "Mul is broken")
    check_is_close(A @ B, np.matmul(Ap, Bp), "Matmul is broken")


def test_logic():
    var np = Python.import_module("numpy")
    var A = nm.mat.ones((5, 1))
    var B = nm.mat.ones((5, 1))
    var Ap = A.to_numpy()
    var Bp = B.to_numpy()
    check(A > B, Ap > Bp, "gt is broken")
    check(A < B, Ap < Bp, "lt is broken")


# ===-----------------------------------------------------------------------===#
# Linear algebra
# ===-----------------------------------------------------------------------===#


def test_transpose():
    var np = Python.import_module("numpy")
    var a = nm.mat.rand[f64]((100, 100))
    var ap = a.to_numpy()
    check_is_close(
        a.transpose(),
        ap.transpose(),
        "Transpose is broken",
    )


def test_solve():
    var np = Python.import_module("numpy")
    var a1 = nm.mat.rand[f64]((100, 100))
    var a2 = nm.mat.rand[f64]((100, 100))
    var y = nm.mat.rand((100, 1))
    var a1p = a1.to_numpy()
    var a2p = a2.to_numpy()
    var yp = y.to_numpy()
    check_is_close(
        nm.mat.solve(a1, a2),
        np.linalg.solve(a1p, a2p),
        "Solve is broken",
    )
    check_is_close(
        nm.mat.inv(a1),
        np.linalg.inv(a1p),
        "Inverse is broken",
    )
    check_is_close(
        nm.mat.lstsq(a1, y),
        np.linalg.lstsq(a1p, yp)[0],
        "Least square is broken",
    )
