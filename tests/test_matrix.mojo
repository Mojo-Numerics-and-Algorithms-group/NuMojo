import numojo as nm
from numojo import mat
from numojo.prelude import *
from time import now
from python import Python, PythonObject
from testing.testing import assert_raises, assert_true

# ===-----------------------------------------------------------------------===#
# Main functions
# ===-----------------------------------------------------------------------===#


fn check[
    dtype: DType
](matrix: mat.Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(np.matrix(matrix.to_numpy()), np_sol)), st)


fn check_is_close[
    dtype: DType
](matrix: mat.Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(
        np.all(np.isclose(np.matrix(matrix.to_numpy()), np_sol, atol=0.1)), st
    )


# ===-----------------------------------------------------------------------===#
# Creation
# ===-----------------------------------------------------------------------===#


def test_full():
    var np = Python.import_module("numpy")
    check(
        mat.full[f64]((10, 10), 10),
        np.full((10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


# ===-----------------------------------------------------------------------===#
# Arithmetic
# ===-----------------------------------------------------------------------===#


def test_arithmetic():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((100, 100))
    var B = mat.rand[f64]((100, 100))
    var C = mat.rand[f64]((100, 1))
    var Ap = A.to_numpy()
    var Bp = B.to_numpy()
    var Cp = C.to_numpy()
    check_is_close(A + B, Ap + Bp, "Add is broken")
    check_is_close(A - B, Ap - Bp, "Sub is broken")
    check_is_close(A * B, Ap * Bp, "Mul is broken")
    check_is_close(A @ B, np.matmul(Ap, Bp), "Matmul is broken")
    check_is_close(A + C, Ap + Cp, "Add (broadcast) is broken")
    check_is_close(A - C, Ap - Cp, "Sub (broadcast) is broken")
    check_is_close(A * C, Ap * Cp, "Mul (broadcast) is broken")
    check_is_close(A / C, Ap / Cp, "Div (broadcast) is broken")
    check_is_close(A + 1, Ap + 1, "Add (to int) is broken")
    check_is_close(A - 1, Ap - 1, "Sub (to int) is broken")
    check_is_close(A * 1, Ap * 1, "Mul (to int) is broken")
    check_is_close(A / 1, Ap / 1, "Div (to int) is broken")


def test_logic():
    var np = Python.import_module("numpy")
    var A = mat.ones((5, 1))
    var B = mat.ones((5, 1))
    var Ap = A.to_numpy()
    var Bp = B.to_numpy()
    check(A > B, Ap > Bp, "gt is broken")
    check(A < B, Ap < Bp, "lt is broken")


# ===-----------------------------------------------------------------------===#
# Linear algebra
# ===-----------------------------------------------------------------------===#


def test_linalg():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((100, 100))
    var B = mat.rand[f64]((100, 100))
    var E = mat.fromstring("[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]", shape=(4, 3))
    var Y = mat.rand((100, 1))
    var Anp = A.to_numpy()
    var Bnp = B.to_numpy()
    var Ynp = Y.to_numpy()
    var Enp = E.to_numpy()
    check_is_close(
        mat.solve(A, B),
        np.linalg.solve(Anp, Bnp),
        "Solve is broken",
    )
    check_is_close(
        mat.inv(A),
        np.linalg.inv(Anp),
        "Inverse is broken",
    )
    check_is_close(
        mat.lstsq(A, Y),
        np.linalg.lstsq(Anp, Ynp)[0],
        "Least square is broken",
    )
    check_is_close(
        A.transpose(),
        Anp.transpose(),
        "Transpose is broken",
    )
    check_is_close(
        Y.transpose(),
        Ynp.transpose(),
        "Transpose is broken",
    )
    assert_true(
        np.all(np.isclose(mat.det(A), np.linalg.det(Anp), atol=0.1)),
        "Determinant is broken",
    )
    for i in range(-10, 10):
        assert_true(
            np.all(
                np.isclose(
                    mat.trace(E, offset=i), np.trace(Enp, offset=i), atol=0.1
                )
            ),
            "Trace is broken",
        )


def test_math():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((100, 100))
    var B = mat.rand[f64]((100, 100))
    var D = mat.rand(shape=(10000, 100))
    var E = mat.fromstring("[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]", shape=(4, 3))
    var Y = mat.rand((100, 1))
    var Anp = np.matrix(A.to_numpy())
    var Bnp = np.matrix(B.to_numpy())
    var Dnp = np.matrix(D.to_numpy())
    var Enp = np.matrix(E.to_numpy())
    var Ynp = np.matrix(Y.to_numpy())
    assert_true(
        np.all(np.isclose(mat.sum(D), np.sum(Dnp), atol=0.1)),
        "Sum is broken",
    )
    check_is_close(
        mat.sum(D, axis=0),
        np.sum(Dnp, axis=0),
        "Sum by axis 0 is broken",
    )
    check_is_close(
        mat.sum(D, axis=1),
        np.sum(Dnp, axis=1),
        "Sum by axis 1 is broken",
    )


def test_statistics():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((100, 100))
    var Anp = np.matrix(A.to_numpy())
    assert_true(
        np.all(np.isclose(mat.mean(A), np.mean(Anp), atol=0.1)),
        "Mean is broken",
    )
    for i in range(2):
        check_is_close(
            mat.sum(A, axis=i),
            np.sum(Anp, axis=i),
            String("Sum by axis {} is broken").format(i),
        )
