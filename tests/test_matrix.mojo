import numojo as nm
from numojo import mat
from numojo.prelude import *
from time import now
from python import Python, PythonObject
from testing.testing import assert_raises, assert_true

# ===-----------------------------------------------------------------------===#
# Main functions
# ===-----------------------------------------------------------------------===#


fn check_matrices_equal[
    dtype: DType
](matrix: mat.Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(np.matrix(matrix.to_numpy()), np_sol)), st)


fn check_matrices_close[
    dtype: DType
](matrix: mat.Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(
        np.all(np.isclose(np.matrix(matrix.to_numpy()), np_sol, atol=0.01)), st
    )


fn check_values_close[
    dtype: DType
](value: Scalar[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.isclose(value, np_sol, atol=0.01), st)


# ===-----------------------------------------------------------------------===#
# Manipulation
# ===-----------------------------------------------------------------------===#


def test_manipulation():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((10, 10)) * 1000
    var Anp = np.matrix(A.to_numpy())
    check_matrices_equal(
        A.astype[nm.i32](),
        Anp.astype(np.int32),
        "`astype` is broken",
    )

    check_matrices_equal(
        A.reshape((50, 2)),
        Anp.reshape((50, 2)),
        "Reshape is broken",
    )

    _ = A.resize((1000, 100))
    _ = Anp.resize((1000, 100))
    check_matrices_equal(
        A,
        Anp,
        "Resize is broken",
    )


# ===-----------------------------------------------------------------------===#
# Creation
# ===-----------------------------------------------------------------------===#


def test_full():
    var np = Python.import_module("numpy")
    check_matrices_equal(
        mat.full[f64]((10, 10), 10),
        np.full((10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


def test_zeros():
    var np = Python.import_module("numpy")
    check_matrices_equal(
        mat.zeros[f64](shape=(10, 10)),
        np.zeros((10, 10), dtype=np.float64),
        "Zeros is broken",
    )


# ===-----------------------------------------------------------------------===#
# Arithmetic
# ===-----------------------------------------------------------------------===#


def test_arithmetic():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((10, 10))
    var B = mat.rand[f64]((10, 10))
    var C = mat.rand[f64]((10, 1))
    var Ap = A.to_numpy()
    var Bp = B.to_numpy()
    var Cp = C.to_numpy()
    check_matrices_close(A + B, Ap + Bp, "Add is broken")
    check_matrices_close(A - B, Ap - Bp, "Sub is broken")
    check_matrices_close(A * B, Ap * Bp, "Mul is broken")
    check_matrices_close(A @ B, np.matmul(Ap, Bp), "Matmul is broken")
    check_matrices_close(A + C, Ap + Cp, "Add (broadcast) is broken")
    check_matrices_close(A - C, Ap - Cp, "Sub (broadcast) is broken")
    check_matrices_close(A * C, Ap * Cp, "Mul (broadcast) is broken")
    check_matrices_close(A / C, Ap / Cp, "Div (broadcast) is broken")
    check_matrices_close(A + 1, Ap + 1, "Add (to int) is broken")
    check_matrices_close(A - 1, Ap - 1, "Sub (to int) is broken")
    check_matrices_close(A * 1, Ap * 1, "Mul (to int) is broken")
    check_matrices_close(A / 1, Ap / 1, "Div (to int) is broken")
    check_matrices_close(A**2, np.power(Ap, 2), "Pow (to int) is broken")
    check_matrices_close(A**0.5, np.power(Ap, 0.5), "Pow (to int) is broken")


def test_logic():
    var np = Python.import_module("numpy")
    var A = mat.ones((5, 1))
    var B = mat.ones((5, 1))
    var L = mat.fromstring[i8](
        "[[0,0,0],[0,0,1],[1,1,1],[1,0,0]]", shape=(4, 3)
    )
    var Anp = np.matrix(A.to_numpy())
    var Bnp = np.matrix(B.to_numpy())
    var Lnp = np.matrix(L.to_numpy())

    check_matrices_equal(A > B, Anp > Bnp, "gt is broken")
    check_matrices_equal(A < B, Anp < Bnp, "lt is broken")
    assert_true(
        np.equal(mat.all(L), np.all(Lnp)),
        "`all` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.all(L, axis=i),
            np.all(Lnp, axis=i),
            String("`all` by axis {i} is broken"),
        )
    assert_true(
        np.equal(mat.any(L), np.any(Lnp)),
        "`any` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.any(L, axis=i),
            np.any(Lnp, axis=i),
            String("`any` by axis {i} is broken"),
        )


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
    check_matrices_close(
        mat.solve(A, B),
        np.linalg.solve(Anp, Bnp),
        "Solve is broken",
    )
    check_matrices_close(
        mat.inv(A),
        np.linalg.inv(Anp),
        "Inverse is broken",
    )
    check_matrices_close(
        mat.lstsq(A, Y),
        np.linalg.lstsq(Anp, Ynp)[0],
        "Least square is broken",
    )
    check_matrices_close(
        A.transpose(),
        Anp.transpose(),
        "Transpose is broken",
    )
    check_matrices_close(
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


# ===-----------------------------------------------------------------------===#
# Mathematics
# ===-----------------------------------------------------------------------===#


def test_math():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((100, 100))
    var Anp = np.matrix(A.to_numpy())
    assert_true(
        np.all(np.isclose(mat.sum(A), np.sum(Anp), atol=0.1)),
        "`sum` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.sum(A, axis=i),
            np.sum(Anp, axis=i),
            String("`sum` by axis {i} is broken"),
        )
    assert_true(
        np.all(np.isclose(mat.prod(A), np.prod(Anp), atol=0.1)),
        "`prod` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.prod(A, axis=i),
            np.prod(Anp, axis=i),
            String("`prod` by axis {i} is broken"),
        )


def test_trigonometric():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((100, 100))
    var Anp = np.matrix(A.to_numpy())
    check_matrices_close(mat.sin(A), np.sin(Anp), "sin is broken")
    check_matrices_close(mat.cos(A), np.cos(Anp), "cos is broken")
    check_matrices_close(mat.tan(A), np.tan(Anp), "tan is broken")
    check_matrices_close(mat.arcsin(A), np.arcsin(Anp), "arcsin is broken")
    check_matrices_close(mat.asin(A), np.arcsin(Anp), "asin is broken")
    check_matrices_close(mat.arccos(A), np.arccos(Anp), "arccos is broken")
    check_matrices_close(mat.acos(A), np.arccos(Anp), "acos is broken")
    check_matrices_close(mat.arctan(A), np.arctan(Anp), "arctan is broken")
    check_matrices_close(mat.atan(A), np.arctan(Anp), "atan is broken")


def test_hyperbolic():
    var np = Python.import_module("numpy")
    var A = mat.fromstring("[[1,2,3],[4,5,6],[7,8,9]]", shape=(3, 3))
    var B = A / 10
    var Anp = np.matrix(A.to_numpy())
    var Bnp = np.matrix(B.to_numpy())
    check_matrices_close(mat.sinh(A), np.sinh(Anp), "sinh is broken")
    check_matrices_close(mat.cosh(A), np.cosh(Anp), "cosh is broken")
    check_matrices_close(mat.tanh(A), np.tanh(Anp), "tanh is broken")
    check_matrices_close(mat.arcsinh(A), np.arcsinh(Anp), "arcsinh is broken")
    check_matrices_close(mat.asinh(A), np.arcsinh(Anp), "asinh is broken")
    check_matrices_close(mat.arccosh(A), np.arccosh(Anp), "arccosh is broken")
    check_matrices_close(mat.acosh(A), np.arccosh(Anp), "acosh is broken")
    check_matrices_close(mat.arctanh(B), np.arctanh(Bnp), "arctanh is broken")
    check_matrices_close(mat.atanh(B), np.arctanh(Bnp), "atanh is broken")


# ===-----------------------------------------------------------------------===#
# Statistics
# ===-----------------------------------------------------------------------===#


def test_statistics():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((100, 100))
    var Anp = np.matrix(A.to_numpy())

    assert_true(
        np.all(np.isclose(mat.mean(A), np.mean(Anp), atol=0.1)),
        "`mean` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.mean(A, i),
            np.mean(Anp, i),
            String("`mean` is broken for {i}-dimension"),
        )

    assert_true(
        np.all(np.isclose(mat.variance(A), np.`var`(Anp), atol=0.1)),
        "`variance` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.variance(A, i),
            np.`var`(Anp, i),
            String("`variance` is broken for {i}-dimension"),
        )

    assert_true(
        np.all(np.isclose(mat.std(A), np.std(Anp), atol=0.1)),
        "`std` is broken",
    )
    for i in range(2):
        check_matrices_close(
            mat.std(A, i),
            np.std(Anp, i),
            String("`std` is broken for {i}-dimension"),
        )


def test_sorting():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((10, 10))
    var Anp = np.matrix(A.to_numpy())

    check_matrices_close(
        mat.sort(A), np.sort(Anp, axis=None), String("Sort is broken")
    )
    for i in range(2):
        check_matrices_close(
            mat.sort(A, axis=i),
            np.sort(Anp, axis=i),
            String("Sort by axis {} is broken").format(i),
        )

    check_matrices_close(
        mat.argsort(A),
        np.argsort(Anp, axis=None),
        String("Argsort is broken")
        + str(mat.argsort(A))
        + str(np.argsort(Anp, axis=None)),
    )
    for i in range(2):
        check_matrices_close(
            mat.argsort(A, axis=i),
            np.argsort(Anp, axis=i),
            String("Argsort by axis {} is broken").format(i),
        )


def test_searching():
    var np = Python.import_module("numpy")
    var A = mat.rand[f64]((10, 10))
    var Anp = np.matrix(A.to_numpy())

    check_values_close(
        mat.max(A), np.max(Anp, axis=None), String("`max` is broken")
    )
    for i in range(2):
        check_matrices_close(
            mat.max(A, axis=i),
            np.max(Anp, axis=i),
            String("`max` by axis {} is broken").format(i),
        )

    check_values_close(
        mat.argmax(A), np.argmax(Anp, axis=None), String("`argmax` is broken")
    )
    for i in range(2):
        check_matrices_close(
            mat.argmax(A, axis=i),
            np.argmax(Anp, axis=i),
            String("`argmax` by axis {} is broken").format(i),
        )

    check_values_close(
        mat.min(A), np.min(Anp, axis=None), String("`min` is broken.")
    )
    for i in range(2):
        check_matrices_close(
            mat.min(A, axis=i),
            np.min(Anp, axis=i),
            String("`min` by axis {} is broken").format(i),
        )
