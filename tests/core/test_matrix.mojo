import numojo as nm
from numojo.prelude import *
from numojo.core.matrix import Matrix, MatrixBase
from python import Python, PythonObject
from testing.testing import assert_raises, assert_true
from sys import is_defined
from testing import assert_equal, TestSuite

alias order: String = String("F") if is_defined["F_CONTIGUOUS"]() else String(
    "C"
)

# ===-----------------------------------------------------------------------===#
# Main functions
# ===-----------------------------------------------------------------------===#


fn check_matrices_equal[
    dtype: DType
](matrix: Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(np.matrix(matrix.to_numpy()), np_sol)), st)


fn check_matrices_close[
    dtype: DType
](matrix: Matrix[dtype], np_sol: PythonObject, st: String) raises:
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
    var A = Matrix.rand[f64]((10, 10), order=order) * 1000
    var Anp = np.matrix(A.to_numpy())
    check_matrices_equal(
        A.astype[nm.i32](),
        Anp.astype(np.int32),
        "`astype` is broken",
    )

    check_matrices_equal(
        A.reshape((50, 2)),
        Anp.reshape(50, 2),
        "Reshape is broken",
    )

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
        Matrix.full[f64]((10, 10), 10, order=order),
        np.full(Python.tuple(10, 10), fill_value=10, dtype=np.float64),
        "Full is broken",
    )


def test_zeros():
    var np = Python.import_module("numpy")
    check_matrices_equal(
        Matrix.zeros[f64](shape=(10, 10), order=order),
        np.zeros(Python.tuple(10, 10), dtype=np.float64),
        "Zeros is broken",
    )


# ===-----------------------------------------------------------------------===#
# Arithmetic
# ===-----------------------------------------------------------------------===#


def test_arithmetic():
    var np = Python.import_module("numpy")
    var A = Matrix.rand[f64]((10, 10), order=order)
    var B = Matrix.rand[f64]((10, 10), order=order)
    var C = Matrix.rand[f64]((10, 1), order=order)
    var Ap = A.to_numpy()
    var Bp = B.to_numpy()
    var Cp = C.to_numpy()
    check_matrices_close(A + B, Ap + Bp, "Add is broken")
    check_matrices_close(A - B, Ap - Bp, "Sub is broken")
    check_matrices_close(A * B, Ap * Bp, "Mul is broken")
    check_matrices_close(A @ B, np.matmul(Ap, Bp), "Matmul is broken")
    check_matrices_close(
        A @ B.reorder_layout(),
        np.matmul(Ap, Bp),
        "Matmul is broken for mixed memory layouts",
    )
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


# FIXME: the gt, lt tests are failing when run together with all other tests even though they pass in isolation. weird behaviour. Commmenting it out temporarily and fix later.
def test_logic():
    var np = Python.import_module("numpy")
    var A = Matrix.ones((5, 1), order=order)
    var B = Matrix.ones((5, 1), order=order)
    var L = Matrix.fromstring[i8](
        "[[0,0,0],[0,0,1],[1,1,1],[1,0,0]]", shape=(4, 3)
    )
    var Anp = np.matrix(A.to_numpy())
    var Bnp = np.matrix(B.to_numpy())
    var Lnp = np.matrix(L.to_numpy())

    var gt_res = A > B
    var gt_res_np = Anp > Bnp
    var lt_res = A < B
    var lt_res_np = Anp < Bnp
    check_matrices_equal[DType.bool](gt_res, gt_res_np, "gt is broken")
    check_matrices_equal[DType.bool](lt_res, lt_res_np, "lt is broken")

    assert_true(
        np.equal(nm.all(L), np.all(Lnp)),
        "`all` is broken",
    )
    for i in range(2):
        check_matrices_close(
            Matrix.all(L, axis=i),
            np.all(Lnp, axis=i),
            String("`all` by axis {i} is broken"),
        )
    assert_true(
        np.equal(Matrix.any(L), np.any(Lnp)),
        "`any` is broken",
    )
    for i in range(2):
        check_matrices_close(
            Matrix.any(L, axis=i),
            np.any(Lnp, axis=i),
            String("`any` by axis {i} is broken"),
        )

    # ===-----------------------------------------------------------------------===#
    # Linear algebra
    # ===-----------------------------------------------------------------------===#

    def test_linalg():
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((100, 100), order=order)
        var B = Matrix.rand[f64]((100, 100), order=order)
        var E = Matrix.fromstring(
            "[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]", shape=(4, 3), order=order
        )
        var Y = Matrix.rand((100, 1), order=order)
        var Anp = A.to_numpy()
        var Bnp = B.to_numpy()
        var Ynp = Y.to_numpy()
        var Enp = E.to_numpy()
        check_matrices_close(
            nm.linalg.solve(A, B),
            np.linalg.solve(Anp, Bnp),
            "Solve is broken",
        )
        check_matrices_close(
            nm.linalg.inv(A),
            np.linalg.inv(Anp),
            "Inverse is broken",
        )
        check_matrices_close(
            nm.linalg.lstsq(A, Y),
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
            np.all(np.isclose(nm.linalg.det(A), np.linalg.det(Anp), atol=0.1)),
            "Determinant is broken",
        )
        for i in range(-10, 10):
            assert_true(
                np.all(
                    np.isclose(
                        nm.linalg.trace(E, offset=i),
                        np.trace(Enp, offset=i),
                        atol=0.1,
                    )
                ),
                "Trace is broken",
            )

    def test_qr_decomposition():
        var A = Matrix.rand[f64]((20, 20), order=order)

        var np = Python.import_module("numpy")

        var Q_R = nm.linalg.qr(A)
        Q = Q_R[0].create_copy()
        R = Q_R[1].create_copy()

        # Check if Q^T Q is close to the identity matrix, i.e Q is orthonormal
        var id = Q.transpose() @ Q
        assert_true(np.allclose(id.to_numpy(), np.eye(Q.shape[0]), atol=1e-14))

        # Check if R is upper triangular
        assert_true(np.allclose(R.to_numpy(), np.triu(R.to_numpy()), atol=1e-14))

        # Check if A = QR
        var A_test = Q @ R
        assert_true(np.allclose(A_test.to_numpy(), A.to_numpy(), atol=1e-14))

    def test_qr_decomposition_asym_reduced():
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((12, 5), order=order)
        var Q_R = nm.linalg.qr(A, mode="reduced")
        Q = Q_R[0].copy()
        R = Q_R[1].copy()

        assert_true(
            Q.shape[0] == 12 and Q.shape[1] == 5,
            "Q has unexpected shape for reduced.",
        )
        assert_true(
            R.shape[0] == 5 and R.shape[1] == 5,
            "R has unexpected shape for reduced.",
        )

        var id = Q.transpose() @ Q
        assert_true(
            np.allclose(id.to_numpy(), np.eye(Q.shape[1]), atol=1e-14),
            "Q not orthonormal for reduced.",
        )
        assert_true(
            np.allclose(R.to_numpy(), np.triu(R.to_numpy()), atol=1e-14),
            "R not upper triangular for reduced.",
        )

        var A_test = Q @ R
        assert_true(np.allclose(A_test.to_numpy(), A.to_numpy(), atol=1e-14))

    def test_qr_decomposition_asym_complete():
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((12, 5), order=order)
        var Q_R = nm.linalg.qr(A, mode="complete")
        var Q = Q_R[0].copy()
        var R = Q_R[1].copy()

        assert_true(
            Q.shape[0] == 12 and Q.shape[1] == 12,
            "Q has unexpected shape for complete.",
        )
        assert_true(
            R.shape[0] == 12 and R.shape[1] == 5,
            "R has unexpected shape for complete.",
        )

        var id = Q.transpose() @ Q
        assert_true(
            np.allclose(id.to_numpy(), np.eye(Q.shape[0]), atol=1e-14),
            "Q not orthonormal for complete.",
        )
        assert_true(
            np.allclose(R.to_numpy(), np.triu(R.to_numpy()), atol=1e-14),
            "R not upper triangular for complete.",
        )

        var A_test = Q @ R
        assert_true(np.allclose(A_test.to_numpy(), A.to_numpy(), atol=1e-14))

    def test_qr_decomposition_asym_complete2():
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((5, 12), order=order)
        var Q_R = nm.linalg.qr(A, mode="complete")
        var Q = Q_R[0].copy()
        var R = Q_R[1].copy()

        assert_true(
            Q.shape[0] == 5 and Q.shape[1] == 5,
            "Q has unexpected shape for complete.",
        )
        assert_true(
            R.shape[0] == 5 and R.shape[1] == 12,
            "R has unexpected shape for complete.",
        )

        var id = Q.transpose() @ Q
        assert_true(
            np.allclose(id.to_numpy(), np.eye(Q.shape[0]), atol=1e-14),
            "Q not orthonormal for complete.",
        )
        assert_true(
            np.allclose(R.to_numpy(), np.triu(R.to_numpy()), atol=1e-14),
            "R not upper triangular for complete.",
        )

        var A_test = Q @ R
        assert_true(np.allclose(A_test.to_numpy(), A.to_numpy(), atol=1e-14))

    def test_eigen_decomposition():
        var np = Python.import_module("numpy")

        # Create a symmetric matrix by adding a matrix to its transpose
        var A_random = Matrix.rand[f64]((10, 10), order=order)
        var A = A_random + A_random.transpose()
        var Anp = A.to_numpy()

        # Compute eigendecomposition
        var Q_Lambda = nm.linalg.eig(A)
        var Q = Q_Lambda[0].copy()
        var Lambda = Q_Lambda[1].copy()

        # Use NumPy for comparison
        namedtuple = np.linalg.eig(Anp)

        np_eigenvalues = namedtuple.eigenvalues

        # Sort eigenvalues and eigenvectors for comparison (numpy doesn't guarantee order)
        var np_sorted_eigenvalues = np.sort(np_eigenvalues)
        var eigenvalues = np.diag(Lambda.to_numpy())
        var sorted_eigenvalues = np.sort(eigenvalues)

        assert_true(
            np.allclose(sorted_eigenvalues, np_sorted_eigenvalues, atol=1e-10),
            "Eigenvalues don't match expected values",
        )

        # Check that eigenvectors are orthogonal (Q^T Q = I)
        var id = Q.transpose() @ Q
        assert_true(
            np.allclose(id.to_numpy(), np.eye(Q.shape[0]), atol=1e-10),
            "Eigenvectors are not orthogonal",
        )

        # Check that A = Q * Lambda * Q^T (eigendecomposition property)
        var A_reconstructed = Q @ Lambda @ Q.transpose()
        assert_true(
            np.allclose(A_reconstructed.to_numpy(), Anp, atol=1e-10),
            "A ≠ Q * Lambda * Q^T",
        )

        # Verify A*v = λ*v for each eigenvector and eigenvalue
        for i in range(A.shape[0]):
            var eigenvector = Matrix.zeros[f64]((A.shape[0], 1), order=order)
            for j in range(A.shape[0]):
                eigenvector[j, 0] = Q[j, i]

            var Av = A @ eigenvector
            var lambda_times_v = eigenvector * Lambda[i, i]

            assert_true(
                np.allclose(Av.to_numpy(), lambda_times_v.to_numpy(), atol=1e-10),
                "Eigenvector verification failed: A*v ≠ λ*v",
            )

        # Verify A*v = λ*v for each eigenvector and eigenvalue
        for i in range(A.shape[0]):
            var eigenvector = Matrix.zeros[f64]((A.shape[0], 1), order=order)
            for j in range(A.shape[0]):
                eigenvector[j, 0] = Q[j, i]

            var Av = A @ eigenvector
            var lambda_times_v = eigenvector * Lambda[i, i]

            assert_true(
                np.allclose(Av.to_numpy(), lambda_times_v.to_numpy(), atol=1e-10),
                "Eigenvector verification failed: A*v ≠ λ*v",
            )

    # ===-----------------------------------------------------------------------===#
    # Mathematics
    # ===-----------------------------------------------------------------------===#

    def test_math():
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((100, 100), order=order)
        var Anp = np.matrix(A.to_numpy())

        assert_true(
            np.all(np.isclose(nm.sum(A), np.sum(Anp), atol=0.1)),
            "`sum` is broken",
        )
        for i in range(2):
            check_matrices_close(
                nm.sum(A, axis=i),
                np.sum(Anp, axis=i),
                String("`sum` by axis {i} is broken"),
            )

        assert_true(
            np.all(np.isclose(nm.prod(A), np.prod(Anp), atol=0.1)),
            "`prod` is broken",
        )
        for i in range(2):
            check_matrices_close(
                nm.prod(A, axis=i),
                np.prod(Anp, axis=i),
                String("`prod` by axis {i} is broken"),
            )

    check_matrices_close(
        nm.cumsum(A),
        np.cumsum(Anp),
        "`cumsum` is broken",
    )
    for i in range(2):
        check_matrices_close(
            nm.cumsum(A, axis=i),
            np.cumsum(Anp, axis=i),
            String("`cumsum` by axis {i} is broken"),
        )

    check_matrices_close(
        nm.cumprod(A),
        np.cumprod(Anp),
        "`cumprod` is broken",
    )
    for i in range(2):
        check_matrices_close(
            nm.cumprod(A.copy(), axis=i),
            np.cumprod(Anp, axis=i),
            String("`cumprod` by axis {i} is broken"),
        )

    def test_trigonometric():
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((100, 100), order=order)
        var Anp = np.matrix(A.to_numpy())
        check_matrices_close(nm.sin(A), np.sin(Anp), "sin is broken")
        check_matrices_close(nm.cos(A), np.cos(Anp), "cos is broken")
        check_matrices_close(nm.tan(A), np.tan(Anp), "tan is broken")
        check_matrices_close(nm.arcsin(A), np.arcsin(Anp), "arcsin is broken")
        check_matrices_close(nm.asin(A), np.arcsin(Anp), "asin is broken")
        check_matrices_close(nm.arccos(A), np.arccos(Anp), "arccos is broken")
        check_matrices_close(nm.acos(A), np.arccos(Anp), "acos is broken")
        check_matrices_close(nm.arctan(A), np.arctan(Anp), "arctan is broken")
        check_matrices_close(nm.atan(A), np.arctan(Anp), "atan is broken")

    def test_hyperbolic():
        var np = Python.import_module("numpy")
        var A = Matrix.fromstring(
            "[[1,2,3],[4,5,6],[7,8,9]]", shape=(3, 3), order=order
        )
        var B = A / 10
        var Anp = np.matrix(A.to_numpy())
        var Bnp = np.matrix(B.to_numpy())
        check_matrices_close(nm.sinh(A), np.sinh(Anp), "sinh is broken")
        check_matrices_close(nm.cosh(A), np.cosh(Anp), "cosh is broken")
        check_matrices_close(nm.tanh(A), np.tanh(Anp), "tanh is broken")
        check_matrices_close(nm.arcsinh(A), np.arcsinh(Anp), "arcsinh is broken")
        check_matrices_close(nm.asinh(A), np.arcsinh(Anp), "asinh is broken")
        check_matrices_close(nm.arccosh(A), np.arccosh(Anp), "arccosh is broken")
        check_matrices_close(nm.acosh(A), np.arccosh(Anp), "acosh is broken")
        check_matrices_close(nm.arctanh(B), np.arctanh(Bnp), "arctanh is broken")
        check_matrices_close(nm.atanh(B), np.arctanh(Bnp), "atanh is broken")

    def test_sorting():
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((10, 10), order=order)
        var Anp = np.matrix(A.to_numpy())

        check_matrices_close(
            nm.sort(A), np.sort(Anp, axis=None), String("Sort is broken")
        )
        for i in range(2):
            check_matrices_close(
                nm.sort(A.copy(), axis=i),
                np.sort(Anp, axis=i),
                String("Sort by axis {} is broken").format(i),
            )

        check_matrices_close(
            nm.argsort(A), np.argsort(Anp, axis=None), String("Argsort is broken")
        )
        for i in range(2):
            check_matrices_close(
                nm.argsort(A.copy(), axis=i),
                np.argsort(Anp, axis=i),
                String("Argsort by axis {} is broken").format(i),
            )

    def test_searching():
        var np = Python.import_module("numpy")
        var A = Matrix.rand[f64]((10, 10), order=order)
        var Anp = np.matrix(A.to_numpy())

        check_values_close(
            nm.max(A), np.max(Anp, axis=None), String("`max` is broken")
        )
        for i in range(2):
            check_matrices_close(
                nm.max(A, axis=i),
                np.max(Anp, axis=i),
                String("`max` by axis {} is broken").format(i),
            )

        check_values_close(
            nm.argmax(A), np.argmax(Anp, axis=None), String("`argmax` is broken")
        )
        for i in range(2):
            check_matrices_close(
                nm.argmax(A, axis=i),
                np.argmax(Anp, axis=i),
                String("`argmax` by axis {} is broken").format(i),
            )

        check_values_close(
            nm.min(A), np.min(Anp, axis=None), String("`min` is broken.")
        )
        for i in range(2):
            check_matrices_close(
                nm.min(A, axis=i),
                np.min(Anp, axis=i),
                String("`min` by axis {} is broken").format(i),
            )

    check_values_close(
        nm.argmin(A), np.argmin(Anp, axis=None), String("`argmin` is broken.")
    )
    for i in range(2):
        check_matrices_close(
            nm.argmin(A, axis=i),
            np.argmin(Anp, axis=i),
            String("`argmin` by axis {} is broken").format(i),
        )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
