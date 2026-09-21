from std.python import Python, PythonObject
from utils_for_test import check, check_is_close, check_values_close
from std.testing import TestSuite
from std.testing.testing import assert_true

import numojo as nm
from numojo.prelude import *

# ===-----------------------------------------------------------------------===#
# Matmul
# ===-----------------------------------------------------------------------===#
# ! MATMUL RESULTS IN A SEGMENTATION FAULT EXCEPT FOR NAIVE ONE, BUT NAIVE OUTPUTS WRONG VALUES


def test_matmul_small() raises:
    var np = Python.import_module("numpy")
    var arr = nm.ones[i8](Shape(4, 4))
    var np_arr = np.ones(Python.tuple(4, 4), dtype=np.int8)
    check_is_close(
        arr @ arr, np.matmul(np_arr, np_arr), "Dunder matmul is broken"
    )


def test_matmul() raises:
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 100)
    arr.resize(Shape(10, 10))
    var np_arr = np.arange(0, 100).reshape(10, 10)
    check_is_close(
        arr @ arr, np.matmul(np_arr, np_arr), "Dunder matmul is broken"
    )
    # The only matmul that currently works is par (__matmul__)
    # check_is_close(nm.matmul_tiled_unrolled_parallelized(arr,arr),np.matmul(np_arr,np_arr),"TUP matmul is broken")


def test_matmul_4dx4d() raises:
    var np = Python.import_module("numpy")
    var A = nm.random.randn(2, 3, 4, 5)
    var B = nm.random.randn(2, 3, 5, 4)
    check_is_close(
        A @ B,
        np.matmul(A.to_numpy(), B.to_numpy()),
        "`matmul_4dx4d` is broken",
    )


def test_matmul_8dx8d() raises:
    var np = Python.import_module("numpy")
    var A = nm.random.randn(2, 3, 4, 5, 6, 5, 4, 3)
    var B = nm.random.randn(2, 3, 4, 5, 6, 5, 3, 2)
    check_is_close(
        A @ B,
        np.matmul(A.to_numpy(), B.to_numpy()),
        "`matmul_8dx8d` is broken",
    )


def test_matmul_1dx2d() raises:
    var np = Python.import_module("numpy")
    var arr1 = nm.random.randn(4)
    var arr2 = nm.random.randn(4, 8)
    var nparr1 = arr1.to_numpy()
    var nparr2 = arr2.to_numpy()
    check_is_close(
        arr1 @ arr2, np.matmul(nparr1, nparr2), "Dunder matmul is broken"
    )


def test_matmul_2dx2d_wide() raises:
    """Test 2D matmul on rows wider than the kernel's vectorization width.

    The kernel broadcasts a single element of `A` against a vector of `B`, so
    it only exercises its full-width path once the last dimension reaches
    `max(simd_width_of[dtype](), 16)`. Uniform values hide a broadcast that is
    wrongly widened into a vector load, so these arrays are random.
    """

    def check_shape(m: Int, k: Int, n: Int) raises:
        var np = Python.import_module("numpy")
        var A = nm.random.randn(m, k)
        var B = nm.random.randn(k, n)
        check_is_close(
            A @ B,
            np.matmul(A.to_numpy(), B.to_numpy()),
            String("`matmul` on a {}x{} @ {}x{} is broken").format(m, k, k, n),
        )

    # Widths on either side of the vectorization boundary: an exact multiple,
    # a width that leaves a scalar remainder, and a non-square shape.
    check_shape(32, 32, 32)
    check_shape(20, 20, 20)
    check_shape(17, 33, 20)
    check_shape(5, 5, 64)


def test_matmul_2dx2d_wide_f_order() raises:
    """Test 2D matmul on wide rows when an operand is not C-contiguous."""
    var np = Python.import_module("numpy")

    var A = nm.random.randn(24, 20)
    var B = nm.random.randn(20, 24)
    var A_f = nm.random.randn(24, 20).reshape(Shape(24, 20), order="F")
    var B_f = nm.random.randn(20, 24).reshape(Shape(20, 24), order="F")
    assert_true(not A_f.is_c_contiguous(), "`A_f` should be F-order")
    assert_true(not B_f.is_c_contiguous(), "`B_f` should be F-order")

    check_is_close(
        A_f @ B,
        np.matmul(A_f.to_numpy(), B.to_numpy()),
        "`matmul` with an F-order A is broken",
    )
    check_is_close(
        A @ B_f,
        np.matmul(A.to_numpy(), B_f.to_numpy()),
        "`matmul` with an F-order B is broken",
    )
    check_is_close(
        A_f @ B_f,
        np.matmul(A_f.to_numpy(), B_f.to_numpy()),
        "`matmul` with two F-order operands is broken",
    )


def test_matmul_2dx1d() raises:
    var np = Python.import_module("numpy")
    var arr1 = nm.random.randn(11, 4)
    var arr2 = nm.random.randn(4)
    var nparr1 = arr1.to_numpy()
    var nparr2 = arr2.to_numpy()
    check_is_close(
        arr1 @ arr2, np.matmul(nparr1, nparr2), "Dunder matmul is broken"
    )


# ! The `inv` is broken, it outputs -INF for some values
def test_inv() raises:
    var np = Python.import_module("numpy")
    var arr = nm.random.rand(100, 100)
    var np_arr = arr.to_numpy()
    check_is_close(
        nm.math.linalg.inv(arr), np.linalg.inv(np_arr), "Inverse is broken"
    )


# ! The `solve` is broken, it outputs -INF, nan, 0 etc for some values
def test_solve() raises:
    var np = Python.import_module("numpy")
    var A = nm.random.randn(100, 100)
    var B = nm.random.randn(100, 50)
    var A_np = A.to_numpy()
    var B_np = B.to_numpy()
    check_is_close(
        nm.linalg.solve(A, B),
        np.linalg.solve(A_np, B_np),
        "Solve is broken",
    )


def norms() raises:
    var np = Python.import_module("numpy")
    var arr = nm.random.rand(20, 20)
    var np_arr = arr.to_numpy()
    check_values_close(
        nm.math.linalg.det(arr), np.linalg.det(np_arr), "`det` is broken"
    )


def test_misc() raises:
    var np = Python.import_module("numpy")
    var arr = nm.random.rand(4, 8)
    var np_arr = arr.to_numpy()
    for i in range(-(arr.shape[0] - 1), arr.shape[1]):
        check_is_close(
            nm.diagonal(arr, offset=i),
            np.diagonal(np_arr, offset=i),
            String("`diagonal` with offset {} is broken").format(i),
        )


# ===-----------------------------------------------------------------------===#
# Products: outer, kron, tensordot
# ===-----------------------------------------------------------------------===#


def test_outer() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.f64](6).reshape(Shape(2, 3))
    var b = nm.arange[nm.f64](4)
    check_is_close(
        nm.linalg.outer(a, b),
        np.outer(a.to_numpy(), b.to_numpy()),
        "`outer` is broken",
    )


def test_kron() raises:
    var np = Python.import_module("numpy")
    var A = nm.arange[nm.f64](6).reshape(Shape(2, 3))
    var B = nm.eye[nm.f64](2, 2)
    check_is_close(
        nm.linalg.kron(A, B),
        np.kron(A.to_numpy(), B.to_numpy()),
        "`kron` is broken for equal-ndim inputs",
    )

    var A1 = nm.arange[nm.f64](6)
    var B1 = nm.arange[nm.f64](6).reshape(Shape(2, 3))
    check_is_close(
        nm.linalg.kron(A1, B1),
        np.kron(A1.to_numpy(), B1.to_numpy()),
        "`kron` is broken for mismatched ndim inputs",
    )


def test_tensordot() raises:
    var np = Python.import_module("numpy")
    var t1 = nm.arange[nm.f64](60).reshape(Shape(3, 4, 5))
    var t2 = nm.arange[nm.f64](24).reshape(Shape(4, 3, 2))
    var py_axes = Python.list(Python.list(1, 0), Python.list(0, 1))
    check_is_close(
        nm.linalg.tensordot(t1, t2, axes_a=[1, 0], axes_b=[0, 1]),
        np.tensordot(t1.to_numpy(), t2.to_numpy(), axes=py_axes),
        "`tensordot` with explicit axes lists is broken",
    )

    var t3 = nm.arange[nm.f64](120).reshape(Shape(4, 5, 6))
    check_is_close(
        nm.linalg.tensordot(t1, t3, axes=2),
        np.tensordot(t1.to_numpy(), t3.to_numpy(), axes=2),
        "`tensordot` with `axes=2` is broken",
    )

    var v1 = nm.arange[nm.f64](3)
    var v2 = nm.arange[nm.f64](4)
    check_is_close(
        nm.linalg.tensordot(v1, v2, axes=0),
        np.tensordot(v1.to_numpy(), v2.to_numpy(), axes=0),
        "`tensordot` with `axes=0` is broken",
    )


# ===-----------------------------------------------------------------------===#
# Decompositions: qr, cholesky; Solving: lstsq
# ===-----------------------------------------------------------------------===#


def test_qr() raises:
    def check_qr(m: Int, n: Int, msg: String) raises:
        var np = Python.import_module("numpy")
        var A = nm.random.randn(m, n)
        var Q_R = nm.linalg.qr(A)
        var Q = Q_R[0].copy()
        var R = Q_R[1].copy()
        var k = min(m, n)
        assert_true(Q.shape[0] == m and Q.shape[1] == k, msg + ": Q shape")
        assert_true(R.shape[0] == k and R.shape[1] == n, msg + ": R shape")
        check_is_close(
            Q @ R, A.to_numpy(), msg + ": Q @ R does not reconstruct A"
        )
        var identity_k = np.eye(k)
        check_is_close(
            nm.transpose(Q) @ Q,
            identity_k,
            msg + ": columns of Q are not orthonormal",
        )

    check_qr(5, 3, "qr tall")
    check_qr(3, 6, "qr wide")
    check_qr(4, 4, "qr square")


def test_cholesky() raises:
    var np = Python.import_module("numpy")
    var M = nm.random.randn(5, 5)
    var SPD = M @ nm.transpose(M) + nm.eye[nm.f64](5, 5) * 5.0
    var L = nm.linalg.cholesky(SPD)
    check_is_close(
        L, np.linalg.cholesky(SPD.to_numpy()), "`cholesky` is broken"
    )


def test_lstsq() raises:
    var np = Python.import_module("numpy")

    # Overdetermined, vector rhs.
    var A = nm.random.randn(8, 3)
    var b = nm.random.randn(8)
    check_is_close(
        nm.linalg.lstsq(A, b),
        np.linalg.lstsq(A.to_numpy(), b.to_numpy(), rcond=Python.none())[0],
        "`lstsq` is broken for an overdetermined system",
    )

    # Overdetermined, matrix rhs.
    var B = nm.random.randn(8, 2)
    check_is_close(
        nm.linalg.lstsq(A, B),
        np.linalg.lstsq(A.to_numpy(), B.to_numpy(), rcond=Python.none())[0],
        "`lstsq` is broken for a matrix right-hand side",
    )

    # Underdetermined, full row rank.
    var A2 = nm.random.randn(3, 6)
    var b2 = nm.random.randn(3)
    check_is_close(
        nm.linalg.lstsq(A2, b2),
        np.linalg.lstsq(A2.to_numpy(), b2.to_numpy(), rcond=Python.none())[0],
        "`lstsq` is broken for an underdetermined system",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
