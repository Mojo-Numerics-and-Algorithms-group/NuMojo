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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
