import numojo as nm
from numojo.prelude import *
from python import Python, PythonObject
from utils_for_test import check, check_is_close, check_values_close

# ===-----------------------------------------------------------------------===#
# Matmul
# ===-----------------------------------------------------------------------===#
# ! MATMUL RESULTS IN A SEGMENTATION FAULT EXCEPT FOR NAIVE ONE, BUT NAIVE OUTPUTS WRONG VALUES


def test_matmul_small():
    var np = Python.import_module("numpy")
    var arr = nm.ones[i8](Shape(4, 4))
    var np_arr = np.ones((4, 4), dtype=np.int8)
    check_is_close(
        arr @ arr, np.matmul(np_arr, np_arr), "Dunder matmul is broken"
    )


def test_matmul():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 100)
    arr.resize(Shape(10, 10))
    var np_arr = np.arange(0, 100).reshape(10, 10)
    check_is_close(
        arr @ arr, np.matmul(np_arr, np_arr), "Dunder matmul is broken"
    )
    # The only matmul that currently works is par (__matmul__)
    # check_is_close(nm.matmul_tiled_unrolled_parallelized(arr,arr),np.matmul(np_arr,np_arr),"TUP matmul is broken")


def test_matmul_4dx4d():
    var np = Python.import_module("numpy")
    var A = nm.random.randn(2, 3, 4, 5)
    var B = nm.random.randn(2, 3, 5, 4)
    check_is_close(
        A @ B,
        np.matmul(A.to_numpy(), B.to_numpy()),
        "`matmul_4dx4d` is broken",
    )


def test_matmul_8dx8d():
    var np = Python.import_module("numpy")
    var A = nm.random.randn(2, 3, 4, 5, 6, 5, 4, 3)
    var B = nm.random.randn(2, 3, 4, 5, 6, 5, 3, 2)
    check_is_close(
        A @ B,
        np.matmul(A.to_numpy(), B.to_numpy()),
        "`matmul_8dx8d` is broken",
    )


def test_matmul_1dx2d():
    var np = Python.import_module("numpy")
    var arr1 = nm.random.randn(4)
    var arr2 = nm.random.randn(4, 8)
    var nparr1 = arr1.to_numpy()
    var nparr2 = arr2.to_numpy()
    check_is_close(
        arr1 @ arr2, np.matmul(nparr1, nparr2), "Dunder matmul is broken"
    )


def test_matmul_2dx1d():
    var np = Python.import_module("numpy")
    var arr1 = nm.random.randn(11, 4)
    var arr2 = nm.random.randn(4)
    var nparr1 = arr1.to_numpy()
    var nparr2 = arr2.to_numpy()
    check_is_close(
        arr1 @ arr2, np.matmul(nparr1, nparr2), "Dunder matmul is broken"
    )


# ! The `inv` is broken, it outputs -INF for some values
def test_inv():
    var np = Python.import_module("numpy")
    var arr = nm.core.random.rand(100, 100)
    var np_arr = arr.to_numpy()
    check_is_close(
        nm.math.linalg.inv(arr), np.linalg.inv(np_arr), "Inverse is broken"
    )


# ! The `solve` is broken, it outputs -INF, nan, 0 etc for some values
def test_solve():
    var np = Python.import_module("numpy")
    var A = nm.core.random.randn(100, 100)
    var B = nm.core.random.randn(100, 50)
    var A_np = A.to_numpy()
    var B_np = B.to_numpy()
    check_is_close(
        nm.linalg.solve(A, B),
        np.linalg.solve(A_np, B_np),
        "Solve is broken",
    )


def norms():
    var np = Python.import_module("numpy")
    var arr = nm.core.random.rand(20, 20)
    var np_arr = arr.to_numpy()
    check_values_close(
        nm.math.linalg.det(arr), np.linalg.det(np_arr), "`det` is broken"
    )


def test_misc():
    var np = Python.import_module("numpy")
    var arr = nm.core.random.rand(4, 8)
    var np_arr = arr.to_numpy()
    for i in range(-3, 8):
        check_is_close(
            nm.diagonal(arr, offset=i),
            np.diagonal(np_arr, offset=i),
            String("`diagonal` by axis {} is broken").format(i),
        )
