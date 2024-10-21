import numojo as nm
from numojo.prelude import *
from time import now
from python import Python, PythonObject
from utils_for_test import check, check_is_close


def test_add_array():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check(nm.add[nm.f64](arr, 5.0), np.arange(0, 15) + 5, "Add array + scalar")
    check(
        nm.add[nm.f64](arr, arr),
        np.arange(0, 15) + np.arange(0, 15),
        "Add array + array",
    )


def test_add_array_par():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 500)

    check(
        nm.add[nm.f64, backend = nm.math.math_funcs.Vectorized](arr, 5.0),
        np.arange(0, 500) + 5,
        "Add array + scalar",
    )
    check(
        nm.add[nm.f64, backend = nm.math.math_funcs.Vectorized](arr, arr),
        np.arange(0, 500) + np.arange(0, 500),
        "Add array + array",
    )


def test_sin():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check_is_close(
        nm.sin[nm.f64](arr), np.sin(np.arange(0, 15)), "Add array + scalar"
    )


def test_sin_par():
    var np = Python.import_module("numpy")
    var arr = nm.arange[nm.f64](0, 15)

    check_is_close(
        nm.sin[
            nm.f64,
            backend = nm.math.math_funcs.Vectorized,
        ](arr),
        np.sin(np.arange(0, 15)),
        "Add array + scalar",
    )


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
    arr.reshape(10, 10)
    var np_arr = np.arange(0, 100).reshape(10, 10)
    check_is_close(
        arr @ arr, np.matmul(np_arr, np_arr), "Dunder matmul is broken"
    )
    # The only matmul that currently works is par (__matmul__)
    # check_is_close(nm.matmul_tiled_unrolled_parallelized(arr,arr),np.matmul(np_arr,np_arr),"TUP matmul is broken")


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
        nm.math.linalg.solver.solve(A, B),
        np.linalg.solve(A_np, B_np),
        "Solve is broken",
    )
