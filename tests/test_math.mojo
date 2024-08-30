import numojo as nm
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
        nm.add[
            nm.f64,
            backend = nm.math.math_funcs.VectorizedParallelizedNWorkers[6],
        ](arr, 5.0),
        np.arange(0, 500) + 5,
        "Add array + scalar",
    )
    check(
        nm.add[nm.f64, nm.math.math_funcs.VectorizedParallelizedNWorkers[6]](
            arr, arr
        ),
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
            backend = nm.math.math_funcs.VectorizedParallelizedNWorkers[6],
        ](arr),
        np.sin(np.arange(0, 15)),
        "Add array + scalar",
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
