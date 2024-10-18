import numojo as nm
from numojo.prelude import *
from time import now
from python import Python, PythonObject
from testing.testing import assert_raises, assert_true


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


def test_full():
    var np = Python.import_module("numpy")
    check(
        nm.mat.full[f64]((10, 10), 10),
        np.full((10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


def test_matmul():
    var np = Python.import_module("numpy")
    var arr = nm.mat.rand[f64]((100, 100))
    var np_arr = arr.to_numpy()
    check_is_close(
        arr @ arr, np.matmul(np_arr, np_arr), "Dunder matmul is broken"
    )


def test_inv():
    var np = Python.import_module("numpy")
    var arr1 = nm.mat.rand[f64]((100, 100))
    var arr2 = nm.mat.rand[f64]((100, 100))
    var np_arr1 = arr1.to_numpy()
    var np_arr2 = arr2.to_numpy()
    check_is_close(
        nm.mat.solve(arr1, arr2),
        np.linalg.solve(np_arr1, np_arr2),
        "Dunder matmul is broken",
    )
