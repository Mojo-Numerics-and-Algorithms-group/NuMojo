import numojo as nm
from numojo.prelude import *
from time import now
from python import Python, PythonObject
from utils_for_test import check, check_is_close
from testing.testing import assert_raises


def test_list_creation_methods():
    var np = Python.import_module("numpy")
    var fill_val: Scalar[nm.i32] = 5
    check(
        nm.NDArray(5, 5, 5, fill=fill_val),
        np.zeros([5, 5, 5], dtype=np.float64) + 5,
        "*shape broken",
    )
    check(
        nm.NDArray(List[Int](5, 5, 5), fill=fill_val),
        np.zeros([5, 5, 5], dtype=np.float64) + 5,
        "List[int] shape broken",
    )
    check(
        nm.NDArray(VariadicList[Int](5, 5, 5), fill=fill_val),
        np.zeros([5, 5, 5], dtype=np.float64) + 5,
        "VariadicList[Int] shape broken",
    )
    check(
        nm.NDArray(nm.NDArrayShape(5, 5, 5), fill=fill_val),
        np.zeros([5, 5, 5], dtype=np.float64) + 5,
        "NDArrayShape shape broken",
    )
    # with assert_raises(
    #     contains="Error if random is true you cannot set a fill value"
    # ):
    #     _ = nm.NDArray(nm.NDArrayShape(5, 5, 5), fill=fill_val)


def test_arange():
    var np = Python.import_module("numpy")
    check(
        nm.arange[nm.i64](0, 100),
        np.arange(0, 100, dtype=np.int64),
        "Arange is broken",
    )
    check(
        nm.arange[nm.f64](0, 100),
        np.arange(0, 100, dtype=np.float64),
        "Arange is broken",
    )


def test_linspace():
    var np = Python.import_module("numpy")
    check(
        nm.linspace[nm.f64](0, 100),
        np.linspace(0, 100, dtype=np.float64),
        "Linspace is broken",
    )


def test_logspace():
    var np = Python.import_module("numpy")
    check_is_close(
        nm.logspace[nm.f64](0, 100, 5),
        np.logspace(0, 100, 5, dtype=np.float64),
        "Logspace is broken",
    )


def test_geomspace():
    var np = Python.import_module("numpy")
    check_is_close(
        nm.geomspace[nm.f64](1, 100, 5),
        np.geomspace(1, 100, 5, dtype=np.float64),
        "Logspace is broken",
    )


def test_zeros():
    var np = Python.import_module("numpy")
    check(
        nm.zeros[f64](Shp(10, 10, 10, 10)),
        np.zeros((10, 10, 10, 10), dtype=np.float64),
        "Zeros is broken",
    )


def test_ones():
    var np = Python.import_module("numpy")
    check(
        nm.ones[nm.f64](Shp(10, 10, 10, 10)),
        np.ones((10, 10, 10, 10), dtype=np.float64),
        "Ones is broken",
    )


def test_full():
    var np = Python.import_module("numpy")
    check(
        nm.full[nm.f64](Shp(10, 10, 10, 10), fill_value=10),
        np.full((10, 10, 10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


def test_identity():
    var np = Python.import_module("numpy")
    check(
        nm.identity[nm.i64](100),
        np.identity(100, dtype=np.int64),
        "Identity is broken",
    )


def test_eye():
    var np = Python.import_module("numpy")
    check(
        nm.eye[nm.i64](100, 100),
        np.eye(100, 100, dtype=np.int64),
        "Eye is broken",
    )


def test_fromstring():
    var A = nm.fromstring("[[[1,2],[3,4]],[[5,6],[7,8]]]")
    var B = nm.array[DType.int32](String("[0.1, -2.3, 41.5, 19.29145, -199]"))
    print(A)
    print(B)


# def test_diagflat():
#     var np = Python.import_module("numpy")
#     var temp = nm.arange[nm.i64](1, 10, 10)
#     temp.reshape(3,3)
#     check(nm.diagflat[nm.i64](nm), np.diagflat(np.arange(1,10,10).reshape(3,3), "Diagflat is broken")
