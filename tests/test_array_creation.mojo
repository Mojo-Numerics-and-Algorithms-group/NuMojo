import numojo as nm
from numojo.prelude import *
from numojo.prelude import *
from testing.testing import (
    assert_true,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)
from python import Python, PythonObject
from utils_for_test import check, check_is_close


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
        nm.zeros[f64](Shape(10, 10, 10, 10)),
        np.zeros((10, 10, 10, 10), dtype=np.float64),
        "Zeros is broken",
    )


def test_ones_from_shape():
    var np = Python.import_module("numpy")
    check(
        nm.ones[nm.f64](Shape(10, 10, 10, 10)),
        np.ones((10, 10, 10, 10), dtype=np.float64),
        "Ones is broken",
    )


def test_ones_from_list():
    var np = Python.import_module("numpy")
    check(
        nm.ones[nm.f64](List[Int](10, 10, 10, 10)),
        np.ones((10, 10, 10, 10), dtype=np.float64),
        "Ones is broken",
    )


def test_ones_from_vlist():
    var np = Python.import_module("numpy")
    check(
        nm.ones[nm.f64](VariadicList[Int](10, 10, 10, 10)),
        np.ones((10, 10, 10, 10), dtype=np.float64),
        "Ones is broken",
    )


def test_full():
    var np = Python.import_module("numpy")
    check(
        nm.full[nm.f64](Shape(10, 10, 10, 10), fill_value=10),
        np.full((10, 10, 10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


def test_full_from_shape():
    var np = Python.import_module("numpy")
    check(
        nm.full[nm.f64](Shape(10, 10, 10, 10), fill_value=10),
        np.full((10, 10, 10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


def test_full_from_list():
    var np = Python.import_module("numpy")
    check(
        nm.full[nm.f64](List[Int](10, 10, 10, 10), fill_value=10),
        np.full((10, 10, 10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


def test_full_from_vlist():
    var np = Python.import_module("numpy")
    check(
        nm.full[nm.f64](VariadicList[Int](10, 10, 10, 10), fill_value=10),
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


def test_fromstring_complicated():
    var s = """
    [[[[1,2,10],
       [3,4,2]],
       [[5,6,4],
       [7,8,10]]],
     [[[1,2,12],
       [3,4,41]],
       [[5,6,12],
       [7,8,99]]]]
    """
    var A = nm.fromstring(s)
    print(A)


def test_diag():
    var np = Python.import_module("numpy")
    var x_nm = nm.arange[f32](0, 9, step=1)
    x_nm.resize(Shape(3, 3))
    var x_np = np.arange(0, 9, step=1).reshape(3, 3)

    x_nm_k0 = nm.diag[f32](x_nm, k=0)
    x_np_k0 = np.diag(x_np, k=0)
    check(x_nm_k0, x_np_k0, "Diag is broken (k=0)")

    x_nm_k1 = nm.diag[f32](x_nm, k=1)
    x_np_k1 = np.diag(x_np, k=1)
    check(x_nm_k1, x_np_k1, "Diag is broken (k=1)")

    x_nm_km1 = nm.diag[f32](x_nm, k=-1)
    x_np_km1 = np.diag(x_np, k=-1)
    check(x_nm_km1, x_np_km1, "Diag is broken (k=-1)")

    x_nm_rev_k1 = nm.diag[f32](x_nm_k0, k=0)
    x_np_rev_k1 = np.diag(x_np_k0, k=0)
    check(x_nm_rev_k1, x_np_rev_k1, "Diag reverse is broken (k=0)")

    x_nm_rev_km1 = nm.diag[f32](x_nm_km1, k=-1)
    x_np_rev_km1 = np.diag(x_np_km1, k=-1)
    check(x_nm_rev_km1, x_np_rev_km1, "Diag reverse is broken (k=-1)")


def test_diagflat():
    var np = Python.import_module("numpy")
    var nm_arr = nm.arange[nm.i64](0, 9, 1)
    nm_arr.resize(Shape(3, 3))
    var np_arr = np.arange(0, 9, 1).reshape(3, 3)

    var x_nm = nm.diagflat[nm.i64](nm_arr, k=0)
    var x_np = np.diagflat(np_arr, k=0)
    check(x_nm, x_np, "Diagflat is broken (k=0)")

    var x_nm_k1 = nm.diagflat[nm.i64](nm_arr, k=1)
    var x_np_k1 = np.diagflat(np_arr, k=1)
    check(x_nm_k1, x_np_k1, "Diagflat is broken (k=1)")

    var x_nm_km1 = nm.diagflat[nm.i64](nm_arr, k=-1)
    var x_np_km1 = np.diagflat(np_arr, k=-1)
    check(x_nm_km1, x_np_km1, "Diagflat is broken (k=-1)")


def test_tri():
    var np = Python.import_module("numpy")

    var x_nm = nm.tri[nm.f32](3, 4, k=0)
    var x_np = np.tri(3, 4, k=0, dtype=np.float32)
    check(x_nm, x_np, "Tri is broken (k=0)")

    var x_nm_k1 = nm.tri[nm.f32](3, 4, k=1)
    var x_np_k1 = np.tri(3, 4, k=1, dtype=np.float32)
    check(x_nm_k1, x_np_k1, "Tri is broken (k=1)")

    var x_nm_km1 = nm.tri[nm.f32](3, 4, k=-1)
    var x_np_km1 = np.tri(3, 4, k=-1, dtype=np.float32)
    check(x_nm_km1, x_np_km1, "Tri is broken (k=-1)")


def test_tril():
    var np = Python.import_module("numpy")
    var nm_arr = nm.arange[nm.f32](0, 9, 1)
    nm_arr.resize(Shape(3, 3))
    var np_arr = np.arange(0, 9, 1, dtype=np.float32).reshape(3, 3)

    var x_nm = nm.tril[nm.f32](nm_arr, k=0)
    var x_np = np.tril(np_arr, k=0)
    check(x_nm, x_np, "Tril is broken (k=0)")

    var x_nm_k1 = nm.tril[nm.f32](nm_arr, k=1)
    var x_np_k1 = np.tril(np_arr, k=1)
    check(x_nm_k1, x_np_k1, "Tril is broken (k=1)")

    var x_nm_km1 = nm.tril[nm.f32](nm_arr, k=-1)
    var x_np_km1 = np.tril(np_arr, k=-1)
    check(x_nm_km1, x_np_km1, "Tril is broken (k=-1)")

    # Test with higher dimensional array
    var nm_arr_3d = nm.arange[nm.f32](0, 60, 1)
    nm_arr_3d.resize(Shape(3, 4, 5))
    var np_arr_3d = np.arange(0, 60, 1, dtype=np.float32).reshape(3, 4, 5)

    var x_nm_3d = nm.tril[nm.f32](nm_arr_3d, k=0)
    var x_np_3d = np.tril(np_arr_3d, k=0)
    check(x_nm_3d, x_np_3d, "Tril is broken for 3D array (k=0)")

    var x_nm_3d_k1 = nm.tril[nm.f32](nm_arr_3d, k=1)
    var x_np_3d_k1 = np.tril(np_arr_3d, k=1)
    check(x_nm_3d_k1, x_np_3d_k1, "Tril is broken for 3D array (k=1)")

    var x_nm_3d_km1 = nm.tril[nm.f32](nm_arr_3d, k=-1)
    var x_np_3d_km1 = np.tril(np_arr_3d, k=-1)
    check(x_nm_3d_km1, x_np_3d_km1, "Tril is broken for 3D array (k=-1)")


def test_triu():
    var np = Python.import_module("numpy")
    var nm_arr = nm.arange[nm.f32](0, 9, 1)
    nm_arr.resize(Shape(3, 3))
    var np_arr = np.arange(0, 9, 1, dtype=np.float32).reshape(3, 3)

    var x_nm = nm.triu[nm.f32](nm_arr, k=0)
    var x_np = np.triu(np_arr, k=0)
    check(x_nm, x_np, "Triu is broken (k=0)")

    var x_nm_k1 = nm.triu[nm.f32](nm_arr, k=1)
    var x_np_k1 = np.triu(np_arr, k=1)
    check(x_nm_k1, x_np_k1, "Triu is broken (k=1)")

    var x_nm_km1 = nm.triu[nm.f32](nm_arr, k=-1)
    var x_np_km1 = np.triu(np_arr, k=-1)
    check(x_nm_km1, x_np_km1, "Triu is broken (k=-1)")

    # Test with higher dimensional array
    var nm_arr_3d = nm.arange[nm.f32](0, 60, 1)
    nm_arr_3d.resize(Shape(3, 4, 5))
    var np_arr_3d = np.arange(0, 60, 1, dtype=np.float32).reshape(3, 4, 5)

    var x_nm_3d = nm.triu[nm.f32](nm_arr_3d, k=0)
    var x_np_3d = np.triu(np_arr_3d, k=0)
    check(x_nm_3d, x_np_3d, "Triu is broken for 3D array (k=0)")

    var x_nm_3d_k1 = nm.triu[nm.f32](nm_arr_3d, k=1)
    var x_np_3d_k1 = np.triu(np_arr_3d, k=1)
    check(x_nm_3d_k1, x_np_3d_k1, "Triu is broken for 3D array (k=1)")

    var x_nm_3d_km1 = nm.triu[nm.f32](nm_arr_3d, k=-1)
    var x_np_3d_km1 = np.triu(np_arr_3d, k=-1)
    check(x_nm_3d_km1, x_np_3d_km1, "Tril is broken for 3D array (k=-1)")


def test_vander():
    var np = Python.import_module("numpy")
    var nm_arr = nm.arange[nm.f32](1, 5, 1)
    var np_arr = np.arange(1, 5, 1, dtype=np.float32)

    var x_nm = nm.vander[nm.f32](nm_arr)
    var x_np = np.vander(np_arr)
    check(x_nm, x_np, "Vander is broken (default)")

    var x_nm_N3 = nm.vander[nm.f32](nm_arr, N=3)
    var x_np_N3 = np.vander(np_arr, N=3)
    check(x_nm_N3, x_np_N3, "Vander is broken (N=3)")

    var x_nm_inc = nm.vander[nm.f32](nm_arr, increasing=True)
    var x_np_inc = np.vander(np_arr, increasing=True)
    check(x_nm_inc, x_np_inc, "Vander is broken (increasing=True)")

    var x_nm_N3_inc = nm.vander[nm.f32](nm_arr, N=3, increasing=True)
    var x_np_N3_inc = np.vander(np_arr, N=3, increasing=True)
    check(x_nm_N3_inc, x_np_N3_inc, "Vander is broken (N=3, increasing=True)")

    # Test with different dtype
    var nm_arr_int = nm.arange[nm.i32](1, 5, 1)
    var np_arr_int = np.arange(1, 5, 1, dtype=np.int32)

    var x_nm_int = nm.vander[nm.i32](nm_arr_int)
    var x_np_int = np.vander(np_arr_int)
    check(x_nm_int, x_np_int, "Vander is broken (int32)")


def test_arr_manipulation():
    var np = Python.import_module("numpy")
    var arr6 = nm.array[f32](
        data=List[SIMD[f32, 1]](1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        shape=List[Int](2, 5),
    )
    assert_true(
        arr6.shape[0] == 2,
        "NDArray constructor with data and shape: shape element 0",
    )
    assert_true(
        arr6.shape[1] == 5,
        "NDArray constructor with data and shape: shape element 1",
    )
    assert_equal(
        arr6[idx(1, 4)],
        10.0,
        "NDArray constructor with data: value check",
    )
