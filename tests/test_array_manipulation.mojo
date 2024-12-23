import numojo as nm
from numojo import *
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close
from python import Python


def test_arr_manipulation():
    var np = Python.import_module("numpy")

    # Test arange
    var A = nm.arange[nm.i16](1, 7, 1)
    var np_A = np.arange(1, 7, 1, dtype=np.int16)
    check_is_close(A, np_A, "Arange operation")

    # Test flip
    var flipped_A = nm.flip(A)
    var np_flipped_A = np.flip(np_A)
    check_is_close(flipped_A, np_flipped_A, "Flip operation")

    # Test reshape and ravel
    var B = nm.random.randn(2, 3, 4)
    var Bnp = B.to_numpy()
    check_is_close(
        nm.reshape(B, Shape(4, 3, 2), "C"),
        np.reshape(Bnp, (4, 3, 2), "C"),
        "`reshape` by C is broken",
    )
    check_is_close(
        nm.reshape(B, Shape(4, 3, 2), "F"),
        np.reshape(Bnp, (4, 3, 2), "F"),
        "`reshape` by F is broken",
    )

    check_is_close(
        nm.ravel(B, "C"), np.ravel(Bnp, "C"), "`ravel` by C is broken"
    )
    check_is_close(
        nm.ravel(B, "F"), np.ravel(Bnp, "F"), "`ravel` by F is broken"
    )


def test_transpose():
    var np = Python.import_module("numpy")
    var A = nm.random.randn(2)
    var Anp = A.to_numpy()
    check_is_close(
        nm.transpose(A), np.transpose(Anp), "1-d `transpose` is broken."
    )
    A = nm.random.randn(2, 3)
    Anp = A.to_numpy()
    check_is_close(
        nm.transpose(A), np.transpose(Anp), "2-d `transpose` is broken."
    )
    A = nm.random.randn(2, 3, 4)
    Anp = A.to_numpy()
    check_is_close(
        nm.transpose(A), np.transpose(Anp), "3-d `transpose` is broken."
    )
    A = nm.random.randn(2, 3, 4, 5)
    Anp = A.to_numpy()
    check_is_close(
        nm.transpose(A), np.transpose(Anp), "4-d `transpose` is broken."
    )
    check_is_close(
        A.T(), np.transpose(Anp), "4-d `transpose` with `.T` is broken."
    )
    check_is_close(
        nm.transpose(A, axes=List(1, 3, 0, 2)),
        np.transpose(Anp, [1, 3, 0, 2]),
        "4-d `transpose` with arbitrary `axes` is broken.",
    )


def test_setitem():
    var np = Python.import_module("numpy")
    var arr = nm.NDArray(Shape(4, 4))
    var np_arr = arr.to_numpy()
    arr.itemset(List(2, 2), 1000)
    np_arr[(2, 2)] = 1000
    check_is_close(arr, np_arr, "Itemset is broken")
