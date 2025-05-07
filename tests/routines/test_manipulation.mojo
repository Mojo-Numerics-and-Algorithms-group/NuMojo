import numojo as nm
from numojo import *
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close
from python import Python


fn test_arr_manipulation() raises:
    var np = Python.import_module("numpy")

    # Test arange
    var A = nm.arange[nm.i16](1, 7, 1)
    var Anp = np.arange(1, 7, 1, dtype=np.int16)
    check_is_close(A, Anp, "Arange operation")

    var B = nm.random.randn(2, 3, 4)
    var Bnp = B.to_numpy()

    # Test flip
    check_is_close(nm.flip(B), np.flip(Bnp), "`flip` without `axis` fails.")
    for i in range(3):
        check_is_close(
            nm.flip(B, axis=i),
            np.flip(Bnp, axis=i),
            String("`flip` by `axis` {} fails.").format(i),
        )


def test_ravel_reshape():
    var np = Python.import_module("numpy")
    var c = nm.fromstring[i8](
        "[[[1,2,3,4][5,6,7,8]][[9,10,11,12][13,14,15,16]]]", order="C"
    )
    var cnp = c.to_numpy()
    var f = nm.fromstring[i8](
        "[[[1,2,3,4][5,6,7,8]][[9,10,11,12][13,14,15,16]]]", order="F"
    )
    var fnp = f.to_numpy()

    # Test ravel
    check_is_close(
        nm.ravel(c, order="C"),
        np.ravel(cnp, order="C"),
        "`ravel` C-order array by C order is broken.",
    )
    check_is_close(
        nm.ravel(c, order="F"),
        np.ravel(cnp, order="F"),
        "`ravel` C-order array by F order is broken.",
    )
    check_is_close(
        nm.ravel(f, order="C"),
        np.ravel(fnp, order="C"),
        "`ravel` F-order array by C order is broken.",
    )
    check_is_close(
        nm.ravel(f, order="F"),
        np.ravel(fnp, order="F"),
        "`ravel` F-order array by F order is broken.",
    )

    # Test reshape
    check_is_close(
        nm.reshape(c, Shape(4, 2, 2), "C"),
        np.reshape(cnp, Python.tuple(4, 2, 2), "C"),
        "`reshape` C by C is broken",
    )
    check_is_close(
        nm.reshape(c, Shape(4, 2, 2), "F"),
        np.reshape(cnp, Python.tuple(4, 2, 2), "F"),
        "`reshape` C by F is broken",
    )
    check_is_close(
        nm.reshape(f, Shape(4, 2, 2), "C"),
        np.reshape(fnp, Python.tuple(4, 2, 2), "C"),
        "`reshape` F by C is broken",
    )
    check_is_close(
        nm.reshape(f, Shape(4, 2, 2), "F"),
        np.reshape(fnp, Python.tuple(4, 2, 2), "F"),
        "`reshape` F by F is broken",
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
        np.transpose(Anp, Python.list(1, 3, 0, 2)),
        "4-d `transpose` with arbitrary `axes` is broken.",
    )


def test_broadcast():
    var np = Python.import_module("numpy")
    var a = nm.random.rand(Shape(2, 1, 3))
    var Anp = a.to_numpy()
    check(
        nm.broadcast_to(a, Shape(2, 2, 3)),
        np.broadcast_to(a.to_numpy(), Python.tuple(2, 2, 3)),
        "`broadcast_to` fails.",
    )
    check(
        nm.broadcast_to(a, Shape(2, 2, 2, 3)),
        np.broadcast_to(a.to_numpy(), Python.tuple(2, 2, 2, 3)),
        "`broadcast_to` fails.",
    )
