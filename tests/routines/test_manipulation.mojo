from std.testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close
from std.python import Python, PythonObject
from std.testing import TestSuite

import numojo as nm
from numojo import *

def test_arr_manipulation() raises:
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


def test_ravel_reshape() raises:
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
        np.ravel(cnp, order=PythonObject("C")),
        "`ravel` C-order array by C order is broken.",
    )
    check_is_close(
        nm.ravel(c, order="F"),
        np.ravel(cnp, order=PythonObject("F")),
        "`ravel` C-order array by F order is broken.",
    )
    check_is_close(
        nm.ravel(f, order="C"),
        np.ravel(fnp, order=PythonObject("C")),
        "`ravel` F-order array by C order is broken.",
    )
    check_is_close(
        nm.ravel(f, order="F"),
        np.ravel(fnp, order=PythonObject("F")),
        "`ravel` F-order array by F order is broken.",
    )

    # Test reshape
    var reshape_c = nm.reshape(c, Shape(4, 2, 2), "C")
    var reshape_cnp = np.reshape(cnp, Python.tuple(4, 2, 2), "C")
    check_is_close(
        reshape_c,
        reshape_cnp,
        "`reshape` C by C is broken",
    )
    # TODO: This test is breaking, gotta fix reshape.
    var reshape_f = nm.reshape(c, Shape(4, 2, 2), "F")
    var reshape_fnp = np.reshape(cnp, Python.tuple(4, 2, 2), "F")
    check_is_close(
        reshape_f,
        reshape_fnp,
        "`reshape` C by F is broken",
    )
    var reshape_fc = nm.reshape(f, Shape(4, 2, 2), "C")
    var reshape_fcnp = np.reshape(fnp, Python.tuple(4, 2, 2), "C")
    check_is_close(
        reshape_fc,
        reshape_fcnp,
        "`reshape` F by C is broken",
    )
    check_is_close(
        nm.reshape(f, Shape(4, 2, 2), "F"),
        np.reshape(fnp, Python.tuple(4, 2, 2), "F"),
        "`reshape` F by F is broken",
    )


def test_transpose() raises:
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
        nm.transpose(A, axes=[Int(1), 3, 0, 2]),
        np.transpose(Anp, Python.list(1, 3, 0, 2)),
        "4-d `transpose` with arbitrary `axes` is broken.",
    )


def test_broadcast() raises:
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


def test_concatenate() raises:
    var np = Python.import_module("numpy")

    # 1-D concatenation
    var a1 = nm.arange[nm.f64](0, 3, 1)
    var b1 = nm.arange[nm.f64](3, 6, 1)
    var c1 = nm.concatenate(a1, b1, axis=0)
    var c1np = np.concatenate(
        Python.list(a1.to_numpy(), b1.to_numpy()), axis=PythonObject(0)
    )
    check_is_close(c1, c1np, "`concatenate` 1-D along axis=0 fails.")

    # 2-D concatenation along axis=0
    var a2 = nm.reshape(nm.arange[nm.f64](0, 6, 1), Shape(2, 3))
    var b2 = nm.reshape(nm.arange[nm.f64](6, 12, 1), Shape(2, 3))
    var c2 = nm.concatenate(a2, b2, axis=0)
    var c2np = np.concatenate(
        Python.list(a2.to_numpy(), b2.to_numpy()), axis=PythonObject(0)
    )
    check_is_close(c2, c2np, "`concatenate` 2-D along axis=0 fails.")

    # 2-D concatenation along axis=1
    var c3 = nm.concatenate(a2, b2, axis=1)
    var c3np = np.concatenate(
        Python.list(a2.to_numpy(), b2.to_numpy()), axis=PythonObject(1)
    )
    check_is_close(c3, c3np, "`concatenate` 2-D along axis=1 fails.")

    # 3-D concatenation
    var a3 = nm.reshape(nm.arange[nm.f64](0, 24, 1), Shape(2, 3, 4))
    var b3 = nm.reshape(nm.arange[nm.f64](24, 48, 1), Shape(2, 3, 4))
    for ax in range(3):
        var c = nm.concatenate(a3, b3, axis=ax)
        var cnp = np.concatenate(
            Python.list(a3.to_numpy(), b3.to_numpy()), axis=PythonObject(ax)
        )
        check_is_close(
            c,
            cnp,
            String("`concatenate` 3-D along axis={} fails.").format(ax),
        )


def test_column_stack() raises:
    var np = Python.import_module("numpy")

    # Two 1-D arrays -> (N, 2)
    var a = nm.arange[nm.f64](0, 3, 1)
    var b = nm.arange[nm.f64](3, 6, 1)
    var c = nm.column_stack(a, b)
    var cnp = np.column_stack(Python.list(a.to_numpy(), b.to_numpy()))
    check_is_close(c, cnp, "`column_stack` two 1-D arrays fails.")

    # Three 1-D arrays -> (N, 3)
    var d = nm.arange[nm.f64](6, 9, 1)
    var e = nm.column_stack(a, b, d)
    var enp = np.column_stack(
        Python.list(a.to_numpy(), b.to_numpy(), d.to_numpy())
    )
    check_is_close(e, enp, "`column_stack` three 1-D arrays fails.")

    # Two 2-D arrays (like hstack along axis=1)
    var a2 = nm.reshape(nm.arange[nm.f64](0, 6, 1), Shape(2, 3))
    var b2 = nm.reshape(nm.arange[nm.f64](6, 10, 1), Shape(2, 2))
    var f = nm.column_stack(a2, b2)
    var fnp = np.column_stack(Python.list(a2.to_numpy(), b2.to_numpy()))
    check_is_close(f, fnp, "`column_stack` two 2-D arrays fails.")

    # Mix of 1-D and 2-D arrays
    var g1 = nm.arange[nm.f64](0, 3, 1)  # Shape (3,)
    var g2 = nm.reshape(nm.arange[nm.f64](3, 9, 1), Shape(3, 2))  # Shape (3,2)
    var g = nm.column_stack(g1, g2)
    var gnp = np.column_stack(Python.list(g1.to_numpy(), g2.to_numpy()))
    check_is_close(g, gnp, "`column_stack` mix of 1-D and 2-D fails.")


def test_hstack() raises:
    var np = Python.import_module("numpy")

    # 1-D arrays
    var a = nm.arange[nm.f64](0, 3, 1)
    var b = nm.arange[nm.f64](3, 6, 1)
    var c = nm.hstack(a, b)
    var cnp = np.hstack(Python.list(a.to_numpy(), b.to_numpy()))
    check_is_close(c, cnp, "`hstack` 1-D arrays fails.")

    # 2-D arrays
    var a2 = nm.reshape(nm.arange[nm.f64](0, 6, 1), Shape(2, 3))
    var b2 = nm.reshape(nm.arange[nm.f64](6, 10, 1), Shape(2, 2))
    var d = nm.hstack(a2, b2)
    var dnp = np.hstack(Python.list(a2.to_numpy(), b2.to_numpy()))
    check_is_close(d, dnp, "`hstack` 2-D arrays fails.")


def test_vstack() raises:
    var np = Python.import_module("numpy")

    # 1-D arrays -> (2, N)
    var a = nm.arange[nm.f64](0, 3, 1)
    var b = nm.arange[nm.f64](3, 6, 1)
    var c = nm.vstack(a, b)
    var cnp = np.vstack(Python.list(a.to_numpy(), b.to_numpy()))
    check_is_close(c, cnp, "`vstack` 1-D arrays fails.")

    # 2-D arrays
    var a2 = nm.reshape(nm.arange[nm.f64](0, 6, 1), Shape(2, 3))
    var b2 = nm.reshape(nm.arange[nm.f64](6, 12, 1), Shape(2, 3))
    var d = nm.vstack(a2, b2)
    var dnp = np.vstack(Python.list(a2.to_numpy(), b2.to_numpy()))
    check_is_close(d, dnp, "`vstack` 2-D arrays fails.")


def test_row_stack() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[nm.f64](0, 3, 1)
    var b = nm.arange[nm.f64](3, 6, 1)
    var c = nm.row_stack(a, b)
    var cnp = np.row_stack(Python.list(a.to_numpy(), b.to_numpy()))
    check_is_close(c, cnp, "`row_stack` 1-D arrays fails.")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
