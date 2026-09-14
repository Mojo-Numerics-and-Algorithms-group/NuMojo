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


def test_swapaxes() raises:
    var np = Python.import_module("numpy")

    var A = nm.random.randn(2, 3, 4)
    var Anp = A.to_numpy()
    check_is_close(
        nm.swapaxes(A, 0, 1),
        np.swapaxes(Anp, 0, 1),
        "`swapaxes` (0, 1) fails.",
    )
    check_is_close(
        nm.swapaxes(A, 0, 2),
        np.swapaxes(Anp, 0, 2),
        "`swapaxes` (0, 2) fails.",
    )
    check_is_close(
        nm.swapaxes(A, -1, -2),
        np.swapaxes(Anp, -1, -2),
        "`swapaxes` with negative axes fails.",
    )


def test_moveaxis() raises:
    var np = Python.import_module("numpy")

    var A = nm.random.randn(2, 3, 4, 5)
    var Anp = A.to_numpy()
    check_is_close(
        nm.moveaxis(A, 0, -1),
        np.moveaxis(Anp, 0, -1),
        "`moveaxis` single axis fails.",
    )
    var source: List[Int] = [0, 1]
    var destination: List[Int] = [-1, -2]
    check_is_close(
        nm.moveaxis(A, source, destination),
        np.moveaxis(Anp, Python.list(0, 1), Python.list(-1, -2)),
        "`moveaxis` multiple axes fails.",
    )


def test_roll() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[nm.i32](0, 10, 1)
    var anp = a.to_numpy()
    check_is_close(nm.roll(a, 2), np.roll(anp, 2), "`roll` 1-D fails.")
    check_is_close(
        nm.roll(a, -3), np.roll(anp, -3), "`roll` 1-D negative shift fails."
    )

    var B = nm.reshape(nm.arange[nm.i32](0, 24, 1), Shape(2, 3, 4))
    var Bnp = B.to_numpy()
    check_is_close(
        nm.roll(B, 1), np.roll(Bnp, 1), "`roll` without `axis` fails."
    )
    for i in range(3):
        check_is_close(
            nm.roll(B, 2, axis=i),
            np.roll(Bnp, 2, axis=i),
            String("`roll` by `axis` {} fails.").format(i),
        )
    var shifts: List[Int] = [1, -2]
    var axes: List[Int] = [0, 2]
    check_is_close(
        nm.roll(B, shifts, axes),
        np.roll(Bnp, Python.list(1, -2), axis=Python.list(0, 2)),
        "`roll` with multiple axes fails.",
    )


def test_repeat() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[nm.i32](0, 3, 1)
    var anp = a.to_numpy()
    check_is_close(
        nm.repeat(a, 2), np.repeat(anp, 2), "`repeat` 1-D uniform fails."
    )
    var reps: List[Int] = [1, 2, 3]
    check_is_close(
        nm.repeat(a, reps),
        np.repeat(anp, Python.list(1, 2, 3)),
        "`repeat` 1-D variable fails.",
    )

    var B = nm.reshape(nm.arange[nm.i32](0, 6, 1), Shape(2, 3))
    var Bnp = B.to_numpy()
    check_is_close(
        nm.repeat(B, 2, axis=0),
        np.repeat(Bnp, 2, axis=0),
        "`repeat` axis=0 uniform fails.",
    )
    check_is_close(
        nm.repeat(B, 2, axis=1),
        np.repeat(Bnp, 2, axis=1),
        "`repeat` axis=1 uniform fails.",
    )
    var reps2: List[Int] = [1, 2]
    check_is_close(
        nm.repeat(B, reps2, axis=0),
        np.repeat(Bnp, Python.list(1, 2), axis=0),
        "`repeat` axis=0 variable fails.",
    )


def test_pad() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[nm.i32](0, 3, 1)
    var anp = a.to_numpy()
    check_is_close(
        nm.pad(a, 2), np.pad(anp, 2), "`pad` constant (uniform int) fails."
    )

    var width: List[Int] = [1, 2]
    var pad_width = List[List[Int]]()
    pad_width.append(width^)
    check_is_close(
        nm.pad(a, pad_width),
        np.pad(anp, Python.tuple(1, 2)),
        "`pad` constant with [before, after] fails.",
    )
    check_is_close(
        nm.pad(a, pad_width, mode="edge"),
        np.pad(anp, Python.tuple(1, 2), mode="edge"),
        "`pad` edge mode fails.",
    )
    check_is_close(
        nm.pad(a, pad_width, mode="reflect"),
        np.pad(anp, Python.tuple(1, 2), mode="reflect"),
        "`pad` reflect mode fails.",
    )
    check_is_close(
        nm.pad(a, pad_width, mode="symmetric"),
        np.pad(anp, Python.tuple(1, 2), mode="symmetric"),
        "`pad` symmetric mode fails.",
    )
    check_is_close(
        nm.pad(a, pad_width, mode="wrap"),
        np.pad(anp, Python.tuple(1, 2), mode="wrap"),
        "`pad` wrap mode fails.",
    )

    var B = nm.reshape(nm.arange[nm.i32](0, 6, 1), Shape(2, 3))
    var Bnp = B.to_numpy()
    var w0: List[Int] = [1, 1]
    var w1: List[Int] = [0, 2]
    var pad_width_2d = List[List[Int]]()
    pad_width_2d.append(w0^)
    pad_width_2d.append(w1^)
    check_is_close(
        nm.pad(B, pad_width_2d, mode="edge"),
        np.pad(
            Bnp,
            Python.tuple(Python.tuple(1, 1), Python.tuple(0, 2)),
            mode="edge",
        ),
        "`pad` 2-D per-axis edge mode fails.",
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


def test_split() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[nm.i32](0, 6, 1)
    var anp = a.to_numpy()

    var parts = nm.split(a, 3)
    var partsnp = np.split(anp, 3)
    for i in range(3):
        check_is_close(
            parts[i],
            partsnp[i],
            String("`split` by sections, part {} fails.").format(i),
        )

    var indices: List[Int] = [2, 4]
    var parts2 = nm.split(a, indices)
    var parts2np = np.split(anp, Python.list(2, 4))
    for i in range(3):
        check_is_close(
            parts2[i],
            parts2np[i],
            String("`split` by indices, part {} fails.").format(i),
        )

    # Unsorted / negative indices produce the same slices as raw Python
    # slicing, including empty sub-arrays.
    var indices2: List[Int] = [2, -2]
    var parts3 = nm.split(a, indices2)
    var parts3np = np.split(anp, Python.list(2, -2))
    for i in range(3):
        check_is_close(
            parts3[i],
            parts3np[i],
            String("`split` unsorted/negative indices, part {} fails.").format(
                i
            ),
        )

    # 2-D array, split along axis=1
    var B = nm.reshape(nm.arange[nm.i32](0, 12, 1), Shape(3, 4))
    var Bnp = B.to_numpy()
    var partsB = nm.split(B, 2, axis=1)
    var partsBnp = np.split(Bnp, 2, axis=1)
    for i in range(2):
        check_is_close(
            partsB[i],
            partsBnp[i],
            String("`split` 2-D axis=1, part {} fails.").format(i),
        )


def test_array_split() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[nm.i32](0, 7, 1)
    var anp = a.to_numpy()

    var parts = nm.array_split(a, 3)
    var partsnp = np.array_split(anp, 3)
    for i in range(3):
        check_is_close(
            parts[i],
            partsnp[i],
            String("`array_split` uneven sections, part {} fails.").format(i),
        )

    var parts2 = nm.array_split(a, 4)
    var parts2np = np.array_split(anp, 4)
    for i in range(4):
        check_is_close(
            parts2[i],
            parts2np[i],
            String("`array_split` uneven sections (4), part {} fails.").format(
                i
            ),
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
    var cnp = np.vstack(Python.list(a.to_numpy(), b.to_numpy()))
    check_is_close(c, cnp, "`row_stack` 1-D arrays fails.")


def test_delete() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[nm.i32](0, 5, 1)
    var anp = a.to_numpy()
    check_is_close(
        nm.delete(a, 1), np.delete(anp, 1), "`delete` single index fails."
    )
    check_is_close(
        nm.delete(a, -1), np.delete(anp, -1), "`delete` negative index fails."
    )
    var idx: List[Int] = [1, 1, 2]
    check_is_close(
        nm.delete(a, idx),
        np.delete(anp, Python.list(1, 1, 2)),
        "`delete` duplicate indices fails.",
    )

    var B = nm.reshape(nm.arange[nm.i32](0, 12, 1), Shape(3, 4))
    var Bnp = B.to_numpy()
    check_is_close(
        nm.delete(B, 1, axis=0),
        np.delete(Bnp, 1, axis=0),
        "`delete` axis=0 fails.",
    )
    var idx2: List[Int] = [0, 2]
    check_is_close(
        nm.delete(B, idx2, axis=1),
        np.delete(Bnp, Python.list(0, 2), axis=1),
        "`delete` axis=1 fails.",
    )
    check_is_close(
        nm.delete(B, 1),
        np.delete(Bnp, 1),
        "`delete` axis=None (flatten) fails.",
    )


def test_append() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[nm.i32](0, 3, 1)
    var b = nm.arange[nm.i32](3, 6, 1)
    check_is_close(
        nm.append(a, b),
        np.append(a.to_numpy(), b.to_numpy()),
        "`append` axis=None fails.",
    )

    var A = nm.reshape(nm.arange[nm.i32](0, 4, 1), Shape(2, 2))
    var Bv = nm.reshape(nm.arange[nm.i32](4, 6, 1), Shape(1, 2))
    check_is_close(
        nm.append(A, Bv, axis=0),
        np.append(A.to_numpy(), Bv.to_numpy(), axis=0),
        "`append` axis=0 fails.",
    )

    check_is_close(
        nm.append(A, Bv),
        np.append(A.to_numpy(), Bv.to_numpy()),
        "`append` axis=None on 2-D arrays fails.",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
