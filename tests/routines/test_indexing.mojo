# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Test indexing module `nm.routines.indexing`.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.python import Python
from std.testing import TestSuite
from std.testing.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
import numojo as nm
from numojo.prelude import *
from utils_for_test import check, check_is_close


def test_compress() raises:
    var np = Python.import_module("numpy")
    var a = nm.arange[i8](24).reshape(Shape(2, 3, 4))
    var anp = a.to_numpy()
    var b = nm.arange[i8](6)
    var bnp = b.to_numpy()

    check(
        nm.indexing.compress(nm.array[boolean]("[0, 1, 1, 0]"), b),
        np.compress(np.array(Python.list(0, 1, 1, 0)), bnp),
        "`compress` 1-d array is broken",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[0,1,1,0,1,0,1,0,1,1,1]"), a),
        np.compress(
            np.array(Python.list(0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1)), anp
        ),
        "`compress` 3-d array without `axis` is broken",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[0, 1]"), a, axis=0),
        np.compress(np.array(Python.list(0, 1)), anp, axis=0),
        "`compress` 3-d array by axis 0 is broken",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[1]"), a, axis=0),
        np.compress(np.array(Python.list(1)), anp, axis=0),
        "`compress` 3-d array by axis 0 is broken",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[1, 0, 1]"), a, axis=1),
        np.compress(np.array(Python.list(1, 0, 1)), anp, axis=1),
        "`compress` 3-d array by axis 1 is broken (#1)",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[0, 1]"), a, axis=1),
        np.compress(np.array(Python.list(0, 1)), anp, axis=1),
        "`compress` 3-d array by axis 1 is broken (#2)",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[1, 0, 1, 1]"), a, axis=2),
        np.compress(np.array(Python.list(1, 0, 1, 1)), anp, axis=2),
        "`compress` 3-d array by axis 2 is broken",
    )
    check(
        nm.indexing.compress(nm.array[boolean]("[0, 1]"), a, axis=2),
        np.compress(np.array(Python.list(0, 1)), anp, axis=2),
        "`compress` 3-d array by axis 2 is broken",
    )


def test_compress_all_false_returns_empty() raises:
    var a = nm.arange[i8](24).reshape(Shape(2, 3, 4))

    var flat = nm.indexing.compress(
        nm.array[boolean]("[0, 0, 0, 0]"), nm.arange[i8](4)
    )
    assert_equal(flat.ndim, 1)
    assert_equal(flat.shape[0], 0)
    assert_equal(flat.size, 0)

    var axis0 = nm.indexing.compress(nm.array[boolean]("[0, 0]"), a, axis=0)
    assert_equal(axis0.ndim, 3)
    assert_equal(axis0.shape[0], 0)
    assert_equal(axis0.shape[1], 3)
    assert_equal(axis0.shape[2], 4)
    assert_equal(axis0.size, 0)

    var axis1 = nm.indexing.compress(nm.array[boolean]("[0, 0, 0]"), a, axis=1)
    assert_equal(axis1.ndim, 3)
    assert_equal(axis1.shape[0], 2)
    assert_equal(axis1.shape[1], 0)
    assert_equal(axis1.shape[2], 4)
    assert_equal(axis1.size, 0)

    var axis2 = nm.indexing.compress(
        nm.array[boolean]("[0, 0, 0, 0]"), a, axis=2
    )
    assert_equal(axis2.ndim, 3)
    assert_equal(axis2.shape[0], 2)
    assert_equal(axis2.shape[1], 3)
    assert_equal(axis2.shape[2], 0)
    assert_equal(axis2.size, 0)


def test_take_along_axis() raises:
    var np = Python.import_module("numpy")

    # Test 1-D array
    var a1d = nm.arange[i8](10)
    var a1d_np = a1d.to_numpy()
    var indices1d = nm.array[int]("[2, 3, 1, 8]")
    var indices1d_np = indices1d.to_numpy()

    check(
        nm.indexing.take_along_axis(a1d, indices1d, axis=0),
        np.take_along_axis(a1d_np, indices1d_np, axis=0),
        "`take_along_axis` with 1-D array is broken",
    )

    # Test 2-D array with axis=0
    var a2d = nm.arange[i8](12).reshape(Shape(3, 4))
    var a2d_np = a2d.to_numpy()
    var indices2d_0 = nm.array[int]("[[0, 1, 2, 0], [1, 2, 0, 1]]")
    var indices2d_0_np = indices2d_0.to_numpy()

    check(
        nm.indexing.take_along_axis(a2d, indices2d_0, axis=0),
        np.take_along_axis(a2d_np, indices2d_0_np, axis=0),
        "`take_along_axis` with 2-D array on axis=0 is broken",
    )

    # Test 2-D array with axis=1
    var indices2d_1 = nm.array[int](
        "[[3, 0, 2, 1], [1, 3, 0, 0], [2, 1, 0, 3]]"
    )
    var indices2d_1_np = indices2d_1.to_numpy()

    check(
        nm.indexing.take_along_axis(a2d, indices2d_1, axis=1),
        np.take_along_axis(a2d_np, indices2d_1_np, axis=1),
        "`take_along_axis` with 2-D array on axis=1 is broken",
    )

    # Test 3-D array
    var a3d = nm.arange[i8](24).reshape(Shape(2, 3, 4))
    var a3d_np = a3d.to_numpy()

    # Test with axis=0
    var indices3d_0 = nm.zeros[int](Shape(1, 3, 4))
    var indices3d_0_np = indices3d_0.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d, indices3d_0, axis=0),
        np.take_along_axis(a3d_np, indices3d_0_np, axis=0),
        "`take_along_axis` with 3-D array on axis=0 is broken",
    )

    # Test with axis=1
    var indices3d_1 = nm.array[int](
        "[[[0, 1, 0, 2], [2, 1, 0, 1], [1, 2, 2, 0]], [[1, 0, 1, 2], [0, 2, 1,"
        " 0], [2, 0, 0, 1]]]"
    )
    var indices3d_1_np = indices3d_1.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d, indices3d_1, axis=1),
        np.take_along_axis(a3d_np, indices3d_1_np, axis=1),
        "`take_along_axis` with 3-D array on axis=1 is broken",
    )

    # Test with axis=2
    var indices3d_2 = nm.array[int](
        "[[[2, 0, 3, 1], [1, 3, 0, 2], [3, 1, 2, 0]], [[0, 2, 1, 3], [2, 0, 3,"
        " 1], [1, 3, 0, 2]]]"
    )
    var indices3d_2_np = indices3d_2.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d, indices3d_2, axis=2),
        np.take_along_axis(a3d_np, indices3d_2_np, axis=2),
        "`take_along_axis` with 3-D array on axis=2 is broken",
    )

    # Test with negative axis
    check(
        nm.indexing.take_along_axis(a3d, indices3d_2, axis=-1),
        np.take_along_axis(a3d_np, indices3d_2_np, axis=-1),
        "`take_along_axis` with negative axis is broken",
    )

    # Test cases where indices shape matches array shape except on the target axis

    # For 2D array (3, 4)
    var a2d_test = nm.arange[i8](12).reshape(Shape(3, 4))
    var a2d_test_np = a2d_test.to_numpy()

    # For axis=0, using indices of shape (2, 4) - different first dim, same second dim
    var indices2d_axis0 = nm.array[int]("[[0, 1, 2, 0], [1, 0, 2, 1]]")
    var indices2d_axis0_np = indices2d_axis0.to_numpy()

    check(
        nm.indexing.take_along_axis(a2d_test, indices2d_axis0, axis=0),
        np.take_along_axis(a2d_test_np, indices2d_axis0_np, axis=0),
        (
            "`take_along_axis` with shape-aligned indices on axis=0 for 2D"
            " array is broken"
        ),
    )

    # For axis=1, using indices of shape (3, 2) - same first dim, different second dim
    var indices2d_axis1 = nm.array[int]("[[0, 3], [2, 1], [1, 3]]")
    var indices2d_axis1_np = indices2d_axis1.to_numpy()

    check(
        nm.indexing.take_along_axis(a2d_test, indices2d_axis1, axis=1),
        np.take_along_axis(a2d_test_np, indices2d_axis1_np, axis=1),
        (
            "`take_along_axis` with shape-aligned indices on axis=1 for 2D"
            " array is broken"
        ),
    )

    # For 3D array (2, 3, 4)
    # Reshape and get base numpy array
    var a3d_test = nm.arange[i8](24).reshape(Shape(2, 3, 4))
    var a3d_test_np = a3d_test.to_numpy()

    # For axis=0, indices of shape (1, 3, 4) - same shape except dim 0
    var ind_axis0 = nm.zeros[int](Shape(1, 3, 4))
    var ind_axis0_np = ind_axis0.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_test, ind_axis0, axis=0),
        np.take_along_axis(a3d_test_np, ind_axis0_np, axis=0),
        (
            "`take_along_axis` with shape-aligned indices on axis=0 for 3D"
            " array is broken"
        ),
    )

    # For axis=2, indices of shape (2, 3, 2) - same shape except dim 2
    var ind_axis2 = nm.array[int](
        "[[[0, 3], [2, 1], [3, 0]], [[1, 2], [0, 3], [2, 1]]]"
    )
    var ind_axis2_np = ind_axis2.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_test, ind_axis2, axis=2),
        np.take_along_axis(a3d_test_np, ind_axis2_np, axis=2),
        (
            "`take_along_axis` with shape-aligned indices on axis=2 for 3D"
            " array is broken"
        ),
    )


def test_take_along_axis_fortran_order() raises:
    var np = Python.import_module("numpy")

    # Create 3-D F-order array
    var a3d_f = nm.arange[i8](24).reshape(Shape(2, 3, 4), order="F")
    var a3d_f_np = a3d_f.to_numpy()

    # Test with axis=0
    var indices3d_0 = nm.zeros[int](Shape(1, 3, 4))
    var indices3d_0_np = indices3d_0.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_f, indices3d_0, axis=0),
        np.take_along_axis(a3d_f_np, indices3d_0_np, axis=0),
        "`take_along_axis` with 3-D F-order array on axis=0 is broken",
    )

    # Test with axis=1
    var indices3d_1 = nm.array[int](
        "[[[0, 1, 0, 2], [2, 1, 0, 1], [1, 2, 2, 0]], [[1, 0, 1, 2], [0, 2, 1,"
        " 0], [2, 0, 0, 1]]]"
    )
    var indices3d_1_np = indices3d_1.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_f, indices3d_1, axis=1),
        np.take_along_axis(a3d_f_np, indices3d_1_np, axis=1),
        "`take_along_axis` with 3-D F-order array on axis=1 is broken",
    )

    # Test with axis=2
    var indices3d_2 = nm.array[int](
        "[[[2, 0, 3, 1], [1, 3, 0, 2], [3, 1, 2, 0]], [[0, 2, 1, 3], [2, 0, 3,"
        " 1], [1, 3, 0, 2]]]"
    )
    var indices3d_2_np = indices3d_2.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_f, indices3d_2, axis=2),
        np.take_along_axis(a3d_f_np, indices3d_2_np, axis=2),
        "`take_along_axis` with 3-D F-order array on axis=2 is broken",
    )

    # Test with argsort use case on each axis
    var sorted_indices_0 = nm.argsort(a3d_f, axis=0)
    var sorted_indices_0_np = sorted_indices_0.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_f, sorted_indices_0, axis=0),
        np.take_along_axis(a3d_f_np, sorted_indices_0_np, axis=0),
        (
            "`take_along_axis` with argsorted indices on axis=0 for F-order"
            " array is broken"
        ),
    )

    var sorted_indices_1 = nm.argsort(a3d_f, axis=1)
    var sorted_indices_1_np = sorted_indices_1.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_f, sorted_indices_1, axis=1),
        np.take_along_axis(a3d_f_np, sorted_indices_1_np, axis=1),
        (
            "`take_along_axis` with argsorted indices on axis=1 for F-order"
            " array is broken"
        ),
    )

    var sorted_indices_2 = nm.argsort(a3d_f, axis=2)
    var sorted_indices_2_np = sorted_indices_2.to_numpy()

    check(
        nm.indexing.take_along_axis(a3d_f, sorted_indices_2, axis=2),
        np.take_along_axis(a3d_f_np, sorted_indices_2_np, axis=2),
        (
            "`take_along_axis` with argsorted indices on axis=2 for F-order"
            " array is broken"
        ),
    )


# ===----------------------------------------------------------------------=== #
# Index generators
# ===----------------------------------------------------------------------=== #


def test_indices() raises:
    var np = Python.import_module("numpy")

    var dims: List[Int] = [2, 3]
    var grid = nm.indices(dims)
    var np_grid = np.indices(Python.list(2, 3))

    check(grid, np_grid, "`indices` is broken for (2, 3)")

    var dims3: List[Int] = [2, 3, 4]
    var grid3 = nm.indices(dims3)
    var np_grid3 = np.indices(Python.list(2, 3, 4))

    check(grid3, np_grid3, "`indices` is broken for (2, 3, 4)")


def test_ix_() raises:
    var np = Python.import_module("numpy")

    var rows = nm.array[int]("[0, 1, 2]")
    var cols = nm.array[int]("[0, 1]")
    var rows_np = rows.to_numpy()
    var cols_np = cols.to_numpy()

    var arrs = List[NDArray[int]]()
    arrs.append(rows.copy())
    arrs.append(cols.copy())

    var mesh = nm.indexing.ix_(arrs)
    var np_mesh = np.ix_(rows_np, cols_np)

    check(mesh[0], np_mesh[0], "`ix_` is broken for first array")
    check(mesh[1], np_mesh[1], "`ix_` is broken for second array")


def test_diag_indices() raises:
    var np = Python.import_module("numpy")

    var idx = nm.indexing.diag_indices(4)
    var np_idx = np.diag_indices(4)

    check(idx[0], np_idx[0], "`diag_indices` is broken for rows")
    check(idx[1], np_idx[1], "`diag_indices` is broken for cols")

    var idx3 = nm.indexing.diag_indices(4, ndim=3)
    var np_idx3 = np.diag_indices(4, ndim=3)
    check(idx3[0], np_idx3[0], "`diag_indices` is broken for ndim=3 (0)")
    check(idx3[2], np_idx3[2], "`diag_indices` is broken for ndim=3 (2)")


def test_diag_indices_from() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[i32](16).reshape(Shape(4, 4))
    var anp = a.to_numpy()

    var idx = nm.indexing.diag_indices_from(a)
    var np_idx = np.diag_indices_from(anp)

    check(idx[0], np_idx[0], "`diag_indices_from` is broken for rows")
    check(idx[1], np_idx[1], "`diag_indices_from` is broken for cols")

    with assert_raises():
        var b = nm.arange[i32](12).reshape(Shape(3, 4))
        _ = nm.indexing.diag_indices_from(b)


def test_tril_triu_indices() raises:
    var np = Python.import_module("numpy")

    for k in range(-2, 3):
        var rc = nm.indexing.tril_indices(4, k)
        var np_rc = np.tril_indices(4, k)
        check(
            rc[0],
            np_rc[0],
            String("`tril_indices` rows broken for k={}").format(k),
        )
        check(
            rc[1],
            np_rc[1],
            String("`tril_indices` cols broken for k={}").format(k),
        )

        var ru = nm.indexing.triu_indices(4, k)
        var np_ru = np.triu_indices(4, k)
        check(
            ru[0],
            np_ru[0],
            String("`triu_indices` rows broken for k={}").format(k),
        )
        check(
            ru[1],
            np_ru[1],
            String("`triu_indices` cols broken for k={}").format(k),
        )

    var rc_nm = nm.indexing.tril_indices(3, 0, 5)
    var np_rc_nm = np.tril_indices(3, 0, 5)
    check(rc_nm[0], np_rc_nm[0], "`tril_indices` rows broken for n!=m")
    check(rc_nm[1], np_rc_nm[1], "`tril_indices` cols broken for n!=m")


def test_tril_triu_indices_from() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[i32](12).reshape(Shape(3, 4))
    var anp = a.to_numpy()

    var rc = nm.indexing.tril_indices_from(a, 1)
    var np_rc = np.tril_indices_from(anp, 1)
    check(rc[0], np_rc[0], "`tril_indices_from` rows broken")
    check(rc[1], np_rc[1], "`tril_indices_from` cols broken")

    var ru = nm.indexing.triu_indices_from(a, -1)
    var np_ru = np.triu_indices_from(anp, -1)
    check(ru[0], np_ru[0], "`triu_indices_from` rows broken")
    check(ru[1], np_ru[1], "`triu_indices_from` cols broken")


def test_mask_indices() raises:
    var np = Python.import_module("numpy")

    var rc = nm.indexing.mask_indices(4, "triu")
    var np_rc = np.mask_indices(4, np.triu)
    check(rc[0], np_rc[0], "`mask_indices` triu rows broken")
    check(rc[1], np_rc[1], "`mask_indices` triu cols broken")

    var rc2 = nm.indexing.mask_indices(4, "tril", -1)
    var np_rc2 = np.mask_indices(4, np.tril, -1)
    check(rc2[0], np_rc2[0], "`mask_indices` tril rows broken")
    check(rc2[1], np_rc2[1], "`mask_indices` tril cols broken")

    with assert_raises():
        _ = nm.indexing.mask_indices(4, "bogus")


def test_argwhere() raises:
    var np = Python.import_module("numpy")

    var a = nm.array[i32]("[[1, 0, 3], [0, 0, 6]]")
    var anp = a.to_numpy()

    check(
        nm.indexing.argwhere(a), np.argwhere(anp), "`argwhere` is broken"
    )


def test_fill_diagonal() raises:
    var np = Python.import_module("numpy")

    var a = nm.zeros[i32](Shape(3, 3))
    var anp = a.to_numpy()
    nm.indexing.fill_diagonal(a, Scalar[i32](7))
    np.fill_diagonal(anp, 7)
    check(a, anp, "`fill_diagonal` is broken for square 2-D")

    var b = nm.zeros[i32](Shape(4, 3))
    var bnp = b.to_numpy()
    nm.indexing.fill_diagonal(b, Scalar[i32](9), wrap=True)
    np.fill_diagonal(bnp, 9, wrap=True)
    check(b, bnp, "`fill_diagonal` is broken for tall 2-D with wrap")

    var c = nm.zeros[i32](Shape(4, 3))
    var cnp = c.to_numpy()
    nm.indexing.fill_diagonal(c, Scalar[i32](9), wrap=False)
    np.fill_diagonal(cnp, 9, wrap=False)
    check(c, cnp, "`fill_diagonal` is broken for tall 2-D without wrap")

    var d = nm.zeros[i32](Shape(3, 3, 3))
    var dnp = d.to_numpy()
    nm.indexing.fill_diagonal(d, Scalar[i32](5))
    np.fill_diagonal(dnp, 5)
    check(d, dnp, "`fill_diagonal` is broken for 3-D")

    with assert_raises():
        var e = nm.zeros[i32](Shape(3))
        nm.indexing.fill_diagonal(e, Scalar[i32](1))


# ===----------------------------------------------------------------------=== #
# Insertion / extraction helpers
# ===----------------------------------------------------------------------=== #


def test_insert() raises:
    var np = Python.import_module("numpy")

    var a = nm.array[i32]("[1, 2, 3, 4]")
    var anp = a.to_numpy()
    var v = nm.array[i32]("[9, 10]")
    var vnp = v.to_numpy()

    check(
        nm.indexing.insert(a, 2, v),
        np.insert(anp, 2, vnp),
        "`insert` is broken for flattened insertion",
    )

    var a2 = nm.arange[i32](12).reshape(Shape(3, 4))
    var a2np = a2.to_numpy()
    var v2 = nm.array[i32]("[[90, 91, 92, 93]]")
    var v2np = v2.to_numpy()

    check(
        nm.indexing.insert(a2, 1, v2, axis=0),
        np.insert(a2np, 1, v2np, axis=0),
        "`insert` is broken for axis=0",
    )


def test_place() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[i32](6)
    var anp = a.to_numpy()
    var mask = nm.array[boolean]("[1, 0, 1, 0, 1, 0]")
    var masknp = mask.to_numpy()
    var vals = nm.array[i32]("[10, 20]")
    var valsnp = vals.to_numpy()

    np.place(anp, masknp, valsnp)
    nm.indexing.place(a, mask, vals)
    check(a, anp, "`place` is broken")


def test_extract() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[i32](6)
    var anp = a.to_numpy()
    var mask = nm.array[boolean]("[1, 0, 1, 0, 1, 0]")
    var masknp = mask.to_numpy()

    check(
        nm.indexing.extract(mask, a),
        np.extract(masknp, anp),
        "`extract` is broken",
    )


def test_put_along_axis() raises:
    var np = Python.import_module("numpy")

    var a = nm.zeros[i32](Shape(2, 3))
    var anp = a.to_numpy()
    var ind = nm.array[int]("[[0], [2]]")
    var indnp = ind.to_numpy()
    var vals = nm.array[i32]("[[9], [8]]")
    var valsnp = vals.to_numpy()

    np.put_along_axis(anp, indnp, valsnp, axis=1)
    nm.indexing.put_along_axis(a, ind, vals, axis=1)
    check(a, anp, "`put_along_axis` is broken")


def test_putmask() raises:
    var np = Python.import_module("numpy")

    var a = nm.arange[i32](6)
    var anp = a.to_numpy()
    var mask = nm.array[boolean]("[1, 0, 1, 0, 1, 0]")
    var masknp = mask.to_numpy()
    var vals = nm.array[i32]("[10, 20]")
    var valsnp = vals.to_numpy()

    np.putmask(anp, masknp, valsnp)
    nm.indexing.putmask(a, mask, vals)
    check(a, anp, "`putmask` is broken")


def test_choose() raises:
    var np = Python.import_module("numpy")

    var a = nm.array[int]("[0, 1, 0, 2]")
    var anp = a.to_numpy()
    var c0 = nm.array[i32]("[1, 1, 1, 1]")
    var c1 = nm.array[i32]("[9, 9, 9, 9]")
    var c2 = nm.array[i32]("[5, 5, 5, 5]")
    var c0np = c0.to_numpy()
    var c1np = c1.to_numpy()
    var c2np = c2.to_numpy()

    var choices = List[NDArray[i32]]()
    choices.append(c0.copy())
    choices.append(c1.copy())
    choices.append(c2.copy())

    check(
        nm.indexing.choose(a, choices),
        np.choose(anp, [c0np, c1np, c2np]),
        "`choose` is broken",
    )

    with assert_raises():
        var empty_choices = List[NDArray[i32]]()
        _ = nm.indexing.choose(a, empty_choices)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
