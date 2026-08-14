from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_true

import numojo as nm
from numojo.prelude import *


def test_fancy_index_2d_pointwise() raises:
    """Two 1-D index arrays on a 2-D array select pointwise elements."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    # a[0,2] = 2,  a[1,3] = 7,  a[2,0] = 8
    var rows = nm.array[nm.int]("[0, 1, 2]")
    var cols = nm.array[nm.int]("[2, 3, 0]")
    var result = a.fancy_index(rows, cols)
    assert_equal(result.ndim, 1)
    assert_equal(result.shape[0], 3)
    assert_equal(Int(result.item(0)), 2)
    assert_equal(Int(result.item(1)), 7)
    assert_equal(Int(result.item(2)), 8)


def test_fancy_index_2d_free_fn() raises:
    """Free function nm.fancy_index produces the same result as the method."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var rows = nm.array[nm.int]("[0, 1]")
    var cols = nm.array[nm.int]("[2, 3]")
    var r1 = a.fancy_index(rows, cols)
    var r2 = nm.fancy_index(a, rows, cols)
    assert_equal(Int(r1.item(0)), Int(r2.item(0)))
    assert_equal(Int(r1.item(1)), Int(r2.item(1)))


def test_fancy_index_getitem_list_syntax() raises:
    """A[[rows, cols]] subscript syntax via List[NDArray[DType.int]]."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var rows = nm.array[nm.int]("[0, 1, 2]")
    var cols = nm.array[nm.int]("[2, 3, 0]")
    var idx = List[nm.NDArray[DType.int]]()
    idx.append(rows^)
    idx.append(cols^)
    var result = a[idx]
    assert_equal(result.ndim, 1)
    assert_equal(result.shape[0], 3)
    assert_equal(Int(result.item(0)), 2)
    assert_equal(Int(result.item(1)), 7)
    assert_equal(Int(result.item(2)), 8)


def test_fancy_index_2d_grid() raises:
    """2-D index arrays select a grid of elements."""
    # a = [[0,1,2,3],[4,5,6,7]]  shape (2,4)
    var a = nm.arange[nm.f32](0, 8).reshape(Shape(2, 4))
    # row_idx = [[0, 1],[1, 0]], col_idx = [[1, 2],[3, 0]]
    # selected: a[0,1]=1, a[1,2]=6, a[1,3]=7, a[0,0]=0
    var r = nm.array[nm.int]("[[0, 1],[1, 0]]")
    var c = nm.array[nm.int]("[[1, 2],[3, 0]]")
    var result = a.fancy_index(r, c)
    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 2)
    assert_equal(Int(result.item(0, 0)), 1)
    assert_equal(Int(result.item(0, 1)), 6)
    assert_equal(Int(result.item(1, 0)), 7)
    assert_equal(Int(result.item(1, 1)), 0)


def test_fancy_index_negative_indices() raises:
    """Negative indices are normalised correctly."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    # a[-1, -1] = a[2, 3] = 11
    var rows = nm.array[nm.int]("[-1]")
    var cols = nm.array[nm.int]("[-1]")
    var result = a.fancy_index(rows, cols)
    assert_equal(Int(result.item(0)), 11)


def test_fancy_index_3d() raises:
    """Three index arrays on a 3-D source array."""
    # shape (2, 3, 4), values 0..23
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4))
    # a[0, 1, 2] = 0*12 + 1*4 + 2 = 6
    # a[1, 2, 3] = 1*12 + 2*4 + 3 = 23
    var i0 = nm.array[nm.int]("[0, 1]")
    var i1 = nm.array[nm.int]("[1, 2]")
    var i2 = nm.array[nm.int]("[2, 3]")
    var result = a.fancy_index(i0, i1, i2)
    assert_equal(result.shape[0], 2)
    assert_equal(Int(result.item(0)), 6)
    assert_equal(Int(result.item(1)), 23)


def test_fancy_index_broadcast() raises:
    """Index arrays of different shapes broadcast against each other."""
    # a shape (3, 4): use row scalar-like [1] broadcast against col [0,1,2,3]
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    # row [1] broadcasts to [1,1,1,1], col [0,1,2,3]
    # selects a[1,0]=4, a[1,1]=5, a[1,2]=6, a[1,3]=7
    var rows = nm.array[nm.int]("[1]")
    var cols = nm.array[nm.int]("[0, 1, 2, 3]")
    var result = a.fancy_index(rows, cols)
    assert_equal(result.shape[0], 4)
    assert_equal(Int(result.item(0)), 4)
    assert_equal(Int(result.item(1)), 5)
    assert_equal(Int(result.item(2)), 6)
    assert_equal(Int(result.item(3)), 7)


def test_fancy_index_wrong_n_arrays_raises() raises:
    """Passing wrong number of index arrays raises an error."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var raised = False
    try:
        # a is 2-D, but only 1 index array → should raise
        var _r = nm.fancy_index(a, nm.array[nm.int]("[0, 1]"))
    except:
        raised = True
    assert_true(raised, "wrong number of index arrays should raise")


def test_fancy_index_out_of_bounds_raises() raises:
    """An out-of-bounds index raises an error."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var raised = False
    try:
        var rows = nm.array[nm.int]("[5]")  # axis 0 size = 3
        var cols = nm.array[nm.int]("[0]")
        var _r = a.fancy_index(rows, cols)
    except:
        raised = True
    assert_true(raised, "out-of-bounds fancy index should raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
