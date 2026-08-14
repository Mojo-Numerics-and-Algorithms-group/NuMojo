from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_true

import numojo as nm
from numojo.prelude import *


def test_diagonal_2d_main() raises:
    """2-D diagonal, default offset=0, matches the legacy 2-D behavior."""
    var a = nm.arange[nm.i32](0, 9).reshape(Shape(3, 3))
    var d = a.diagonal()
    assert_equal(d.ndim, 1)
    assert_equal(d.size, 3)
    assert_equal(Int(d.item(0)), 0)
    assert_equal(Int(d.item(1)), 4)
    assert_equal(Int(d.item(2)), 8)


def test_diagonal_2d_offset() raises:
    """2-D diagonal with positive and negative offsets."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var d_pos = a.diagonal(offset=1)
    assert_equal(d_pos.size, 3)
    assert_equal(Int(d_pos.item(0)), 1)
    assert_equal(Int(d_pos.item(1)), 6)
    assert_equal(Int(d_pos.item(2)), 11)

    var d_neg = a.diagonal(offset=-1)
    assert_equal(d_neg.size, 2)
    assert_equal(Int(d_neg.item(0)), 4)
    assert_equal(Int(d_neg.item(1)), 9)


def test_diagonal_3d_default_axes() raises:
    """3-D diagonal with axis1=0, axis2=1: result shape (5, min(3,4))=(5,3)."""
    # a[i,j,k] = i*20 + j*5 + k, shape (3, 4, 5)
    var a = nm.arange[nm.i32](0, 60).reshape(Shape(3, 4, 5))
    var d = a.diagonal(axis1=0, axis2=1)
    assert_equal(d.ndim, 2)
    assert_equal(d.shape[0], 5)
    assert_equal(d.shape[1], 3)
    # d[k, i] = a[i, i, k] = i*20 + i*5 + k = 25*i + k
    for k in range(5):
        for i in range(3):
            assert_equal(Int(d.item(k, i)), 25 * i + k)


def test_diagonal_3d_axis1_axis2() raises:
    """3-D diagonal with axis1=1, axis2=2: result shape (3, min(4,5))=(3,4)."""
    # a[i,j,k] = i*20 + j*5 + k, shape (3, 4, 5)
    var a = nm.arange[nm.i32](0, 60).reshape(Shape(3, 4, 5))
    var d = a.diagonal(axis1=1, axis2=2)
    assert_equal(d.ndim, 2)
    assert_equal(d.shape[0], 3)
    assert_equal(d.shape[1], 4)
    # d[i, j] = a[i, j, j] = i*20 + j*5 + j = i*20 + 6*j
    for i in range(3):
        for j in range(4):
            assert_equal(Int(d.item(i, j)), i * 20 + 6 * j)


def test_diagonal_not_2d_no_longer_errors_for_3d() raises:
    """A 3-D array no longer raises (previous 2-D-only restriction lifted)."""
    var a = nm.arange[nm.i32](0, 60).reshape(Shape(3, 4, 5))
    var d = a.diagonal()
    assert_equal(d.ndim, 2)


def test_diagonal_1d_raises() raises:
    """Diagonal on a 1-D array raises (fewer than 2 dims)."""
    var a = nm.arange[nm.i32](0, 5)
    var raised = False
    try:
        var _d = a.diagonal()
    except:
        raised = True
    assert_true(raised, "1-D diagonal should raise")


def test_diagonal_same_axis_raises() raises:
    """Diagonal with axis1 == axis2 raises."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var raised = False
    try:
        var _d = a.diagonal(axis1=0, axis2=0)
    except:
        raised = True
    assert_true(raised, "axis1 == axis2 should raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
