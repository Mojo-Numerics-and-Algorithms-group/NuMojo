from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_true

import numojo as nm
from numojo.prelude import *


def test_take_axis0() raises:
    """Take along axis 0 selects rows."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var result = a.take(nm.array[nm.int]("[2, 0, 1]"), axis=0)
    assert_equal(result.shape[0], 3)
    assert_equal(result.shape[1], 4)
    assert_equal(Int(result.item(0, 0)), 8)
    assert_equal(Int(result.item(1, 0)), 0)
    assert_equal(Int(result.item(2, 0)), 4)


def test_take_axis1() raises:
    """Take along axis 1 selects columns."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var result = a.take(nm.array[nm.int]("[1, 3]"), axis=1)
    assert_equal(result.shape[0], 3)
    assert_equal(result.shape[1], 2)
    assert_equal(Int(result.item(0, 0)), 1)
    assert_equal(Int(result.item(0, 1)), 3)


def test_take_flat_no_axis() raises:
    """Take without axis flattens first."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var result = a.take(nm.array[nm.int]("[0, 5, 11]"))
    assert_equal(result.ndim, 1)
    assert_equal(Int(result.item(0)), 0)
    assert_equal(Int(result.item(1)), 5)
    assert_equal(Int(result.item(2)), 11)


def test_take_negative_indices() raises:
    """Take normalises negative indices along the axis."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var result = a.take(nm.array[nm.int]("[-1]"), axis=0)
    assert_equal(Int(result.item(0, 0)), 8)


def test_take_out_of_bounds_raises() raises:
    """Take raises for an out-of-bounds index along the axis."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var raised = False
    try:
        var _r = a.take(nm.array[nm.int]("[5]"), axis=0)
    except:
        raised = True
    assert_true(raised, "out-of-bounds take index should raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
