from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_true

import numojo as nm
from numojo.prelude import *


def test_take_along_axis_method_axis0() raises:
    """a.take_along_axis(indices, axis) delegates to the routine correctly."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var ind = nm.array[nm.int](
        "[[0, 1, 2, 0], [1, 0, 2, 1]]"
    ).reshape(Shape(2, 4))
    var result = a.take_along_axis(ind, axis=0)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 4)
    # column 0: take rows [0,1] -> a[0,0]=0, a[1,0]=4
    assert_equal(Int(result.item(0, 0)), 0)
    assert_equal(Int(result.item(1, 0)), 4)


def test_take_along_axis_method_default_axis() raises:
    """Default axis is 0."""
    var a = nm.arange[nm.i32](0, 6).reshape(Shape(3, 2))
    var ind = nm.array[nm.int]("[[0, 0], [1, 1], [2, 2]]")
    var result = a.take_along_axis(ind)
    assert_equal(result.shape[0], 3)
    assert_equal(result.shape[1], 2)
    assert_equal(Int(result.item(0, 0)), 0)
    assert_equal(Int(result.item(1, 0)), 2)
    assert_equal(Int(result.item(2, 1)), 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
