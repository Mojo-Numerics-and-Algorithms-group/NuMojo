from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_true

import numojo as nm
from numojo.prelude import *


def test_searchsorted_scalar_left() raises:
    """Searchsorted with a scalar value, default side='left'."""
    var a = nm.array[nm.i32]("[1, 3, 5, 7]")
    assert_equal(a.searchsorted(Scalar[nm.i32](4)), 2)
    assert_equal(a.searchsorted(Scalar[nm.i32](1)), 0)
    assert_equal(a.searchsorted(Scalar[nm.i32](8)), 4)
    assert_equal(a.searchsorted(Scalar[nm.i32](0)), 0)


def test_searchsorted_scalar_right() raises:
    """Searchsorted with side='right' returns the rightmost insertion point."""
    var a = nm.array[nm.i32]("[1, 3, 5, 7]")
    assert_equal(a.searchsorted(Scalar[nm.i32](3), side="right"), 2)
    assert_equal(a.searchsorted(Scalar[nm.i32](3), side="left"), 1)


def test_searchsorted_array_values() raises:
    """Searchsorted with an array of values returns one index per value."""
    var a = nm.array[nm.i32]("[1, 3, 5, 7]")
    var result = a.searchsorted(nm.array[nm.i32]("[2, 6]"))
    assert_equal(result.size, 2)
    assert_equal(Int(result.item(0)), 1)
    assert_equal(Int(result.item(1)), 3)


def test_searchsorted_duplicates_left_right() raises:
    """Searchsorted handles duplicate values correctly for both sides."""
    var a = nm.array[nm.i32]("[1, 2, 2, 2, 5]")
    assert_equal(a.searchsorted(Scalar[nm.i32](2), side="left"), 1)
    assert_equal(a.searchsorted(Scalar[nm.i32](2), side="right"), 4)


def test_searchsorted_non_1d_raises() raises:
    """Searchsorted raises when `self` is not 1-D."""
    var a = nm.arange[nm.i32](0, 6).reshape(Shape(2, 3))
    var raised = False
    try:
        var _r = a.searchsorted(Scalar[nm.i32](2))
    except:
        raised = True
    assert_true(raised, "searchsorted on non-1D array should raise")


def test_searchsorted_invalid_side_raises() raises:
    """Searchsorted raises for an invalid `side` argument."""
    var a = nm.array[nm.i32]("[1, 3, 5, 7]")
    var raised = False
    try:
        var _r = a.searchsorted(Scalar[nm.i32](2), side="middle")
    except:
        raised = True
    assert_true(raised, "invalid side value should raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
