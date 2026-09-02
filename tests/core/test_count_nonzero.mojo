from std.testing import TestSuite
from std.testing.testing import assert_equal

import numojo as nm
from numojo.prelude import *


def test_count_nonzero_1d() raises:
    """Count_nonzero counts non-zero entries in a 1-D array."""
    var a = nm.array[nm.i32]("[3, 0, 5, 0, 2]")
    assert_equal(nm.count_nonzero(a), 3)


def test_count_nonzero_method() raises:
    """NDArray.count_nonzero delegates to the indexing routine."""
    var a = nm.array[nm.i32]("[0, 4, 0, 7]")
    assert_equal(a.count_nonzero(), 2)


def test_count_nonzero_2d_flat() raises:
    """Count_nonzero without axis counts over the whole array."""
    var a = nm.array[nm.i32]("[[0, 1, 0], [2, 0, 3]]")
    assert_equal(nm.count_nonzero(a), 3)


def test_count_nonzero_bool() raises:
    """Count_nonzero treats True bool values as non-zero."""
    var a = nm.array[boolean]("[[0, 1], [1, 0]]")
    assert_equal(nm.count_nonzero(a), 2)


def test_count_nonzero_all_zero_returns_zero() raises:
    """Count_nonzero returns 0 when nothing is non-zero."""
    var a = nm.zeros[nm.i32](Shape(2, 3))
    assert_equal(nm.count_nonzero(a), 0)


def test_count_nonzero_axis0() raises:
    """Count_nonzero(axis=0) counts non-zero entries down each column."""
    var a = nm.array[nm.i32]("[[1, 0, 3], [0, 0, 4], [1, 1, 0]]")
    var result = nm.count_nonzero(a, axis=0)
    assert_equal(result.ndim, 1)
    assert_equal(Int(result.item(0)), 2)
    assert_equal(Int(result.item(1)), 1)
    assert_equal(Int(result.item(2)), 2)


def test_count_nonzero_axis1() raises:
    """Count_nonzero(axis=1) counts non-zero entries across each row."""
    var a = nm.array[nm.i32]("[[1, 0, 3], [0, 0, 4], [1, 1, 0]]")
    var result = nm.count_nonzero(a, axis=1)
    assert_equal(result.ndim, 1)
    assert_equal(Int(result.item(0)), 2)
    assert_equal(Int(result.item(1)), 1)
    assert_equal(Int(result.item(2)), 2)


def test_count_nonzero_negative_axis() raises:
    """Count_nonzero accepts negative axis indices."""
    var a = nm.array[nm.i32]("[[1, 0, 3], [0, 0, 4], [1, 1, 0]]")
    var result_pos = nm.count_nonzero(a, axis=1)
    var result_neg = nm.count_nonzero(a, axis=-1)
    assert_equal(Int(result_pos.item(0)), Int(result_neg.item(0)))
    assert_equal(Int(result_pos.item(1)), Int(result_neg.item(1)))
    assert_equal(Int(result_pos.item(2)), Int(result_neg.item(2)))


def test_count_nonzero_axis_out_of_bound_raises() raises:
    """Count_nonzero raises when axis is out of bounds."""
    var a = nm.array[nm.i32]("[[1, 0], [0, 4]]")
    var raised = False
    try:
        _ = nm.count_nonzero(a, axis=5)
    except:
        raised = True
    assert_equal(raised, True)


def test_count_nonzero_axis_on_1d_raises() raises:
    """Count_nonzero raises when axis is passed for a 1-D array."""
    var a = nm.array[nm.i32]("[1, 0, 2]")
    var raised = False
    try:
        _ = nm.count_nonzero(a, axis=0)
    except:
        raised = True
    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
