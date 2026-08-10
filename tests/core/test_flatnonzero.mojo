from std.testing import TestSuite
from std.testing.testing import assert_equal

import numojo as nm
from numojo.prelude import *


def test_flatnonzero_1d() raises:
    """Flatnonzero returns flat positions of non-zero entries."""
    var a = nm.array[nm.i32]("[3, 0, 5, 0, 2]")
    var idx = nm.flatnonzero(a)
    assert_equal(idx.ndim, 1)
    assert_equal(idx.size, 3)
    assert_equal(Int(idx.item(0)), 0)
    assert_equal(Int(idx.item(1)), 2)
    assert_equal(Int(idx.item(2)), 4)


def test_flatnonzero_method() raises:
    """NDArray.flatnonzero delegates to the indexing routine."""
    var a = nm.array[nm.i32]("[0, 4, 0, 7]")
    var idx = a.flatnonzero()
    assert_equal(idx.size, 2)
    assert_equal(Int(idx.item(0)), 1)
    assert_equal(Int(idx.item(1)), 3)


def test_flatnonzero_2d_c_order() raises:
    """Flatnonzero reports C-order flattened positions for N-D arrays."""
    var a = nm.array[nm.i32]("[[0, 1, 0], [2, 0, 3]]")
    var idx = nm.flatnonzero(a)
    assert_equal(idx.size, 3)
    assert_equal(Int(idx.item(0)), 1)
    assert_equal(Int(idx.item(1)), 3)
    assert_equal(Int(idx.item(2)), 5)


def test_flatnonzero_bool() raises:
    """Flatnonzero treats True bool values as non-zero."""
    var a = nm.array[boolean]("[[0, 1], [1, 0]]")
    var idx = nm.flatnonzero(a)
    assert_equal(idx.size, 2)
    assert_equal(Int(idx.item(0)), 1)
    assert_equal(Int(idx.item(1)), 2)


def test_flatnonzero_all_zero_returns_empty() raises:
    """Flatnonzero returns an empty 1-D array when nothing is non-zero."""
    var a = nm.zeros[nm.i32](Shape(2, 3))
    var idx = nm.flatnonzero(a)
    assert_equal(idx.ndim, 1)
    assert_equal(idx.shape[0], 0)
    assert_equal(idx.size, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
