from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_true

import numojo as nm
from numojo.prelude import *


def test_nonzero_1d() raises:
    """Nonzero on a 1-D array returns flat positions of non-zero entries."""
    var a = nm.array[nm.i32]("[3, 0, 5, 0, 2]")
    var idx = a.nonzero()
    assert_equal(len(idx), 1)
    assert_equal(idx[0].size, 3)
    assert_equal(Int(idx[0].item(0)), 0)
    assert_equal(Int(idx[0].item(1)), 2)
    assert_equal(Int(idx[0].item(2)), 4)


def test_nonzero_2d() raises:
    """Nonzero on a 2-D array returns one coordinate array per dimension."""
    var b = nm.array[nm.i32]("[[1, 0], [0, 4]]")
    var idx2 = b.nonzero()
    assert_equal(len(idx2), 2)
    assert_equal(Int(idx2[0].item(0)), 0)
    assert_equal(Int(idx2[1].item(0)), 0)
    assert_equal(Int(idx2[0].item(1)), 1)
    assert_equal(Int(idx2[1].item(1)), 1)


def test_nonzero_all_zero_returns_empty_indices() raises:
    """Nonzero returns empty index arrays when no elements are non-zero."""
    var a = nm.zeros[nm.i32](Shape(3))
    var idx = a.nonzero()
    assert_equal(len(idx), 1)
    assert_equal(idx[0].size, 0)


def test_nonzero_all_zero_2d_returns_empty_per_dim() raises:
    """Nonzero returns one empty index array per dimension."""
    var a = nm.zeros[nm.i32](Shape(2, 3))
    var idx = a.nonzero()
    assert_equal(len(idx), 2)
    assert_equal(idx[0].size, 0)
    assert_equal(idx[1].size, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
