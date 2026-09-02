from std.testing import TestSuite
from std.testing.testing import assert_equal

import numojo as nm
from numojo.prelude import *


def test_flipud_2d() raises:
    """Flipud reverses rows, keeping each row's contents intact."""
    var a = nm.array[nm.i32]("[[1, 2], [3, 4]]")
    var result = nm.flipud(a)
    assert_equal(Int(result.item(0, 0)), 3)
    assert_equal(Int(result.item(0, 1)), 4)
    assert_equal(Int(result.item(1, 0)), 1)
    assert_equal(Int(result.item(1, 1)), 2)


def test_fliplr_2d() raises:
    """Fliplr reverses columns, keeping each column's contents intact."""
    var a = nm.array[nm.i32]("[[1, 2], [3, 4]]")
    var result = nm.fliplr(a)
    assert_equal(Int(result.item(0, 0)), 2)
    assert_equal(Int(result.item(0, 1)), 1)
    assert_equal(Int(result.item(1, 0)), 4)
    assert_equal(Int(result.item(1, 1)), 3)


def test_flipud_1d() raises:
    """Flipud on a 1-D array reverses the whole array."""
    var a = nm.array[nm.i32]("[1, 2, 3, 4]")
    var result = nm.flipud(a)
    assert_equal(Int(result.item(0)), 4)
    assert_equal(Int(result.item(1)), 3)
    assert_equal(Int(result.item(2)), 2)
    assert_equal(Int(result.item(3)), 1)


def test_fliplr_1d_raises() raises:
    """Fliplr on a 1-D array raises (numpy requires at least 2 dimensions)."""
    var a = nm.array[nm.i32]("[1, 2, 3, 4]")
    var raised = False
    try:
        _ = nm.fliplr(a)
    except:
        raised = True
    assert_equal(raised, True)


def test_flipud_preserves_shape() raises:
    """Flipud does not change the array's shape."""
    var a = nm.array[nm.i32]("[[1, 2, 3], [4, 5, 6]]")
    var result = nm.flipud(a)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 3)


def test_fliplr_preserves_shape() raises:
    """Fliplr does not change the array's shape."""
    var a = nm.array[nm.i32]("[[1, 2, 3], [4, 5, 6]]")
    var result = nm.fliplr(a)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 3)


def test_flipud_then_fliplr_matches_flip_both_axes() raises:
    """Flipud+fliplr composed matches flipping both axes independently."""
    var a = nm.array[nm.i32]("[[1, 2, 3], [4, 5, 6]]")
    var composed = nm.fliplr(nm.flipud(a))
    var expected = nm.flip(nm.flip(a, axis=0), axis=1)
    for i in range(2):
        for j in range(3):
            assert_equal(Int(composed.item(i, j)), Int(expected.item(i, j)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
