from std.testing import TestSuite
from std.testing.testing import assert_equal

import numojo as nm
from numojo.prelude import *


def test_expand_dims_axis0() raises:
    """Expand_dims(axis=0) inserts a leading size-1 dimension."""
    var a = nm.array[nm.i32]("[1, 2, 3]")
    var result = nm.expand_dims(a, axis=0)
    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 1)
    assert_equal(result.shape[1], 3)


def test_expand_dims_axis1() raises:
    """Expand_dims(axis=1) inserts a trailing size-1 dimension."""
    var a = nm.array[nm.i32]("[1, 2, 3]")
    var result = nm.expand_dims(a, axis=1)
    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 3)
    assert_equal(result.shape[1], 1)


def test_expand_dims_negative_axis() raises:
    """Expand_dims accepts negative axis indices."""
    var a = nm.array[nm.i32]("[1, 2, 3]")
    var result = nm.expand_dims(a, axis=-1)
    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 3)
    assert_equal(result.shape[1], 1)


def test_expand_dims_preserves_data() raises:
    """Expand_dims does not change the underlying element values."""
    var a = nm.array[nm.i32]("[1, 2, 3]")
    var result = nm.expand_dims(a, axis=0)
    assert_equal(Int(result.item(0, 0)), 1)
    assert_equal(Int(result.item(0, 1)), 2)
    assert_equal(Int(result.item(0, 2)), 3)


def test_expand_dims_axis_out_of_bound_raises() raises:
    """Expand_dims raises when axis is out of bounds."""
    var a = nm.array[nm.i32]("[1, 2, 3]")
    var raised = False
    try:
        _ = nm.expand_dims(a, axis=5)
    except:
        raised = True
    assert_equal(raised, True)


def test_atleast_1d_already_1d() raises:
    """Atleast_1d leaves an already 1-D array unchanged."""
    var a = nm.array[nm.i32]("[1, 2, 3]")
    var result = nm.atleast_1d(a)
    assert_equal(result.ndim, 1)
    assert_equal(result.shape[0], 3)


def test_atleast_1d_already_2d() raises:
    """Atleast_1d leaves an already 2-D array unchanged."""
    var a = nm.array[nm.i32]("[[1, 2], [3, 4]]")
    var result = nm.atleast_1d(a)
    assert_equal(result.ndim, 2)


def test_atleast_2d_from_1d() raises:
    """Atleast_2d reshapes a 1-D array of size n to shape (1, n)."""
    var a = nm.array[nm.i32]("[1, 2, 3]")
    var result = nm.atleast_2d(a)
    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 1)
    assert_equal(result.shape[1], 3)
    assert_equal(Int(result.item(0, 0)), 1)
    assert_equal(Int(result.item(0, 2)), 3)


def test_atleast_2d_already_2d() raises:
    """Atleast_2d leaves an already 2-D array unchanged."""
    var a = nm.array[nm.i32]("[[1, 2], [3, 4]]")
    var result = nm.atleast_2d(a)
    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 2)


def test_atleast_3d_from_1d() raises:
    """Atleast_3d reshapes a 1-D array of size n to shape (1, n, 1)."""
    var a = nm.array[nm.i32]("[1, 2, 3]")
    var result = nm.atleast_3d(a)
    assert_equal(result.ndim, 3)
    assert_equal(result.shape[0], 1)
    assert_equal(result.shape[1], 3)
    assert_equal(result.shape[2], 1)


def test_atleast_3d_from_2d() raises:
    """Atleast_3d reshapes a (m, n) array to shape (m, n, 1)."""
    var a = nm.array[nm.i32]("[[1, 2], [3, 4]]")
    var result = nm.atleast_3d(a)
    assert_equal(result.ndim, 3)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 2)
    assert_equal(result.shape[2], 1)
    assert_equal(Int(result.item(0, 0, 0)), 1)
    assert_equal(Int(result.item(1, 1, 0)), 4)


def test_atleast_3d_already_3d() raises:
    """Atleast_3d leaves an already 3-D array unchanged."""
    var a = nm.zeros[nm.i32](Shape(2, 3, 4))
    var result = nm.atleast_3d(a)
    assert_equal(result.ndim, 3)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 3)
    assert_equal(result.shape[2], 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
