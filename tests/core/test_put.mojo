from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_true

import numojo as nm
from numojo.prelude import *


def test_put_array_values_happy_path() raises:
    """put with an array of values writes into flat positions."""
    var a = nm.arange[nm.i32](0, 6)
    a.put(nm.array[nm.int]("[0, 2]"), nm.array[nm.i32]("[10, 20]"))
    assert_equal(Int(a.item(0)), 10)
    assert_equal(Int(a.item(1)), 1)
    assert_equal(Int(a.item(2)), 20)
    assert_equal(Int(a.item(3)), 3)


def test_put_scalar_broadcast() raises:
    """put with a scalar value broadcasts to all indices."""
    var a = nm.arange[nm.i32](0, 6)
    a.put(nm.array[nm.int]("[0, 2]"), Scalar[nm.i32](99))
    assert_equal(Int(a.item(0)), 99)
    assert_equal(Int(a.item(1)), 1)
    assert_equal(Int(a.item(2)), 99)
    assert_equal(Int(a.item(3)), 3)


def test_put_negative_indices() raises:
    """put accepts negative (from-the-end) flat indices."""
    var a = nm.arange[nm.i32](0, 6)
    a.put(nm.array[nm.int]("[-1, -2]"), nm.array[nm.i32]("[100, 200]"))
    assert_equal(Int(a.item(5)), 100)
    assert_equal(Int(a.item(4)), 200)


def test_put_values_broadcast_cyclically() raises:
    """When values is shorter than indices, it cycles."""
    var a = nm.arange[nm.i32](0, 6)
    a.put(nm.array[nm.int]("[0, 1, 2, 3]"), nm.array[nm.i32]("[7, 8]"))
    assert_equal(Int(a.item(0)), 7)
    assert_equal(Int(a.item(1)), 8)
    assert_equal(Int(a.item(2)), 7)
    assert_equal(Int(a.item(3)), 8)


def test_put_on_2d_array_flat_indexing() raises:
    """put indexes into the flattened (ravel-order) positions of an N-D array.
    """
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    a.put(nm.array[nm.int]("[0, 5, 11]"), nm.array[nm.i32]("[-1, -2, -3]"))
    assert_equal(Int(a.item(0, 0)), -1)
    assert_equal(Int(a.item(1, 1)), -2)
    assert_equal(Int(a.item(2, 3)), -3)


def test_put_out_of_bounds_raises() raises:
    """put raises when an index is out of bounds for the flattened array."""
    var a = nm.arange[nm.i32](0, 6)
    var raised = False
    try:
        a.put(nm.array[nm.int]("[0, 10]"), nm.array[nm.i32]("[1, 2]"))
    except:
        raised = True
    assert_true(raised, "out-of-bounds put index should raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
