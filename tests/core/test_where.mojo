from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_true

import numojo as nm
from numojo.prelude import *


def test_where_array_array() raises:
    """where(cond, x, y) with two arrays picks x where True, y where False."""
    var a = nm.array[nm.f32]("[1.0, 2.0, 3.0, 4.0]")
    var b = nm.array[nm.f32]("[10.0, 20.0, 30.0, 40.0]")
    var mask = nm.array[boolean]("[1, 0, 1, 0]")
    var result = nm.`where`(mask, a, b)
    assert_equal(result.shape[0], 4)
    assert_equal(Int(result.item(0)), 1)
    assert_equal(Int(result.item(1)), 20)
    assert_equal(Int(result.item(2)), 3)
    assert_equal(Int(result.item(3)), 40)


def test_where_array_scalar() raises:
    """where(cond, x, scalar) fills scalar where False."""
    var a = nm.array[nm.f32]("[1.0, 2.0, 3.0, 4.0]")
    var mask = nm.array[boolean]("[1, 0, 1, 0]")
    var result = nm.`where`(mask, a, Scalar[nm.f32](0.0))
    assert_equal(Int(result.item(0)), 1)
    assert_equal(Int(result.item(1)), 0)
    assert_equal(Int(result.item(2)), 3)
    assert_equal(Int(result.item(3)), 0)


def test_where_scalar_array() raises:
    """where(cond, scalar, y) fills scalar where True."""
    var b = nm.array[nm.f32]("[10.0, 20.0, 30.0, 40.0]")
    var mask = nm.array[boolean]("[1, 0, 1, 0]")
    var result = nm.`where`(mask, Scalar[nm.f32](-1.0), b)
    assert_equal(Int(result.item(0)), -1)
    assert_equal(Int(result.item(1)), 20)
    assert_equal(Int(result.item(2)), -1)
    assert_equal(Int(result.item(3)), 40)


def test_where_all_true() raises:
    """All-True mask returns x unchanged."""
    var a = nm.array[nm.i32]("[5, 6, 7]")
    var b = nm.array[nm.i32]("[0, 0, 0]")
    var mask = nm.array[boolean]("[1, 1, 1]")
    var result = nm.`where`(mask, a, b)
    assert_equal(Int(result.item(0)), 5)
    assert_equal(Int(result.item(1)), 6)
    assert_equal(Int(result.item(2)), 7)


def test_where_all_false() raises:
    """All-False mask returns y unchanged."""
    var a = nm.array[nm.i32]("[5, 6, 7]")
    var b = nm.array[nm.i32]("[1, 2, 3]")
    var mask = nm.array[boolean]("[0, 0, 0]")
    var result = nm.`where`(mask, a, b)
    assert_equal(Int(result.item(0)), 1)
    assert_equal(Int(result.item(1)), 2)
    assert_equal(Int(result.item(2)), 3)


def test_where_2d() raises:
    """where works element-wise on 2-D arrays."""
    var a = nm.arange[nm.i32](1, 5).reshape(Shape(2, 2))
    var b = nm.zeros[nm.i32](Shape(2, 2))
    # mask = [[True, False],[False, True]]
    var mask = nm.array[boolean]("[[1, 0],[0, 1]]")
    var result = nm.`where`(mask, a, b)
    assert_equal(result.ndim, 2)
    assert_equal(Int(result.item(0, 0)), 1)  # from a
    assert_equal(Int(result.item(0, 1)), 0)  # from b
    assert_equal(Int(result.item(1, 0)), 0)  # from b
    assert_equal(Int(result.item(1, 1)), 4)  # from a


def test_where_method_array_array() raises:
    """Bool-mask method picks values from two arrays."""
    var a = nm.array[boolean]("[1, 1, 0, 0]")
    var b = nm.array[boolean]("[0, 0, 1, 1]")
    var mask = nm.array[boolean]("[1, 0, 1, 0]")
    var result = mask.where(a, b)
    assert_true(Bool(result.item(0)))
    assert_true(not Bool(result.item(1)))
    assert_true(not Bool(result.item(2)))
    assert_true(Bool(result.item(3)))


def test_where_method_array_scalar() raises:
    """Bool-mask method supports scalar false branch."""
    var a = nm.array[boolean]("[1, 1, 1, 1]")
    var mask = nm.array[boolean]("[1, 0, 1, 0]")
    var result = mask.where(a, Scalar[nm.boolean](False))
    assert_true(Bool(result.item(0)))
    assert_true(not Bool(result.item(1)))
    assert_true(Bool(result.item(2)))
    assert_true(not Bool(result.item(3)))


def test_where_method_scalar_array() raises:
    """Bool-mask method supports scalar true branch."""
    var b = nm.array[boolean]("[0, 1, 0, 1]")
    var mask = nm.array[boolean]("[1, 0, 1, 0]")
    var result = mask.where(Scalar[nm.boolean](True), b)
    assert_true(Bool(result.item(0)))
    assert_true(Bool(result.item(1)))
    assert_true(Bool(result.item(2)))
    assert_true(Bool(result.item(3)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
