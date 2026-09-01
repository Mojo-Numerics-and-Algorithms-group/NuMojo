from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_almost_equal

import numojo as nm
from numojo.prelude import *


def test_ptp_1d() raises:
    """Ptp returns max - min over a flattened 1-D array."""
    var a = nm.array[nm.i32]("[3, 7, 1, 9, 4]")
    assert_equal(Int(nm.ptp(a)), 8)


def test_ptp_method() raises:
    """NDArray.ptp delegates to the extrema routine."""
    var a = nm.array[nm.i32]("[3, 7, 1, 9, 4]")
    assert_equal(Int(a.ptp()), 8)


def test_ptp_2d_flat() raises:
    """Ptp without axis reduces over the whole array."""
    var a = nm.array[nm.i32]("[[1, 5], [9, 2]]")
    assert_equal(Int(nm.ptp(a)), 8)


def test_ptp_constant_array_is_zero() raises:
    """Ptp of a constant array is zero."""
    var a = nm.full[nm.i32](Shape(3, 3), 5)
    assert_equal(Int(nm.ptp(a)), 0)


def test_ptp_axis0() raises:
    """Ptp(axis=0) computes range down each column."""
    var a = nm.array[nm.i32]("[[1, 5, 3], [4, 2, 9]]")
    var result = nm.ptp(a, axis=0)
    assert_equal(result.ndim, 1)
    assert_equal(Int(result.item(0)), 3)
    assert_equal(Int(result.item(1)), 3)
    assert_equal(Int(result.item(2)), 6)


def test_ptp_axis1() raises:
    """Ptp(axis=1) computes range across each row."""
    var a = nm.array[nm.i32]("[[1, 5, 3], [4, 2, 9]]")
    var result = nm.ptp(a, axis=1)
    assert_equal(result.ndim, 1)
    assert_equal(Int(result.item(0)), 4)
    assert_equal(Int(result.item(1)), 7)


def test_ptp_method_axis() raises:
    """NDArray.ptp(axis) delegates to the extrema routine."""
    var a = nm.array[nm.i32]("[[1, 5, 3], [4, 2, 9]]")
    var result = a.ptp(axis=0)
    assert_equal(Int(result.item(0)), 3)
    assert_equal(Int(result.item(1)), 3)
    assert_equal(Int(result.item(2)), 6)


def test_ptp_float() raises:
    """Ptp works on floating-point arrays."""
    var a = nm.array[nm.f32]("[1.5, 4.5, -2.0, 3.0]")
    assert_almost_equal(Float64(nm.ptp(a)), 6.5)


def test_ptp_axis_out_of_bound_raises() raises:
    """Ptp raises when axis is out of bounds."""
    var a = nm.array[nm.i32]("[[1, 5], [9, 2]]")
    var raised = False
    try:
        _ = nm.ptp(a, axis=5)
    except:
        raised = True
    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
