from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_true

import numojo as nm
from numojo.prelude import *


def test_unravel_index_scalar() raises:
    """Scalar unravel_index returns one coordinate per dimension."""
    var coords = nm.unravel_index(22, Shape(3, 4, 5))
    assert_equal(len(coords), 3)
    assert_equal(coords[0], 1)
    assert_equal(coords[1], 0)
    assert_equal(coords[2], 2)


def test_unravel_index_scalar_fortran_order() raises:
    """Scalar unravel_index supports F-order coordinates."""
    var coords = nm.unravel_index(22, Shape(3, 4, 5), order="F")
    assert_equal(len(coords), 3)
    assert_equal(coords[0], 1)
    assert_equal(coords[1], 3)
    assert_equal(coords[2], 1)


def test_unravel_index_list_shape() raises:
    """Unravel_index accepts a shape list."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var coords = nm.unravel_index(6, shape)
    assert_equal(len(coords), 2)
    assert_equal(coords[0], 1)
    assert_equal(coords[1], 2)


def test_unravel_index_array_1d_indices() raises:
    """Array unravel_index returns coordinate arrays shaped like indices."""
    var idx = nm.array[nm.int]("[22, 41, 37]")
    var coords = nm.unravel_index(idx, Shape(7, 6))
    assert_equal(len(coords), 2)
    assert_equal(coords[0].ndim, 1)
    assert_equal(coords[0].shape[0], 3)
    assert_equal(Int(coords[0].item(0)), 3)
    assert_equal(Int(coords[1].item(0)), 4)
    assert_equal(Int(coords[0].item(1)), 6)
    assert_equal(Int(coords[1].item(1)), 5)
    assert_equal(Int(coords[0].item(2)), 6)
    assert_equal(Int(coords[1].item(2)), 1)


def test_unravel_index_array_fortran_order() raises:
    """Array unravel_index supports F-order coordinates."""
    var idx = nm.array[nm.int]("[6, 11]")
    var coords = nm.unravel_index(idx, Shape(3, 4), order="F")
    assert_equal(len(coords), 2)
    assert_equal(Int(coords[0].item(0)), 0)
    assert_equal(Int(coords[1].item(0)), 2)
    assert_equal(Int(coords[0].item(1)), 2)
    assert_equal(Int(coords[1].item(1)), 3)


def test_unravel_index_array_preserves_index_shape() raises:
    """Coordinate arrays preserve the shape of an N-D indices array."""
    var idx = nm.array[nm.int]("[[0, 5], [6, 11]]")
    var coords = nm.unravel_index(idx, Shape(3, 4))
    assert_equal(coords[0].ndim, 2)
    assert_equal(coords[0].shape[0], 2)
    assert_equal(coords[0].shape[1], 2)
    assert_equal(coords[1].shape[0], 2)
    assert_equal(coords[1].shape[1], 2)
    assert_equal(Int(coords[0].item(0, 0)), 0)
    assert_equal(Int(coords[1].item(0, 0)), 0)
    assert_equal(Int(coords[0].item(0, 1)), 1)
    assert_equal(Int(coords[1].item(0, 1)), 1)
    assert_equal(Int(coords[0].item(1, 0)), 1)
    assert_equal(Int(coords[1].item(1, 0)), 2)
    assert_equal(Int(coords[0].item(1, 1)), 2)
    assert_equal(Int(coords[1].item(1, 1)), 3)


def test_unravel_index_empty_indices() raises:
    """Empty indices return empty coordinate arrays."""
    var idx = nm.zeros[nm.int](Shape(0))
    var coords = nm.unravel_index(idx, Shape(2, 3))
    assert_equal(len(coords), 2)
    assert_equal(coords[0].ndim, 1)
    assert_equal(coords[0].shape[0], 0)
    assert_equal(coords[1].shape[0], 0)


def test_unravel_index_negative_raises() raises:
    """Negative flat indices raise."""
    var raised = False
    try:
        var _coords = nm.unravel_index(-1, Shape(2, 3))
    except:
        raised = True
    assert_true(raised, "negative scalar index should raise")


def test_unravel_index_out_of_bounds_raises() raises:
    """Out-of-bounds flat indices raise."""
    var raised = False
    try:
        var _coords = nm.unravel_index(nm.array[nm.int]("[0, 6]"), Shape(2, 3))
    except:
        raised = True
    assert_true(raised, "out-of-bounds array index should raise")


def test_unravel_index_invalid_order_raises() raises:
    """Invalid order raises."""
    var raised = False
    try:
        var _coords = nm.unravel_index(0, Shape(2, 3), order="A")
    except:
        raised = True
    assert_true(raised, "invalid order should raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
