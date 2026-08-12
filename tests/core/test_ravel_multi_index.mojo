from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_true

import numojo as nm
from numojo.prelude import *


def test_ravel_multi_index_c_order() raises:
    """Ravel_multi_index converts row-major coordinates to flat indices."""
    var rows = nm.array[nm.int]("[3, 6, 6]")
    var cols = nm.array[nm.int]("[4, 5, 1]")
    var coords = List[nm.NDArray[DType.int]]()
    coords.append(rows^)
    coords.append(cols^)

    var result = nm.ravel_multi_index(coords, Shape(7, 6))
    assert_equal(result.ndim, 1)
    assert_equal(result.shape[0], 3)
    assert_equal(Int(result.item(0)), 22)
    assert_equal(Int(result.item(1)), 41)
    assert_equal(Int(result.item(2)), 37)


def test_ravel_multi_index_f_order() raises:
    """Ravel_multi_index converts column-major coordinates to flat indices."""
    var rows = nm.array[nm.int]("[0, 2]")
    var cols = nm.array[nm.int]("[2, 3]")
    var coords = List[nm.NDArray[DType.int]]()
    coords.append(rows^)
    coords.append(cols^)

    var result = nm.ravel_multi_index(coords, Shape(3, 4), order="F")
    assert_equal(result.shape[0], 2)
    assert_equal(Int(result.item(0)), 6)
    assert_equal(Int(result.item(1)), 11)


def test_ravel_multi_index_broadcasts_coordinates() raises:
    """Coordinate arrays broadcast to the result shape."""
    var rows = nm.array[nm.int]("[[0], [1]]")
    var cols = nm.array[nm.int]("[0, 1, 2]")
    var coords = List[nm.NDArray[DType.int]]()
    coords.append(rows^)
    coords.append(cols^)

    var result = nm.ravel_multi_index(coords, Shape(2, 3))
    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 3)
    assert_equal(Int(result.item(0, 0)), 0)
    assert_equal(Int(result.item(0, 1)), 1)
    assert_equal(Int(result.item(0, 2)), 2)
    assert_equal(Int(result.item(1, 0)), 3)
    assert_equal(Int(result.item(1, 1)), 4)
    assert_equal(Int(result.item(1, 2)), 5)


def test_ravel_multi_index_empty_coordinates() raises:
    """Empty coordinate arrays return an empty flat-index array."""
    var rows = nm.zeros[nm.int](Shape(0))
    var cols = nm.zeros[nm.int](Shape(0))
    var coords = List[nm.NDArray[DType.int]]()
    coords.append(rows^)
    coords.append(cols^)

    var result = nm.ravel_multi_index(coords, Shape(2, 3))
    assert_equal(result.ndim, 1)
    assert_equal(result.shape[0], 0)
    assert_equal(result.size, 0)


def test_ravel_multi_index_accepts_shape_list() raises:
    """Ravel_multi_index accepts a shape list."""
    var rows = nm.array[nm.int]("[0, 1]")
    var cols = nm.array[nm.int]("[1, 2]")
    var coords = List[nm.NDArray[DType.int]]()
    coords.append(rows^)
    coords.append(cols^)
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var result = nm.ravel_multi_index(coords, shape)
    assert_equal(Int(result.item(0)), 1)
    assert_equal(Int(result.item(1)), 5)


def test_ravel_multi_index_round_trips_unravel_index() raises:
    """Unravel_index coordinates can be converted back to flat indices."""
    var indices = nm.array[nm.int]("[0, 5, 11]")
    var coords = nm.unravel_index(indices, Shape(3, 4))
    var result = nm.ravel_multi_index(coords, Shape(3, 4))
    assert_equal(result.shape[0], 3)
    assert_equal(Int(result.item(0)), 0)
    assert_equal(Int(result.item(1)), 5)
    assert_equal(Int(result.item(2)), 11)


def test_ravel_multi_index_wrong_coordinate_count_raises() raises:
    """Wrong coordinate-array count raises."""
    var rows = nm.array[nm.int]("[0, 1]")
    var coords = List[nm.NDArray[DType.int]]()
    coords.append(rows^)

    var raised = False
    try:
        var _result = nm.ravel_multi_index(coords, Shape(2, 3))
    except:
        raised = True
    assert_true(raised, "wrong coordinate count should raise")


def test_ravel_multi_index_negative_coordinate_raises() raises:
    """Negative coordinates raise."""
    var rows = nm.array[nm.int]("[-1]")
    var cols = nm.array[nm.int]("[0]")
    var coords = List[nm.NDArray[DType.int]]()
    coords.append(rows^)
    coords.append(cols^)

    var raised = False
    try:
        var _result = nm.ravel_multi_index(coords, Shape(2, 3))
    except:
        raised = True
    assert_true(raised, "negative coordinate should raise")


def test_ravel_multi_index_out_of_bounds_coordinate_raises() raises:
    """Out-of-bounds coordinates raise."""
    var rows = nm.array[nm.int]("[2]")
    var cols = nm.array[nm.int]("[0]")
    var coords = List[nm.NDArray[DType.int]]()
    coords.append(rows^)
    coords.append(cols^)

    var raised = False
    try:
        var _result = nm.ravel_multi_index(coords, Shape(2, 3))
    except:
        raised = True
    assert_true(raised, "out-of-bounds coordinate should raise")


def test_ravel_multi_index_invalid_order_raises() raises:
    """Invalid order raises."""
    var rows = nm.array[nm.int]("[0]")
    var cols = nm.array[nm.int]("[0]")
    var coords = List[nm.NDArray[DType.int]]()
    coords.append(rows^)
    coords.append(cols^)

    var raised = False
    try:
        var _result = nm.ravel_multi_index(coords, Shape(2, 3), order="A")
    except:
        raised = True
    assert_true(raised, "invalid order should raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
