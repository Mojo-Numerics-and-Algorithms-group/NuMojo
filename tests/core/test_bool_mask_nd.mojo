from std.testing import TestSuite
from std.testing.testing import assert_equal, assert_true

import numojo as nm
from numojo.prelude import *


def test_getitem_2d_mask_on_3d_array() raises:
    """2-D mask on 3-D array selects sub-arrays of shape (shape[2],)."""
    # a shape (2, 3, 4):  a[i,j,k] = i*12 + j*4 + k
    # mask shape (2, 3): True at (0,1) and (1,2)
    #: result shape (2, 4): rows a[0,1,:] and a[1,2,:]
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4))
    var mask = nm.zeros[nm.boolean](Shape(2, 3))
    mask.itemset([0, 1], True)
    mask.itemset([1, 2], True)

    var result = a[mask]

    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 4)
    # First selected sub-array: a[0,1,:] = [4,5,6,7]
    assert_equal(Int(result.item(0, 0)), 4)
    assert_equal(Int(result.item(0, 1)), 5)
    assert_equal(Int(result.item(0, 2)), 6)
    assert_equal(Int(result.item(0, 3)), 7)
    # Second selected sub-array: a[1,2,:] = [20,21,22,23]
    assert_equal(Int(result.item(1, 0)), 20)
    assert_equal(Int(result.item(1, 1)), 21)
    assert_equal(Int(result.item(1, 2)), 22)
    assert_equal(Int(result.item(1, 3)), 23)


def test_getitem_2d_mask_single_true() raises:
    """2-D mask with only one True: result shape (1, trailing_dims)."""
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4))
    var mask = nm.zeros[nm.boolean](Shape(2, 3))
    mask.itemset([1, 0], True)

    var result = a[mask]

    assert_equal(result.shape[0], 1)
    assert_equal(result.shape[1], 4)
    # a[1,0,:] = [12,13,14,15]
    assert_equal(Int(result.item(0, 0)), 12)
    assert_equal(Int(result.item(0, 3)), 15)


def test_getitem_2d_mask_all_true() raises:
    """2-D mask all True: result has all sub-arrays in row-major order."""
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4))
    var mask = nm.ones[nm.boolean](Shape(2, 3))

    var result = a[mask]

    assert_equal(result.shape[0], 6)
    assert_equal(result.shape[1], 4)
    # Row-major order: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    assert_equal(Int(result.item(0, 0)), 0)  # a[0,0,0]
    assert_equal(Int(result.item(2, 0)), 8)  # a[0,2,0]
    assert_equal(Int(result.item(3, 0)), 12)  # a[1,0,0]
    assert_equal(Int(result.item(5, 3)), 23)  # a[1,2,3]


def test_getitem_2d_mask_no_true() raises:
    """2-D mask all False: raises because empty arrays are not supported."""
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4))
    var mask = nm.zeros[nm.boolean](Shape(2, 3))

    var raised = False
    try:
        var _ = a[mask]
    except:
        raised = True
    assert_true(raised, "all-False k-D mask should raise")


def test_getitem_2d_mask_on_4d_array() raises:
    """2-D mask on 4-D array: result shape (count, shape[2], shape[3])."""
    # a shape (2, 2, 3, 3): a[i,j,k,l] = i*18 + j*9 + k*3 + l
    var a = nm.arange[nm.i32](0, 36).reshape(Shape(2, 2, 3, 3))
    var mask = nm.zeros[nm.boolean](Shape(2, 2))
    mask.itemset([0, 1], True)
    mask.itemset([1, 1], True)

    var result = a[mask]

    assert_equal(result.ndim, 3)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 3)
    assert_equal(result.shape[2], 3)
    # a[0,1,:,:]: starts at offset 0*18+1*9 = 9
    assert_equal(Int(result.item(0, 0, 0)), 9)
    assert_equal(Int(result.item(0, 2, 2)), 17)
    # a[1,1,:,:]: starts at offset 1*18+1*9 = 27
    assert_equal(Int(result.item(1, 0, 0)), 27)
    assert_equal(Int(result.item(1, 2, 2)), 35)


def test_getitem_exact_shape_mask_regression() raises:
    """CASE 1 still works: exact shape mask: flattened 1-D result."""
    var a = nm.arange[nm.i32](0, 6)
    var mask = nm.array[nm.boolean]("[1,0,1,1,0,1]")
    var result = a[mask]
    assert_equal(result.ndim, 1)
    assert_equal(result.shape[0], 4)
    assert_equal(Int(result.item(0)), 0)
    assert_equal(Int(result.item(1)), 2)
    assert_equal(Int(result.item(3)), 5)


def test_getitem_1d_mask_on_2d_regression() raises:
    """CASE 2 still works: 1-D mask on 2-D array selects axis-0 rows."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var mask = nm.array[nm.boolean]("[0,1,0]")
    var result = a[mask]
    assert_equal(result.shape[0], 1)
    assert_equal(result.shape[1], 4)
    assert_equal(Int(result.item(0, 0)), 4)
    assert_equal(Int(result.item(0, 3)), 7)


def test_setitem_scalar_2d_mask_on_3d() raises:
    """Scalar setitem: 2-D mask on 3-D array zeros selected sub-arrays."""
    # a shape (2, 3, 4), mask True at (0,1) and (1,2)
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4))
    var mask = nm.zeros[nm.boolean](Shape(2, 3))
    mask.itemset([0, 1], True)
    mask.itemset([1, 2], True)

    a.set(mask, val=Scalar[nm.i32](0))

    # a[0,1,:] should be 0
    for k in range(4):
        assert_equal(Int(a.item(0, 1, k)), 0)
    # a[1,2,:] should be 0
    for k in range(4):
        assert_equal(Int(a.item(1, 2, k)), 0)
    # Unmasked positions unchanged
    assert_equal(Int(a.item(0, 0, 0)), 0)  # original value 0
    assert_equal(Int(a.item(0, 2, 3)), 11)
    assert_equal(Int(a.item(1, 0, 0)), 12)
    assert_equal(Int(a.item(1, 1, 3)), 19)


def test_setitem_scalar_2d_mask_non_zero_value() raises:
    """Scalar setitem: 2-D mask writes a non-zero scalar."""
    var a = nm.zeros[nm.i32](Shape(2, 3, 4))
    var mask = nm.zeros[nm.boolean](Shape(2, 3))
    mask.itemset([0, 0], True)
    mask.itemset([1, 1], True)

    a.set(mask, val=Scalar[nm.i32](99))

    for k in range(4):
        assert_equal(Int(a.item(0, 0, k)), 99)
        assert_equal(Int(a.item(1, 1, k)), 99)

    assert_equal(Int(a.item(0, 1, 0)), 0)
    assert_equal(Int(a.item(1, 0, 0)), 0)


def test_setitem_ndarray_2d_mask_single_subarray() raises:
    """NDArray setitem: 2-D mask, val is a single sub-array broadcast."""
    # Broadcast one (4,) row to all selected (2) positions
    var a = nm.zeros[nm.i32](Shape(2, 3, 4))
    var mask = nm.zeros[nm.boolean](Shape(2, 3))
    mask.itemset([0, 1], True)
    mask.itemset([1, 2], True)

    var val = nm.arange[nm.i32](10, 14).reshape(Shape(4))
    a.set(mask, val=val)

    # Both selected sub-arrays get [10,11,12,13]
    for k in range(4):
        assert_equal(Int(a.item(0, 1, k)), 10 + k)
        assert_equal(Int(a.item(1, 2, k)), 10 + k)

    assert_equal(Int(a.item(0, 0, 0)), 0)
    assert_equal(Int(a.item(1, 0, 0)), 0)


def test_setitem_ndarray_2d_mask_per_index_val() raises:
    """NDArray setitem: 2-D mask, val has shape (true_count, *trailing)."""
    var a = nm.zeros[nm.i32](Shape(2, 3, 4))
    var mask = nm.zeros[nm.boolean](Shape(2, 3))
    mask.itemset([0, 0], True)
    mask.itemset([1, 2], True)

    # val shape (2, 4): row 0: [10..13], row 1: [20..23]
    var val = nm.zeros[nm.i32](Shape(2, 4))
    for k in range(4):
        val.itemset(k, Scalar[nm.i32](10 + k))
        val.itemset(4 + k, Scalar[nm.i32](20 + k))

    a.set(mask, val=val)

    for k in range(4):
        assert_equal(Int(a.item(0, 0, k)), 10 + k)
        assert_equal(Int(a.item(1, 2, k)), 20 + k)

    assert_equal(Int(a.item(0, 1, 0)), 0)
    assert_equal(Int(a.item(1, 0, 0)), 0)


def test_setitem_exact_shape_scalar_regression() raises:
    """CASE 1 scalar setitem still works after refactor."""
    var a = nm.arange[nm.i32](0, 6)
    var mask = nm.array[nm.boolean]("[0,1,0,1,0,0]")
    a.set(mask, val=Scalar[nm.i32](99))
    assert_equal(Int(a.item(0)), 0)
    assert_equal(Int(a.item(1)), 99)
    assert_equal(Int(a.item(2)), 2)
    assert_equal(Int(a.item(3)), 99)
    assert_equal(Int(a.item(5)), 5)


def test_setitem_exact_shape_ndarray_regression() raises:
    """CASE 1 NDArray setitem (compact 1-D val) still works after refactor."""
    var a = nm.arange[nm.i32](0, 6).reshape(Shape(2, 3))
    var mask = nm.array[nm.boolean]("[[1,0,1],[0,1,0]]")
    var vals = nm.array[nm.i32]("[50, 60, 70]")
    a.set(mask, val=vals)
    assert_equal(Int(a.item(0, 0)), 50)
    assert_equal(Int(a.item(0, 2)), 60)
    assert_equal(Int(a.item(1, 1)), 70)
    assert_equal(Int(a.item(0, 1)), 1)
    assert_equal(Int(a.item(1, 0)), 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
