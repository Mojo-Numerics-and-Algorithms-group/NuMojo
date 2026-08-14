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
    """2-D mask all False returns an empty leading dimension."""
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4))
    var mask = nm.zeros[nm.boolean](Shape(2, 3))

    var result = a[mask]
    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 0)
    assert_equal(result.shape[1], 4)
    assert_equal(result.size, 0)


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


def test_getitem_exact_shape_mask_no_true() raises:
    """Exact-shape all-False mask returns an empty 1-D array."""
    var a = nm.arange[nm.i32](0, 6)
    var mask = nm.zeros[nm.boolean](Shape(6))
    var result = a[mask]
    assert_equal(result.ndim, 1)
    assert_equal(result.shape[0], 0)
    assert_equal(result.size, 0)


def test_getitem_1d_mask_on_2d_regression() raises:
    """CASE 2 still works: 1-D mask on 2-D array selects axis-0 rows."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var mask = nm.array[nm.boolean]("[0,1,0]")
    var result = a[mask]
    assert_equal(result.shape[0], 1)
    assert_equal(result.shape[1], 4)
    assert_equal(Int(result.item(0, 0)), 4)
    assert_equal(Int(result.item(0, 3)), 7)


def test_getitem_1d_mask_on_2d_no_true() raises:
    """1-D all-False axis-0 mask returns zero rows."""
    var a = nm.arange[nm.i32](0, 12).reshape(Shape(3, 4))
    var mask = nm.zeros[nm.boolean](Shape(3))
    var result = a[mask]
    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 0)
    assert_equal(result.shape[1], 4)
    assert_equal(result.size, 0)


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


# ===-------------------------------------------------------------------=== #
# Layout-safety: F-order and strided-view destinations.
# These regression tests pin the behavior of the k-D / 1-D mask paths when
# `self` is NOT C-contiguous. Previous implementations assumed C-order and
# silently corrupted data for these cases.
# ===-------------------------------------------------------------------=== #


def test_getitem_2d_mask_on_3d_f_order() raises:
    """K-D mask getter must read correct values when self is F-order."""
    # Same logical values as the C-order test, but F-order storage.
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4), order="F")
    var mask = nm.zeros[nm.boolean](Shape(2, 3))
    mask.itemset([0, 1], True)
    mask.itemset([1, 2], True)

    var result = a[mask]

    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 4)
    # a[0, 1, :] in logical (i,j,k) terms — values come from F-storage decode.
    for k in range(4):
        assert_equal(Int(result.item(0, k)), Int(a.item(0, 1, k)))
        assert_equal(Int(result.item(1, k)), Int(a.item(1, 2, k)))


def test_setitem_scalar_2d_mask_on_3d_f_order() raises:
    """K-D scalar setter must write through F-order strides."""
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4), order="F")
    var mask = nm.zeros[nm.boolean](Shape(2, 3))
    mask.itemset([0, 1], True)
    mask.itemset([1, 2], True)

    # Snapshot untouched positions for the "no neighbor corruption" check.
    var v_002 = Int(a.item(0, 0, 2))
    var v_103 = Int(a.item(1, 0, 3))

    a.set(mask, val=Scalar[nm.i32](-7))

    # Selected sub-arrays are all -7
    for k in range(4):
        assert_equal(Int(a.item(0, 1, k)), -7)
        assert_equal(Int(a.item(1, 2, k)), -7)
    # Untouched positions still hold their original values (strides honored).
    assert_equal(Int(a.item(0, 0, 2)), v_002)
    assert_equal(Int(a.item(1, 0, 3)), v_103)


def test_setitem_ndarray_2d_mask_on_3d_f_order_per_index() raises:
    """K-D NDArray setter with per-index val must write through F-order."""
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(2, 3, 4), order="F")
    # Snapshot one untouched value to verify F-strides are respected.
    var v_120 = Int(a.item(1, 2, 0))
    # Override v_120 only if it's not in a selected slice; (1, 2, *) IS selected
    # so use a different untouched probe.
    var v_010 = Int(a.item(0, 1, 0))

    var mask = nm.zeros[nm.boolean](Shape(2, 3))
    mask.itemset([0, 0], True)
    mask.itemset([1, 2], True)

    # val shape (2, 4) — C-contig per design.
    var val = nm.zeros[nm.i32](Shape(2, 4))
    for k in range(4):
        val.itemset(k, Scalar[nm.i32](10 + k))
        val.itemset(4 + k, Scalar[nm.i32](20 + k))

    a.set(mask, val=val)

    for k in range(4):
        assert_equal(Int(a.item(0, 0, k)), 10 + k)
        assert_equal(Int(a.item(1, 2, k)), 20 + k)
    # Untouched position untouched.
    assert_equal(Int(a.item(0, 1, 0)), v_010)


def test_setitem_scalar_1d_mask_on_3d_f_order() raises:
    """CASE 2 scalar setter (1-D mask, n-D self) must respect F-order."""
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(3, 2, 4), order="F")
    # Snapshot row 1 values (will NOT be touched).
    var snap = List[Int]()
    for j in range(2):
        for k in range(4):
            snap.append(Int(a.item(1, j, k)))

    var mask = nm.array[nm.boolean]("[1, 0, 1]")
    a.set(mask, val=Scalar[nm.i32](42))

    # Every element of rows 0 and 2 (across both trailing dims) is 42.
    for j in range(2):
        for k in range(4):
            assert_equal(Int(a.item(0, j, k)), 42)
            assert_equal(Int(a.item(2, j, k)), 42)
    # Row 1 untouched.
    var idx = 0
    for j in range(2):
        for k in range(4):
            assert_equal(Int(a.item(1, j, k)), snap[idx])
            idx += 1


def test_setitem_ndarray_1d_mask_on_3d_f_order_single_subarray() raises:
    """CASE 2 NDArray setter, single broadcast sub-array, F-order self."""
    var a = nm.arange[nm.i32](0, 24).reshape(Shape(3, 2, 4), order="F")
    var snap = List[Int]()
    for j in range(2):
        for k in range(4):
            snap.append(Int(a.item(1, j, k)))

    var mask = nm.array[nm.boolean]("[1, 0, 1]")
    # val shape (2, 4) matches self.shape[1:].
    var val = nm.arange[nm.i32](100, 108).reshape(Shape(2, 4))

    a.set(mask, val=val)

    for j in range(2):
        for k in range(4):
            assert_equal(Int(a.item(0, j, k)), 100 + j * 4 + k)
            assert_equal(Int(a.item(2, j, k)), 100 + j * 4 + k)
    var idx = 0
    for j in range(2):
        for k in range(4):
            assert_equal(Int(a.item(1, j, k)), snap[idx])
            idx += 1


def test_setitem_scalar_2d_mask_on_strided_view() raises:
    """K-D scalar setter into a strided view (slice with step) must not
    corrupt the parent at positions that are not part of the view."""
    # Parent: 4x3x4 C-contig, all zeros. View = parent[0:4:2, :, :], i.e.
    # rows 0 and 2 of the parent. View shape = (2, 3, 4), stride[0] = 24.
    var parent = nm.zeros[nm.i32](Shape(4, 3, 4))
    # Sentinel values into the rows that should NOT be touched (rows 1 and 3).
    for j in range(3):
        for k in range(4):
            parent.itemset([1, j, k], Scalar[nm.i32](-1))
            parent.itemset([3, j, k], Scalar[nm.i32](-2))

    var view = parent[Slice(0, 4, 2), Slice(0, 3), Slice(0, 4)]
    assert_equal(view.shape[0], 2)
    assert_equal(view.shape[1], 3)
    assert_equal(view.shape[2], 4)

    # 2-D mask of shape (2, 3): True at (0, 1) and (1, 2).
    var mask = nm.zeros[nm.boolean](Shape(2, 3))
    mask.itemset([0, 1], True)
    mask.itemset([1, 2], True)

    view.set(mask, val=Scalar[nm.i32](7))

    # Within the view, written positions are 7; others zero.
    for j in range(3):
        for k in range(4):
            if j == 1:
                assert_equal(Int(view.item(0, j, k)), 7)
            else:
                assert_equal(Int(view.item(0, j, k)), 0)
            if j == 2:
                assert_equal(Int(view.item(1, j, k)), 7)
            else:
                assert_equal(Int(view.item(1, j, k)), 0)

    # Critical: the skipped parent rows must still hold their sentinel values.
    for j in range(3):
        for k in range(4):
            assert_equal(Int(parent.item(1, j, k)), -1)
            assert_equal(Int(parent.item(3, j, k)), -2)


def test_setitem_ndarray_2d_mask_on_strided_view_per_index() raises:
    """K-D NDArray setter with per-index val into a strided view."""
    var parent = nm.zeros[nm.i32](Shape(4, 3, 4))
    for j in range(3):
        for k in range(4):
            parent.itemset([1, j, k], Scalar[nm.i32](-1))
            parent.itemset([3, j, k], Scalar[nm.i32](-2))

    var view = parent[Slice(0, 4, 2), Slice(0, 3), Slice(0, 4)]

    var mask = nm.zeros[nm.boolean](Shape(2, 3))
    mask.itemset([0, 0], True)
    mask.itemset([1, 2], True)

    # val shape (2, 4) — row 0 → [10..13], row 1 → [20..23]
    var val = nm.zeros[nm.i32](Shape(2, 4))
    for k in range(4):
        val.itemset(k, Scalar[nm.i32](10 + k))
        val.itemset(4 + k, Scalar[nm.i32](20 + k))

    view.set(mask, val=val)

    for k in range(4):
        assert_equal(Int(view.item(0, 0, k)), 10 + k)
        assert_equal(Int(view.item(1, 2, k)), 20 + k)
    # Other view positions still zero.
    assert_equal(Int(view.item(0, 1, 0)), 0)
    assert_equal(Int(view.item(1, 0, 0)), 0)
    # Skipped parent rows still hold sentinels.
    for j in range(3):
        for k in range(4):
            assert_equal(Int(parent.item(1, j, k)), -1)
            assert_equal(Int(parent.item(3, j, k)), -2)


def test_getitem_2d_mask_on_strided_view() raises:
    """K-D mask getter on a strided view must gather from the correct
    parent positions, not from `view.contiguous()`'s buffer."""
    var parent = nm.arange[nm.i32](0, 48).reshape(Shape(4, 3, 4))
    var view = parent[Slice(0, 4, 2), Slice(0, 3), Slice(0, 4)]
    # view[i, j, k] should equal parent[2*i, j, k].

    var mask = nm.zeros[nm.boolean](Shape(2, 3))
    mask.itemset([0, 1], True)
    mask.itemset([1, 2], True)

    var result = view[mask]

    for k in range(4):
        # view[0, 1, k] == parent[0, 1, k] == 0*12 + 1*4 + k = 4+k
        assert_equal(Int(result.item(0, k)), 4 + k)
        # view[1, 2, k] == parent[2, 2, k] == 2*12 + 2*4 + k = 32+k
        assert_equal(Int(result.item(1, k)), 32 + k)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
