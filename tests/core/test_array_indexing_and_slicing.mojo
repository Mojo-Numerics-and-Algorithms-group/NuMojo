import numojo as nm
from numojo.prelude import *
from std.testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close
from std.python import Python, PythonObject
from std.testing import TestSuite


def test_getitem_scalar() raises:
    var np = Python.import_module("numpy")

    var A = nm.arange(8)
    assert_true(A.load(0) == 0, msg=String("`get` fails"))


def test_setitem() raises:
    var np = Python.import_module("numpy")
    var shape = nm.Shape(4, 4)
    print("Shape: ", shape)
    var arr = nm.NDArray(shape)


# ===== Single-axis integer indexing =====


def test_getitem_single_axis_basic() raises:
    """Positive and negative single-axis int indexing on 2D."""
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.i32](0, 12, step=1).reshape(Shape(3, 4))
    var anp = np.arange(12, dtype=np.int32).reshape(3, 4)
    # positive index
    check(a[1], anp[1], "__getitem__(idx: Int) positive index row slice broken")
    # negative index
    check(
        a[-1],
        anp[-1],
        "__getitem__(idx: Int) negative index row slice broken",
    )


def test_getitem_single_axis_1d_scalar() raises:
    """1D -> 0-D scalar wrapper."""
    var np = Python.import_module("numpy")
    var a = nm.arange[nm.i16](0, 6, step=1).reshape(Shape(6))
    var anp = np.arange(6, dtype=np.int16)
    # 1-D -> 0-D scalar wrapper
    check(a[2], anp[2], "__getitem__(idx: Int) 1-D to scalar (0-D) broken")


def test_negative_int_indexing() raises:
    """Negative integer indexing on 1D and 2D arrays."""
    var np = Python.import_module("numpy")

    # 1D array negative indexing
    var nm_arr_1d = nm.arange[nm.f32](0.0, 10.0, step=1)
    var np_arr_1d = np.arange(0, 10, dtype=np.float32)

    check(nm_arr_1d[-1], np_arr_1d[-1], "1D negative index [-1] failed")
    check(nm_arr_1d[-5], np_arr_1d[-5], "1D negative index [-5] failed")

    # 2D array negative indexing
    var nm_arr_2d = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(Shape(3, 4))
    var np_arr_2d = np.arange(0, 12, dtype=np.float32).reshape(3, 4)

    check(nm_arr_2d[-1], np_arr_2d[-1], "2D negative row index [-1] failed")
    check(nm_arr_2d[-2], np_arr_2d[-2], "2D negative row index [-2] failed")


# ===== Slice-only indexing =====


def test_1d_slicing() raises:
    """1D array slicing with basic, step, and partial ranges."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 10.0, step=1)
    var np_arr = np.arange(0, 10, dtype=np.float32)

    # Basic slicing [2:7]
    var nm_s1 = nm_arr[Slice(2, 7)]
    var np_s1 = np_arr[2:7]
    check(nm_s1, np_s1, "1D basic slice [2:7] failed")

    # With step [1:8:2]
    var nm_s2 = nm_arr[Slice(1, 8, 2)]
    var np_s2 = np_arr[1:8:2]
    check(nm_s2, np_s2, "1D step slice [1:8:2] failed")

    # From start [:5]
    var nm_s3 = nm_arr[Slice(0, 5)]
    var np_s3 = np_arr[:5]
    check(nm_s3, np_s3, "1D from start [:5] failed")


def test_slicing_3d_all_dims() raises:
    """3D C-order slicing across all dimensions."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1).reshape(Shape(2, 3, 4))
    var np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # [:, :, 1:2]
    var nm_s1 = nm_arr[Slice(0, 2), Slice(0, 3), Slice(1, 2)]
    var np_s1 = np_arr[:, :, 1:2]
    check(nm_s1, np_s1, "3D slicing [:, :, 1:2] failed")

    # [0:1, 1:3, 2:4]
    var nm_s2 = nm_arr[Slice(0, 1), Slice(1, 3), Slice(2, 4)]
    var np_s2 = np_arr[0:1, 1:3, 2:4]
    check(nm_s2, np_s2, "3D slicing [0:1, 1:3, 2:4] failed")


def test_positive_slice_indices() raises:
    """Positive indices in slice operations on 3D."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1).reshape(Shape(2, 3, 4))
    var np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # positive start [1:, :, :]
    var nm_s1 = nm_arr[Slice(1, 2), Slice(0, 3), Slice(0, 4)]
    var np_s1 = np_arr[1:, :, :]
    check(nm_s1, np_s1, "Positive start index [1:, :, :] failed")

    # positive start and end [0:2, 1:3, 2:4]
    var nm_s3 = nm_arr[Slice(0, 2), Slice(1, 3), Slice(2, 4)]
    var np_s3 = np_arr[0:2, 1:3, 2:4]
    check(nm_s3, np_s3, "Positive start/end [0:2, 1:3, 2:4] failed")


def test_step_slicing_2d() raises:
    """Step slicing on 2D arrays."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(Shape(3, 4))
    var np_arr = np.arange(0, 12, dtype=np.float32).reshape(3, 4)

    # [::2, :]
    var nm_s1 = nm_arr[Slice(0, 3, 2), Slice(0, 4)]
    var np_s1 = np_arr[::2, :]
    check(nm_s1, np_s1, "Forward step rows [::2, :] failed")

    # [0:3:2, 1:4:2]
    var nm_s2 = nm_arr[Slice(0, 3, 2), Slice(1, 4, 2)]
    var np_s2 = np_arr[0:3:2, 1:4:2]
    check(nm_s2, np_s2, "Step with bounds [0:3:2, 1:4:2] failed")


def test_step_slicing_3d() raises:
    """3D slicing with steps."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 60.0, step=1).reshape(Shape(3, 4, 5))
    var np_arr = np.arange(0, 60, dtype=np.float32).reshape(3, 4, 5)

    # [1:, 1:3, ::2]
    var nm_s1 = nm_arr[Slice(1, 3), Slice(1, 3), Slice(0, 5, 2)]
    var np_s1 = np_arr[1:, 1:3, ::2]
    check(nm_s1, np_s1, "3D complex slice [1:, 1:3, ::2] failed")

    # [::2, :, 1::2]
    var nm_s2 = nm_arr[Slice(0, 3, 2), Slice(0, 4), Slice(1, 5, 2)]
    var np_s2 = np_arr[::2, :, 1::2]
    check(nm_s2, np_s2, "3D alternating [::2, :, 1::2] failed")


# ===== Mixed int + slice indexing (PR #326 fix) =====


def test_mixed_int_slice_2d() raises:
    """Mixed int + slice on 2D array — tests dimension reduction."""
    var np = Python.import_module("numpy")
    var nm_arr = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(Shape(3, 4))

    # int + full slice -> 1D (row selection)
    # a[1, :] should give [4, 5, 6, 7]
    var r1 = nm_arr[1, Slice(0, 4)]
    assert_equal(r1.ndim, 1)
    assert_equal(r1.shape[0], 4)
    assert_equal(Int(r1.load(0)), 4)
    assert_equal(Int(r1.load(3)), 7)

    # full slice + int -> 1D (column selection)
    # a[:, 2] should give [2, 6, 10]
    var r2 = nm_arr[Slice(0, 3), 2]
    assert_equal(r2.ndim, 1)
    assert_equal(r2.shape[0], 3)
    assert_equal(Int(r2.load(0)), 2)
    assert_equal(Int(r2.load(1)), 6)
    assert_equal(Int(r2.load(2)), 10)

    # int + partial slice -> 1D
    # a[2, 1:3] should give [9, 10]
    var r3 = nm_arr[2, Slice(1, 3)]
    assert_equal(r3.ndim, 1)
    assert_equal(r3.shape[0], 2)
    assert_equal(Int(r3.load(0)), 9)
    assert_equal(Int(r3.load(1)), 10)


def test_mixed_int_slice_3d() raises:
    """Mixed int + slice on 3D array — tests dimension reduction."""
    var nm_arr = nm.arange[nm.f32](0.0, 60.0, step=1).reshape(Shape(3, 4, 5))

    # int, slice -> 2D: a[1, 1:3] has shape (2, 5)
    var r1 = nm_arr[1, Slice(1, 3)]
    assert_equal(r1.ndim, 2)
    assert_equal(r1.shape[0], 2)
    assert_equal(r1.shape[1], 5)
    # arr[1,1,0] = 1*20 + 1*5 + 0 = 25
    assert_equal(Int(r1.load(0)), 25)

    # slice, slice, int -> 2D: a[0:2, 1:3, 2] has shape (2, 2)
    var r2 = nm_arr[Slice(0, 2), Slice(1, 3), 2]
    assert_equal(r2.ndim, 2)
    assert_equal(r2.shape[0], 2)
    assert_equal(r2.shape[1], 2)
    # arr[0,1,2] = 0*20 + 1*5 + 2 = 7
    assert_equal(Int(r2.load(0)), 7)

    # int, int -> 1D: a[1, 2] has shape (5,)
    var r3 = nm_arr[1, 2]
    assert_equal(r3.ndim, 1)
    assert_equal(r3.shape[0], 5)
    # arr[1,2,0] = 1*20 + 2*5 = 30
    assert_equal(Int(r3.load(0)), 30)

    # int, int, int -> 0D scalar
    var r4 = nm_arr[1, 2, 3]
    assert_equal(r4.ndim, 0)
    # arr[1,2,3] = 1*20 + 2*5 + 3 = 33
    assert_equal(Int(r4.load(0)), 33)


def test_mixed_int_slice_3d_values() raises:
    """Verify actual values in mixed int+slice 3D indexing."""
    var nm_arr = nm.arange[nm.i32](0, 24, step=1).reshape(Shape(2, 3, 4))

    # a[0, :, 2] should give column 2 of the first matrix: [2, 6, 10]
    var r1 = nm_arr[0, Slice(0, 3), 2]
    assert_equal(r1.ndim, 1)
    assert_equal(r1.shape[0], 3)
    assert_equal(Int(r1.load(0)), 2)
    assert_equal(Int(r1.load(1)), 6)
    assert_equal(Int(r1.load(2)), 10)

    # a[:, 0, :] should give shape (2, 4)
    var r2 = nm_arr[Slice(0, 2), 0, Slice(0, 4)]
    assert_equal(r2.ndim, 2)
    assert_equal(r2.shape[0], 2)
    assert_equal(r2.shape[1], 4)
    # arr[0,0,:] = [0,1,2,3], arr[1,0,:] = [12,13,14,15]
    assert_equal(Int(r2.load(0)), 0)
    assert_equal(Int(r2.load(4)), 12)


def test_mixed_int_slice_preserves_remaining_dims() raises:
    """When fewer indices are given than ndim, remaining dims preserved."""
    var nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1).reshape(Shape(2, 3, 4))

    # Only 1 slice on a 3D array -> should keep all 3 dims
    var r1 = nm_arr[Slice(0, 1)]
    assert_equal(r1.ndim, 3)
    assert_equal(r1.shape[0], 1)
    assert_equal(r1.shape[1], 3)
    assert_equal(r1.shape[2], 4)


# ===== Setitem single-axis tests =====


def test_setitem_single_axis_index_oob_error() raises:
    """Ensure out-of-bounds index raises an error."""
    var a = nm.arange[nm.i32](0, 12, step=1).reshape(Shape(3, 4))
    var row = nm.full[nm.i32](Shape(4), fill_value=Scalar[nm.i32](7))
    var raised: Bool = False
    try:
        a[3] = row  # out of bounds
    except e:
        raised = True
    assert_true(raised, "__setitem__(idx: Int, val) did not raise on OOB index")


# ===== F-order integer indexing =====
# TODO: Fix "F" order issue in NDArray
# def test_getitem_single_axis_f_order():
#     var np = Python.import_module("numpy")
#     var a = nm.arange[nm.i32](0, 12, step=1).reshape(Shape(3, 4), order="F")
#     var anp = np.arange(12, dtype=np.int32).reshape(
#         3, 4, order=PythonObject("F")
#     )
#     check(a[0], anp[0], "__getitem__(idx: Int) F-order first row broken")
#     check(a[2], anp[2], "__getitem__(idx: Int) F-order last row broken")


# ===== Negative-index slicing (known issues, kept commented) =====
# TODO: Fix negative slice indices
# def test_negative_slice_indices():
#     var np = Python.import_module("numpy")
#     var nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1).reshape(Shape(2, 3, 4))
#     var np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)
#     var nm_s1 = nm_arr[Slice(-1, 2), Slice(0, 3), Slice(0, 4)]
#     var np_s1 = np_arr[-1:, :, :]
#     check(nm_s1, np_s1, "Negative start index [-1:, :, :] failed")


# ===== Negative step slicing (known issues, kept commented) =====
# TODO: Fix reverse slicing with negative steps
# def test_negative_step_slicing():
#     var np = Python.import_module("numpy")
#     var nm_arr = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(Shape(3, 4))
#     var np_arr = np.arange(0, 12, dtype=np.float32).reshape(3, 4)
#     var nm_s1 = nm_arr[Slice(2, -1, -1), Slice(0, 4)]
#     var np_s1 = np_arr[::-1, :]
#     check_is_close(nm_s1, np_s1, "Reverse rows [::-1, :] failed")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
