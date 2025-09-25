import numojo as nm
from numojo.prelude import *
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close
from python import Python


def test_getitem_scalar():
    var np = Python.import_module("numpy")

    var A = nm.arange(8)
    assert_true(A.load(0) == 0, msg=String("`get` fails"))


def test_setitem():
    var np = Python.import_module("numpy")
    var arr = nm.NDArray(nm.Shape(4, 4))
    var np_arr = arr.to_numpy()
    arr.itemset(List(2, 2), 1000)
    np_arr[2, 2] = 1000
    check_is_close(arr, np_arr, "Itemset is broken")


# Has issues, not sure why.
# def test_slicing_getter1():
#     var np = Python.import_module("numpy")

#     # Test C-order array slicing
#     nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1).reshape(nm.Shape(2, 3, 4), order="C")
#     np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

#     # Test case 1: Slicing all dimensions
#     nm_slice1 = nm_arr[:, :, 1:2]
#     np_sliced1 = np_arr[:, :, 1:2]
#     check(nm_slice1, np_sliced1, "3D array slicing (C-order) [:, :, 1:2]")


def test_slicing_getter2():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr = nm_arr.reshape(nm.Shape(2, 3, 4), order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 2: Slicing with start and end indices
    nm_slice2 = nm_arr[0:1, 1:3, 2:4]
    np_sliced2 = np_arr[0:1, 1:3, 2:4]
    check(nm_slice2, np_sliced2, "3D array slicing (C-order) [0:1, 1:3, 2:4]")


def test_slicing_getter3():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr = nm_arr.reshape(nm.Shape(2, 3, 4), order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 3: Slicing with mixed start, end, and step values
    nm_slice3 = nm_arr[1:, 0:2, ::2]
    np_sliced3 = np_arr[1:, 0:2, ::2]
    check(nm_slice3, np_sliced3, "3D array slicing (C-order) [1:, 0:2, ::2]")


def test_slicing_getter4():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr = nm_arr.reshape(nm.Shape(2, 3, 4), order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 4: Slicing with step
    nm_slice4 = nm_arr[::2, ::2, ::2]
    np_sliced4 = np_arr[::2, ::2, ::2]
    check(nm_slice4, np_sliced4, "3D array slicing (C-order) [::2, ::2, ::2]")


def test_slicing_getter5():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr = nm_arr.reshape(nm.Shape(2, 3, 4), order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 5: Slicing with combination of integer and slices
    nm_slice5 = nm_arr[1:2, :, 1:3]
    np_sliced5 = np_arr[1:2, :, 1:3]
    check(nm_slice5, np_sliced5, "3D array slicing (C-order) [1:2, :, 1:3]")


def test_slicing_getter6():
    var np = Python.import_module("numpy")

    var b = nm.arange[i8](60).reshape(nm.Shape(3, 4, 5))
    var ind = nm.array[int]("[[2,0,1], [1,0,1]]")
    var mask = nm.array[boolean]("[1,0,1]")

    var bnp = b.to_numpy()
    var indnp = ind.to_numpy()
    var masknp = mask.to_numpy()

    check(b[ind], bnp[indnp], "Get by indices array fails")
    check(b[mask], bnp[masknp], "Get by mask array fails")


def test_getitem_single_axis_basic():
    var np = Python.import_module("numpy")
    var a = nm.arange[i32](0, 12, 1).reshape(nm.Shape(3, 4))
    var anp = np.arange(12, dtype=np.int32).reshape(3, 4)
    # positive index
    check(a[1], anp[1], "__getitem__(idx: Int) positive index row slice broken")
    # negative index
    check(
        a[-1], anp[-1], "__getitem__(idx: Int) negative index row slice broken"
    )


def test_getitem_single_axis_1d_scalar():
    var np = Python.import_module("numpy")
    var a = nm.arange[i16](0, 6, 1).reshape(nm.Shape(6))
    var anp = np.arange(6, dtype=np.int16)
    # 1-D -> 0-D scalar wrapper
    check(a[2], anp[2], "__getitem__(idx: Int) 1-D to scalar (0-D) broken")


def test_getitem_single_axis_f_order():
    var np = Python.import_module("numpy")
    var a = nm.arange[i32](0, 12, 1).reshape(nm.Shape(3, 4), order="F")
    var anp = np.arange(12, dtype=np.int32).reshape(3, 4, order="F")
    check(a[0], anp[0], "__getitem__(idx: Int) F-order first row broken")
    check(a[2], anp[2], "__getitem__(idx: Int) F-order last row broken")


def test_setitem_single_axis_basic():
    var np = Python.import_module("numpy")
    var a = nm.arange[i32](0, 12, 1).reshape(nm.Shape(3, 4))
    var anp = np.arange(12, dtype=np.int32).reshape(3, 4)
    var row = nm.full[i32](nm.Shape(4), fill_value=Scalar[i32](999))
    a[1] = row
    anp[1] = 999
    check(a, anp, "__setitem__(idx: Int, val) C-order assignment broken")
    # negative index assignment
    var row2 = nm.full[i32](nm.Shape(4), fill_value=Scalar[i32](-5))
    a[-1] = row2
    anp[-1] = -5
    check(a, anp, "__setitem__(idx: Int, val) negative index assignment broken")


def test_setitem_single_axis_f_order():
    var np = Python.import_module("numpy")
    var a = nm.arange[i32](0, 12, 1).reshape(nm.Shape(3, 4), order="F")
    var anp = np.arange(12, dtype=np.int32).reshape(3, 4, order="F")
    var row = nm.full[i32](nm.Shape(4), fill_value=Scalar[i32](111))
    a[0] = row
    anp[0] = 111
    check(a, anp, "__setitem__(idx: Int, val) F-order assignment broken")


def test_setitem_single_axis_shape_mismatch_error():
    # Ensure nm.Shape mismatch raises an error (val nm.Shape != self.nm.Shape[1:])
    var a = nm.arange[i32](0, 12, 1).reshape(nm.Shape(3, 4))
    var bad = nm.full[i32](
        nm.Shape(5), fill_value=Scalar[i32](1)
    )  # wrong length
    var raised: Bool = False
    try:
        a[0] = bad
    except e:
        raised = True
    assert_true(
        raised, "__setitem__(idx: Int, val) did not raise on nm.Shape mismatch"
    )


def test_setitem_single_axis_index_oob_error():
    # Ensure out-of-bounds index raises an error
    var a = nm.arange[i32](0, 12, 1).reshape(nm.Shape(3, 4))
    var row = nm.full[i32](nm.Shape(4), fill_value=Scalar[i32](7))
    var raised: Bool = False
    try:
        a[3] = row  # out of bounds
    except e:
        raised = True
    assert_true(raised, "__setitem__(idx: Int, val) did not raise on OOB index")


# def test_slicing_setter1():
#     var np = Python.import_module("numpy")

#     # Test C-order array slicing
#     nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
#     nm_arr.reshape(2, 3, 4, order="C")
#     np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

#     # Test case 2: Setting a slice with another array
#     nm_set_arr = nm.full[nm.f32](2, 2, fill_value=50.0)
#     np_set_arr = np.full((1, 2, 2), 50, dtype=np.float32)
#     nm_arr[1:2, 1:3, 2:4] = nm_set_arr
#     np.put(np_arr, np.ravel_multi_index((np.arange(1, 2), np.arange(1, 3), np.arange(2, 4)), np_arr.nm.Shape), np_set_arr.flatten())
#     check(nm_arr, np_arr, "3D array slice setting (C-order) [1:2, 1:3, 2:4] = array")


def test_positive_indices_basic():
    """Test basic positive indexing (current implementation support)."""
    var np = Python.import_module("numpy")

    # 1D array positive indexing
    var nm_arr_1d = nm.arange[nm.f32](0.0, 10.0, step=1)
    var np_arr_1d = np.arange(0, 10, dtype=np.float32)

    # Test positive single index access (already working)
    check(nm_arr_1d[0], np_arr_1d[0], "1D positive index [0] failed")
    check(nm_arr_1d[5], np_arr_1d[5], "1D positive index [5] failed")

    # 2D array positive indexing
    var nm_arr_2d = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(nm.Shape(3, 4))
    var np_arr_2d = np.arange(0, 12, dtype=np.float32).reshape(3, 4)

    check(nm_arr_2d[0], np_arr_2d[0], "2D positive row index [0] failed")
    check(nm_arr_2d[2], np_arr_2d[2], "2D positive row index [2] failed")


def test_positive_slice_indices():
    """Test positive indices in slice operations."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1).reshape(nm.Shape(2, 3, 4))
    var np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test positive start indices
    nm_slice1 = nm_arr[1:, :, :]
    np_sliced1 = np_arr[1:, :, :]
    check(nm_slice1, np_sliced1, "Positive start index [1:, :, :] failed")

    # Test positive end indices
    nm_slice2 = nm_arr[0:1, :, :]
    np_sliced2 = np_arr[0:1, :, :]
    check(nm_slice2, np_sliced2, "Positive end index [0:1, :, :] failed")

    # Test positive start and end
    nm_slice3 = nm_arr[0:2, 1:3, 2:4]
    np_sliced3 = np_arr[0:2, 1:3, 2:4]
    check(nm_slice3, np_sliced3, "Positive start/end [0:2, 1:3, 2:4] failed")


def test_slice_mixed_dimensions():
    """Test slicing across multiple dimensions with positive indices."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1).reshape(nm.Shape(2, 3, 4))
    var np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Mixed positive indices across dimensions
    nm_slice1 = nm_arr[1:, 1:, 1:]
    np_sliced1 = np_arr[1:, 1:, 1:]
    check(nm_slice1, np_sliced1, "Mixed positive indices [1:, 1:, 1:] failed")

    # Mixed with full ranges
    nm_slice2 = nm_arr[0:1, :, 1:3]
    np_sliced2 = np_arr[0:1, :, 1:3]
    check(nm_slice2, np_sliced2, "Mixed ranges [0:1, :, 1:3] failed")


def test_positive_step_slicing():
    """Test forward slicing with positive steps."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(nm.Shape(3, 4))
    var np_arr = np.arange(0, 12, dtype=np.float32).reshape(3, 4)

    # Forward step patterns
    nm_slice1 = nm_arr[::2, :]
    np_sliced1 = np_arr[::2, :]
    check(nm_slice1, np_sliced1, "Forward step rows [::2, :] failed")

    # Step with bounds
    nm_slice2 = nm_arr[0:3:2, 1:4:2]
    np_sliced2 = np_arr[0:3:2, 1:4:2]
    check(nm_slice2, np_sliced2, "Step with bounds [0:3:2, 1:4:2] failed")


def test_slice_step_variations():
    """Test various positive step sizes and patterns."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 20.0, step=1).reshape(nm.Shape(4, 5))
    var np_arr = np.arange(0, 20, dtype=np.float32).reshape(4, 5)

    # Different step sizes
    nm_slice1 = nm_arr[::3, ::2]
    np_sliced1 = np_arr[::3, ::2]
    check(nm_slice1, np_sliced1, "Step sizes [::3, ::2] failed")

    # Step with start/end
    nm_slice2 = nm_arr[1::2, 2::2]
    np_sliced2 = np_arr[1::2, 2::2]
    check(nm_slice2, np_sliced2, "Step with start [1::2, 2::2] failed")


# def test_boundary_within_limits():
#     """Test boundary conditions within array limits."""
#     var np = Python.import_module("numpy")

#     var nm_arr = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(nm.Shape(3, 4))
#     var np_arr = np.arange(0, 12, dtype=np.float32).reshape(3, 4)

#     # Start from beginning
#     nm_slice1 = nm_arr[0:, 0:]
#     np_sliced1 = np_arr[0:, 0:]
#     check(nm_slice1, np_sliced1, "From beginning [0:, 0:] failed")

#     # Up to end
#     nm_slice2 = nm_arr[:3, :4]
#     np_sliced2 = np_arr[:3, :4]
#     check(nm_slice2, np_sliced2, "Up to end [:3, :4] failed")

#     # Single element slices
#     nm_slice3 = nm_arr[1:2, 2:3]
#     np_sliced3 = np_arr[1:2, 2:3]
#     check(nm_slice3, np_sliced3, "Single element [1:2, 2:3] failed")


def test_1d_array_slicing_positive():
    """Comprehensive tests for 1D array slicing with positive indices."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 10.0, step=1)
    var np_arr = np.arange(0, 10, dtype=np.float32)

    # Basic slicing
    nm_slice1 = nm_arr[2:7]
    np_sliced1 = np_arr[2:7]
    check(nm_slice1, np_sliced1, "1D basic slice [2:7] failed")

    # With step
    nm_slice2 = nm_arr[Slice(1, 8, 2)]
    np_sliced2 = np_arr[1:8:2]
    check(nm_slice2, np_sliced2, "1D step slice [1:8:2] failed")

    # From start
    nm_slice3 = nm_arr[:5]
    np_sliced3 = np_arr[:5]
    check(nm_slice3, np_sliced3, "1D from start [:5] failed")

    # To end
    nm_slice4 = nm_arr[3:]
    np_sliced4 = np_arr[3:]
    check(nm_slice4, np_sliced4, "1D to end [3:] failed")


def test_3d_array_positive_slicing():
    """Advanced 3D array slicing tests with positive indices."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 60.0, step=1).reshape(nm.Shape(3, 4, 5))
    var np_arr = np.arange(0, 60, dtype=np.float32).reshape(3, 4, 5)

    # Complex mixed slicing with positive indices
    nm_slice1 = nm_arr[1:, 1:3, ::2]
    np_sliced1 = np_arr[1:, 1:3, ::2]
    check(nm_slice1, np_sliced1, "3D complex slice [1:, 1:3, ::2] failed")

    # Alternating patterns
    nm_slice2 = nm_arr[::2, :, 1::2]
    np_sliced2 = np_arr[::2, :, 1::2]
    check(nm_slice2, np_sliced2, "3D alternating [::2, :, 1::2] failed")


def test_f_order_array_slicing():
    """Test slicing with F-order (Fortran-order) arrays."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(
        nm.Shape(3, 4), order="F"
    )
    var np_arr = np.arange(0, 12, dtype=np.float32).reshape(3, 4, order="F")

    # Basic F-order slicing
    nm_slice1 = nm_arr[1:, 1:]
    np_sliced1 = np_arr[1:, 1:]
    check(nm_slice1, np_sliced1, "F-order positive slicing [1:, 1:] failed")

    # Step slicing in F-order
    nm_slice2 = nm_arr[::2, 1::2]
    np_sliced2 = np_arr[::2, 1::2]
    check(nm_slice2, np_sliced2, "F-order step [::2, 1::2] failed")


# def test_edge_case_valid_slices():
#     """Test edge cases that should work with current implementation."""
#     var np = Python.import_module("numpy")

#     var nm_arr = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(nm.Shape(3, 4))
#     var np_arr = np.arange(0, 12, dtype=np.float32).reshape(3, 4)

#     # Full array slice
#     nm_slice1 = nm_arr[:, :]
#     np_sliced1 = np_arr[:, :]
#     check(nm_slice1, np_sliced1, "Full array slice [:, :] failed")

#     # First/last elements
#     nm_slice2 = nm_arr[0:1, 0:1]
#     np_sliced2 = np_arr[0:1, 0:1]
#     check(nm_slice2, np_sliced2, "First element [0:1, 0:1] failed")

#     nm_slice3 = nm_arr[2:3, 3:4]
#     np_sliced3 = np_arr[2:3, 3:4]
#     check(nm_slice3, np_sliced3, "Last element [2:3, 3:4] failed")


def test_negative_indices_basic():
    """Test basic negative indexing similar to Python/NumPy."""
    var np = Python.import_module("numpy")

    # 1D array negative indexing
    var nm_arr_1d = nm.arange[nm.f32](0.0, 10.0, step=1)
    var np_arr_1d = np.arange(0, 10, dtype=np.float32)

    # Test negative single index access
    check(nm_arr_1d[-1], np_arr_1d[-1], "1D negative index [-1] failed")
    check(nm_arr_1d[-5], np_arr_1d[-5], "1D negative index [-5] failed")

    # 2D array negative indexing
    var nm_arr_2d = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(nm.Shape(3, 4))
    var np_arr_2d = np.arange(0, 12, dtype=np.float32).reshape(3, 4)

    check(nm_arr_2d[-1], np_arr_2d[-1], "2D negative row index [-1] failed")
    check(nm_arr_2d[-2], np_arr_2d[-2], "2D negative row index [-2] failed")


# def test_negative_slice_indices():
#     """Test negative indices in slice operations."""
#     var np = Python.import_module("numpy")

#     var nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1).reshape(nm.Shape(2, 3, 4))
#     var np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

#     # Test negative start indices
#     nm_slice1 = nm_arr[-1:, :, :]
#     np_sliced1 = np_arr[-1:, :, :]
#     check(nm_slice1, np_sliced1, "Negative start index [-1:, :, :] failed")

#     # Test negative end indices
#     nm_slice2 = nm_arr[:-1, :, :]
#     np_sliced2 = np_arr[:-1, :, :]
#     check(nm_slice2, np_sliced2, "Negative end index [:-1, :, :] failed")

# # Test negative start and end
# nm_slice3 = nm_arr[-2:-1, :, :]
# np_sliced3 = np.take(np_arr, np.arange(-2, -1), axis=0)
# check(nm_slice3, np_sliced3, "Negative start/end [-2:-1, :, :] failed")


# def test_negative_slice_mixed_dimensions():
#     """Test negative slicing across multiple dimensions."""
#     var np = Python.import_module("numpy")

#     var nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1).reshape(nm.Shape(2, 3, 4))
#     var np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

#     # Mixed negative indices across dimensions
#     nm_slice1 = nm_arr[-1:, -2:, -3:]
#     np_sliced1 = np_arr[-1:, -2:, -3:]
#     check(nm_slice1, np_sliced1, "Mixed negative indices [-1:, -2:, -3:] failed")

#     # Mixed positive and negative
#     nm_slice2 = nm_arr[0:-1, -2:2, 1:-1]
#     np_sliced2 = np_arr[0:-1, -2:2, 1:-1]
#     check(nm_slice2, np_sliced2, "Mixed pos/neg indices [0:-1, -2:2, 1:-1] failed")


# def test_negative_step_slicing():
#     """Test reverse slicing with negative steps."""
#     var np = Python.import_module("numpy")

#     var nm_arr = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(nm.Shape(3, 4))
#     var np_arr = np.arange(0, 12, dtype=np.float32).reshape(3, 4)

#     # Reverse entire array
#     nm_slice1 = nm_arr[::-1, :]
#     np_sliced1 = np_arr[::-1, :]
#     check_is_close(nm_slice1, np_sliced1, "Reverse rows [::-1, :] failed")

#     # Reverse columns
#     nm_slice2 = nm_arr[:, ::-1]
#     np_sliced2 = np_arr[:, ::-1]
#     check_is_close(nm_slice2, np_sliced2, "Reverse columns [:, ::-1] failed")

#     # Reverse both dimensions
#     nm_slice3 = nm_arr[::-1, ::-1]
#     np_sliced3 = np_arr[::-1, ::-1]
#     check_is_close(nm_slice3, np_sliced3, "Reverse both [::-1, ::-1] failed")

#     # Step with negative indices
#     nm_slice4 = nm_arr[-1::-2, :]
#     np_sliced4 = np_arr[-1::-2, :]
#     check_is_close(nm_slice4, np_sliced4, "Negative step with neg start [-1::-2, :] failed")


def test_slice_step_variations_positive():
    """Test various step sizes and patterns with positive indices."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 20.0, step=1).reshape(nm.Shape(4, 5))
    var np_arr = np.arange(0, 20, dtype=np.float32).reshape(4, 5)

    # Different step sizes
    nm_slice1 = nm_arr[::3, ::2]
    np_sliced1 = np_arr[::3, ::2]
    check(nm_slice1, np_sliced1, "Step sizes [::3, ::2] failed")

    # Step with start/end
    nm_slice2 = nm_arr[1::2, 2::2]
    np_sliced2 = np_arr[1::2, 2::2]
    check(nm_slice2, np_sliced2, "Step with start [1::2, 2::2] failed")


# def test_boundary_edge_cases_safe():
#     """Test edge cases and boundary conditions that work with current implementation.
#     """
#     var np = Python.import_module("numpy")

#     var nm_arr = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(nm.Shape(3, 4))
#     var np_arr = np.arange(0, 12, dtype=np.float32).reshape(3, 4)

#     # Single element slices
#     nm_slice1 = nm_arr[1:2, 1:2]
#     np_sliced1 = np_arr[1:2, 1:2]
#     check(nm_slice1, np_sliced1, "Single element slice [1:2, 1:2] failed")

#     # Start from beginning
#     nm_slice2 = nm_arr[0:, 0:]
#     np_sliced2 = np_arr[0:, 0:]
#     check(nm_slice2, np_sliced2, "From beginning [0:, 0:] failed")


def test_1d_array_slicing_basic():
    """Basic tests for 1D array slicing with current implementation."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 10.0, step=1)
    var np_arr = np.arange(0, 10, dtype=np.float32)

    # Basic slicing
    nm_slice1 = nm_arr[2:7]
    np_sliced1 = np_arr[2:7]
    check(nm_slice1, np_sliced1, "1D basic slice [2:7] failed")

    # With step
    nm_slice2 = nm_arr[Slice(1, 8, 2)]
    np_sliced2 = np_arr[1:8:2]
    check(nm_slice2, np_sliced2, "1D step slice [1:8:2] failed")

    # From start
    nm_slice3 = nm_arr[:5]
    np_sliced3 = np_arr[:5]
    check(nm_slice3, np_sliced3, "1D from start [:5] failed")


def test_3d_array_basic_slicing():
    """Basic 3D array slicing tests with positive indices."""
    var np = Python.import_module("numpy")

    var nm_arr = nm.arange[nm.f32](0.0, 60.0, step=1).reshape(nm.Shape(3, 4, 5))
    var np_arr = np.arange(0, 60, dtype=np.float32).reshape(3, 4, 5)

    # Basic slicing
    nm_slice1 = nm_arr[1:, 1:3, ::2]
    np_sliced1 = np_arr[1:, 1:3, ::2]
    check(nm_slice1, np_sliced1, "3D basic slice [1:, 1:3, ::2] failed")

    # Alternating patterns
    nm_slice2 = nm_arr[::2, :, 1::2]
    np_sliced2 = np_arr[::2, :, 1::2]
    check(nm_slice2, np_sliced2, "3D alternating [::2, :, 1::2] failed")


# def test_slice_with_basic_dtypes():
#     """Test slicing with different data types using basic operations."""
#     var np = Python.import_module("numpy")

#     # Test with integers
#     var nm_arr_int = nm.arange[nm.i32](0, 12, step=1).reshape(nm.Shape(3, 4))
#     var np_arr_int = np.arange(0, 12, dtype=np.int32).reshape(3, 4)

#     nm_slice_int = nm_arr_int[1:, 1:]
#     np_sliced_int = np_arr_int[1:, 1:]
#     check(nm_slice_int, np_sliced_int, "Integer array positive slicing failed")

#     # Test with different float precision
#     var nm_arr_f64 = nm.arange[nm.f64](0.0, 8.0, step=1).reshape(nm.Shape(2, 4))
#     var np_arr_f64 = np.arange(0, 8, dtype=np.float64).reshape(2, 4)

#     nm_slice_f64 = nm_arr_f64[::-1, 1:-1]
#     np_sliced_f64 = np_arr_f64[::-1, 1:-1]
#     check(nm_slice_f64, np_sliced_f64, "Float64 array slicing failed")


# def test_f_order_array_slicing():
#     """Test slicing with F-order (Fortran-order) arrays."""
#     var np = Python.import_module("numpy")

#     var nm_arr = nm.arange[nm.f32](0.0, 12.0, step=1).reshape(nm.Shape(3, 4), order="F")
#     var np_arr = np.arange(0, 12, dtype=np.float32).reshape(3, 4, order="F")

#     # Basic F-order slicing
#     nm_slice1 = nm_arr[-1:, -2:]
#     np_sliced1 = np_arr[-1:, -2:]
#     check(nm_slice1, np_sliced1, "F-order negative slicing [-1:, -2:] failed")

#     # Reverse F-order slicing
#     nm_slice2 = nm_arr[::-1, ::-1]
#     np_sliced2 = np_arr[::-1, ::-1]
#     check(nm_slice2, np_sliced2, "F-order reverse [::-1, ::-1] failed")

#     # Step slicing in F-order
#     nm_slice3 = nm_arr[::2, 1::2]
#     np_sliced3 = np_arr[::2, 1::2]
#     check(nm_slice3, np_sliced3, "F-order step [::2, 1::2] failed")
