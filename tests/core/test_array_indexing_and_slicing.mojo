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
    var arr = nm.NDArray(Shape(4, 4))
    var np_arr = arr.to_numpy()
    arr.itemset(List(2, 2), 1000)
    np_arr[2, 2] = 1000
    check_is_close(arr, np_arr, "Itemset is broken")


def test_slicing_getter1():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr = nm_arr.reshape(Shape(2, 3, 4), order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 1: Slicing all dimensions
    nm_slice1 = nm_arr[:, :, 1:2]
    np_sliced1 = np.take(
        np.take(
            np.take(np_arr, np.arange(0, 2), axis=0), np.arange(0, 3), axis=1
        ),
        np.arange(1, 2),
        axis=2,
    )
    np_sliced1 = np.squeeze(np_sliced1, axis=2)
    check(nm_slice1, np_sliced1, "3D array slicing (C-order) [:, :, 1:2]")


def test_slicing_getter2():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr = nm_arr.reshape(Shape(2, 3, 4), order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 2: Slicing with start and end indices
    nm_slice2 = nm_arr[0:1, 1:3, 2:4]
    np_sliced2 = np.take(
        np.take(
            np.take(np_arr, np.arange(0, 1), axis=0), np.arange(1, 3), axis=1
        ),
        np.arange(2, 4),
        axis=2,
    )
    check(nm_slice2, np_sliced2, "3D array slicing (C-order) [0:1, 1:3, 2:4]")


def test_slicing_getter3():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr = nm_arr.reshape(Shape(2, 3, 4), order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 3: Slicing with mixed start, end, and step values
    nm_slice3 = nm_arr[1:, 0:2, ::2]
    np_sliced3 = np.take(
        np.take(
            np.take(np_arr, np.arange(1, np_arr.shape[0]), axis=0),
            np.arange(0, 2),
            axis=1,
        ),
        np.arange(0, np_arr.shape[2], 2),
        axis=2,
    )
    check(nm_slice3, np_sliced3, "3D array slicing (C-order) [1:, 0:2, ::2]")


def test_slicing_getter4():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr = nm_arr.reshape(Shape(2, 3, 4), order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 4: Slicing with step
    nm_slice4 = nm_arr[::2, ::2, ::2]
    np_sliced4 = np.take(
        np.take(
            np.take(np_arr, np.arange(0, np_arr.shape[0], 2), axis=0),
            np.arange(0, np_arr.shape[1], 2),
            axis=1,
        ),
        np.arange(0, np_arr.shape[2], 2),
        axis=2,
    )
    check(nm_slice4, np_sliced4, "3D array slicing (C-order) [::2, ::2, ::2]")


def test_slicing_getter5():
    var np = Python.import_module("numpy")

    # Test C-order array slicing
    nm_arr = nm.arange[nm.f32](0.0, 24.0, step=1)
    nm_arr = nm_arr.reshape(Shape(2, 3, 4), order="C")
    np_arr = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

    # Test case 5: Slicing with combination of integer and slices
    nm_slice5 = nm_arr[1:2, :, 1:3]
    np_sliced5 = np.take(
        np.take(np_arr[1], np.arange(0, np_arr.shape[1]), axis=0),
        np.arange(1, 3),
        axis=1,
    )
    check(nm_slice5, np_sliced5, "3D array slicing (C-order) [1, :, 1:3]")


def test_slicing_getter6():
    var np = Python.import_module("numpy")

    var b = nm.arange[i8](60).reshape(Shape(3, 4, 5))
    var ind = nm.array[isize]("[[2,0,1], [1,0,1]]")
    var mask = nm.array[boolean]("[1,0,1]")

    var bnp = b.to_numpy()
    var indnp = ind.to_numpy()
    var masknp = mask.to_numpy()

    check(b[ind], bnp[indnp], "Get by indices array fails")
    check(b[mask], bnp[masknp], "Get by mask array fails")


def test_getitem_single_axis_basic():
    var np = Python.import_module("numpy")
    var a = nm.arange[i32](0, 12, 1).reshape(Shape(3, 4))
    var anp = np.arange(12, dtype=np.int32).reshape(3, 4)
    # positive index
    check(a[1], anp[1], "__getitem__(idx: Int) positive index row slice broken")
    # negative index
    check(
        a[-1], anp[-1], "__getitem__(idx: Int) negative index row slice broken"
    )


def test_getitem_single_axis_1d_scalar():
    var np = Python.import_module("numpy")
    var a = nm.arange[i16](0, 6, 1).reshape(Shape(6))
    var anp = np.arange(6, dtype=np.int16)
    # 1-D -> 0-D scalar wrapper
    check(a[2], anp[2], "__getitem__(idx: Int) 1-D to scalar (0-D) broken")


def test_getitem_single_axis_f_order():
    var np = Python.import_module("numpy")
    var a = nm.arange[i32](0, 12, 1).reshape(Shape(3, 4), order="F")
    var anp = np.arange(12, dtype=np.int32).reshape(3, 4, order="F")
    check(a[0], anp[0], "__getitem__(idx: Int) F-order first row broken")
    check(a[2], anp[2], "__getitem__(idx: Int) F-order last row broken")


def test_setitem_single_axis_basic():
    var np = Python.import_module("numpy")
    var a = nm.arange[i32](0, 12, 1).reshape(Shape(3, 4))
    var anp = np.arange(12, dtype=np.int32).reshape(3, 4)
    var row = nm.full[i32](Shape(4), fill_value=Scalar[i32](999))
    a[1] = row
    anp[1] = 999
    check(a, anp, "__setitem__(idx: Int, val) C-order assignment broken")
    # negative index assignment
    var row2 = nm.full[i32](Shape(4), fill_value=Scalar[i32](-5))
    a[-1] = row2
    anp[-1] = -5
    check(a, anp, "__setitem__(idx: Int, val) negative index assignment broken")


def test_setitem_single_axis_f_order():
    var np = Python.import_module("numpy")
    var a = nm.arange[i32](0, 12, 1).reshape(Shape(3, 4), order="F")
    var anp = np.arange(12, dtype=np.int32).reshape(3, 4, order="F")
    var row = nm.full[i32](Shape(4), fill_value=Scalar[i32](111))
    a[0] = row
    anp[0] = 111
    check(a, anp, "__setitem__(idx: Int, val) F-order assignment broken")


def test_setitem_single_axis_shape_mismatch_error():
    # Ensure shape mismatch raises an error (val shape != self.shape[1:])
    var a = nm.arange[i32](0, 12, 1).reshape(Shape(3, 4))
    var bad = nm.full[i32](Shape(5), fill_value=Scalar[i32](1))  # wrong length
    var raised: Bool = False
    try:
        a[0] = bad
    except e:
        raised = True
    assert_true(
        raised, "__setitem__(idx: Int, val) did not raise on shape mismatch"
    )


def test_setitem_single_axis_index_oob_error():
    # Ensure out-of-bounds index raises an error
    var a = nm.arange[i32](0, 12, 1).reshape(Shape(3, 4))
    var row = nm.full[i32](Shape(4), fill_value=Scalar[i32](7))
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
#     np.put(np_arr, np.ravel_multi_index((np.arange(1, 2), np.arange(1, 3), np.arange(2, 4)), np_arr.shape), np_set_arr.flatten())
#     check(nm_arr, np_arr, "3D array slice setting (C-order) [1:2, 1:3, 2:4] = array")
