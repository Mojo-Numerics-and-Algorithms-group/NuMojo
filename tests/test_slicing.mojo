# import numojo as nm
# from numojo import *
# from testing.testing import assert_true, assert_almost_equal, assert_equal
# from utils_for_test import check, check_is_close


# def test_slicing():
#     var np = Python.import_module("numpy")

#     # Test C-order array slicing
#     w = nm.arange[nm.f32](0.0, 24.0, step=1)
#     w.reshape(2, 3, 4, order="C")
#     np_w = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4)

#     # Test basic indexing
#     check_is_close(
#         w[0, 0, 0], np_w[0, 0, 0], "3D array indexing (C-order) [0,0,0]"
#     )
#     check_is_close(
#         w[1, 2, 3], np_w[1, 2, 3], "3D array indexing (C-order) [1,2,3]"
#     )

#     # Test slicing
#     slicedw = w[0:1, :, 1:2]
#     # var py_list: PythonObject = [0:1, :, 1:2]
#     np_slicedw = np_w[PythonObject0:1, :, 1:2]
#     check_is_close(slicedw, np_slicedw, "3D array slicing (C-order)")

#     # Test F-order array slicing
#     y = nm.arange[nm.f32](0.0, 24.0, step=1)
#     y.reshape(2, 3, 4, order="F")
#     np_y = np.arange(0, 24, dtype=np.float32).reshape(2, 3, 4, order="F")

#     # Test basic indexing
#     check_is_close(
#         y[0, 0, 0], np_y[0, 0, 0], "3D array indexing (F-order) [0,0,0]"
#     )
#     check_is_close(
#         y[1, 2, 3], np_y[1, 2, 3], "3D array indexing (F-order) [1,2,3]"
#     )

#     # Test slicing
#     slicedy = y[:, :, 1:2]
#     np_slicedy = np_y[:, :, 1:2]
#     check_is_close(slicedy, np_slicedy, "3D array slicing (F-order)")

#     # Test integer array
#     z = nm.arange[nm.i32](0, 24, step=1)
#     z.reshape(2, 3, 4, order="C")
#     np_z = np.arange(0, 24, dtype=np.int32).reshape(2, 3, 4, order="C")

#     # Test slicing for integer array
#     slicedz = z[1:2, 0:2, :]
#     np_slicedz = np_z[1:2, 0:2, :]
#     check(slicedz, np_slicedz, "3D integer array slicing (C-order)")
