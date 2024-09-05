import numojo as nm
from numojo import *
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close

# def test_slicing():
#     w = arange[f32](0.0, 24.0, step=1)
#     w.reshape(2, 3, 4, order="C")
#     assert_true(w[0,0,0] == 0 and w[1,2,3] == 23, "3D array indexing")

#     slicedw = w[0:1, :, 1:2]
#     assert_true(slicedw.ndshape == (1,3,1), "3D array slicing shape")
#     assert_true(slicedw[0,0,0] == 1 and slicedw[0,2,0] == 9, "3D array slicing values")

#     y = arange[numojo.f32](0.0, 24.0, step=1)
#     y.reshape(2,3,4, order="F")
#     slicedy = y[:, :, 1:2]
#     assert_true(slicedy.ndshape == (2,3,1), "3D array slicing shape (F order)")
#     assert_true(slicedy[0,0,0] == 1 and slicedy[1,2,0] == 13, "3D array slicing values (F order)")
