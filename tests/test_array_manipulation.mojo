import numojo as nm
from numojo import *
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close

def test_arr_manipulation():
    var A = arange[i16](1, 7, 1)
    assert_true(A.ndshape[0] == 6, "Arange shape: dimension 0")
    assert_equal(A[0].get_scalar(0), SIMD[i16, 1](1), "Arange values 0")
    assert_equal(A[5].get_scalar(0), SIMD[i16, 1](6), "Arange values 5")

    temp = flip(A)
    assert_equal(temp[0].get_scalar(0), SIMD[i16, 1](6), "Flip operation at index 0")
    assert_equal(temp[5].get_scalar(0), SIMD[i16, 1](1), "Flip operation at index 5")

    A.reshape(2, 3, order="F")
    assert_true(A.ndshape[0] == 2, "Reshape operation: check dimension 0")
    assert_true(A.ndshape[1] == 3, "Reshape operation: check dimension 1")

    var B = arange[i16](0, 12, 1)
    B.reshape(3, 2, 2, order="F")
    ravel(B, order="C")
    assert_true(B.ndshape[0] == 12, "Ravel operation")
