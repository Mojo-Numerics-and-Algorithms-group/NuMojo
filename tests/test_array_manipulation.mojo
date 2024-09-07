import numojo as nm
from numojo import *
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close
from python import Python

def test_arr_manipulation():
    var np = Python.import_module("numpy")
    
    # Test arange
    var A = nm.arange[nm.i16](1, 7, 1)
    var np_A = np.arange(1, 7, 1, dtype=np.int16)
    check_is_close(A, np_A, "Arange operation")

    # Test flip
    var flipped_A = nm.flip(A)
    var np_flipped_A = np.flip(np_A)
    check_is_close(flipped_A, np_flipped_A, "Flip operation")

    # Test reshape
    A.reshape(2, 3)
    np_A = np_A.reshape((2, 3))
    check_is_close(A, np_A, "Reshape operation")

    # # Test ravel
    # var B = nm.arange[nm.i16](0, 12, 1)
    # var np_B = np.arange(0, 12, 1, dtype=np.int16)
    # B.reshape(3, 2, 2, order="F")
    # np_B = np_B.reshape((3, 2, 2), order='F')
    # var raveled_B = nm.ravel(B, order="C")
    # var np_raveled_B = np.ravel(np_B, order='C')
    # check_is_close(raveled_B, np_raveled_B, "Ravel operation")
