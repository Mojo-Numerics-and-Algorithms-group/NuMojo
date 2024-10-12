import numojo as nm
from numojo import *
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check
from python import Python

# TODO: there's something wrong with bool comparision even though result looks same.
def test_bool_masks_gt():
    var np = Python.import_module("numpy")
    var np_A = np.arange(0, 24, dtype=np.int16).reshape((3, 2, 4))
    var A = nm.arange[nm.i16](0, 24)
    A.reshape(3, 2, 4)

    var np_gt = np_A > 10
    var gt = A > Scalar[nm.i16](10)
    check(gt, np_gt, "Greater than mask")

    # Test greater than or equal
    var np_ge = np_A >= 10
    var ge = A >= Scalar[nm.i16](10)
    check(ge, np_ge, "Greater than or equal mask")

def test_bool_masks_lt():
    var np = Python.import_module("numpy")

    # Create NumPy and NuMojo arrays using arange and reshape
    var np_A = np.arange(0, 24, dtype=np.int16).reshape((3, 2, 4))
    var A = nm.arange[nm.i16](0, 24)
    A.reshape(3, 2, 4)

    # Test less than
    var np_lt = np_A < 10
    var lt = A < Scalar[nm.i16](10)
    check(lt, np_lt, "Less than mask")

    # Test less than or equal
    var np_le = np_A <= 10
    var le = A <= Scalar[nm.i16](10)
    check(le, np_le, "Less than or equal mask")

def test_bool_masks_eq():
    var np = Python.import_module("numpy")

    # Create NumPy and NuMojo arrays using arange and reshape
    var np_A = np.arange(0, 24, dtype=np.int16).reshape((3, 2, 4))
    var A = nm.arange[nm.i16](0, 24)
    A.reshape(3, 2, 4)

    # Test equal
    var np_eq = np_A == 10
    var eq = A == Scalar[nm.i16](10)
    check(eq, np_eq, "Equal mask")

    # Test not equal
    var np_ne = np_A != 10
    var ne = A != Scalar[nm.i16](10)
    check(ne, np_ne, "Not equal mask")

    # Test masked array
    var np_mask = np_A[np_A > 10]
    var mask = A[A > Scalar[nm.i16](10)]
    check(mask, np_mask, "Masked array")
