import numojo as nm
from numojo import *
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close

def test_bool_masks():
    var A = nm.core.random.rand[i16](3, 2, 2)
    var B = nm.core.random.rand[i16](3, 2, 2)
    
    gt = A > 10.0
    assert_true(gt.dtype == DType.bool, "Greater than mask dtype")

    ge = A >= Scalar[i16](10)
    assert_true(ge.dtype == DType.bool, "Greater than or equal mask dtype")

    lt = A < Scalar[i16](10)
    assert_true(lt.dtype == DType.bool, "Less than mask dtype")

    le = A <= Scalar[i16](10)
    assert_true(le.dtype == DType.bool, "Less than or equal mask dtype")

    eq = A == Scalar[i16](10)
    assert_true(eq.dtype == DType.bool, "Equal mask dtype")

    ne = A != Scalar[i16](10)
    assert_true(ne.dtype == DType.bool, "Not equal mask dtype")

    mask = A[A > Scalar[i16](10)]
    assert_true(mask.ndshape[0] <= A.size(), "Masked array size")