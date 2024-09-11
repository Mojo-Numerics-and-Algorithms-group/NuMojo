import numojo as nm
from numojo import *
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close


def test_constructors():
    # Test NDArray constructor with different input types
    var arr1 = NDArray[f32](3, 4, 5)
    assert_true(arr1.ndim == 3, "NDArray constructor: ndim")
    assert_true(arr1.ndshape[0] == 3, "NDArray constructor: shape element 0")
    assert_true(arr1.ndshape[1] == 4, "NDArray constructor: shape element 1")
    assert_true(arr1.ndshape[2] == 5, "NDArray constructor: shape element 2")
    assert_true(arr1.ndshape.ndsize == 60, "NDArray constructor: size")
    assert_true(arr1.dtype == DType.float32, "NDArray constructor: dtype")

    var arr2 = NDArray[f32](VariadicList[Int](3, 4, 5))
    assert_true(
        arr2.ndshape[0] == 3,
        "NDArray constructor with VariadicList: shape element 0",
    )
    assert_true(
        arr2.ndshape[1] == 4,
        "NDArray constructor with VariadicList: shape element 1",
    )
    assert_true(
        arr2.ndshape[2] == 5,
        "NDArray constructor with VariadicList: shape element 2",
    )

    var arr3 = NDArray[f32](VariadicList[Int](3, 4, 5), fill=Scalar[f32](10.0))
    # maybe it's better to return a scalar for arr[0, 0, 0]
    assert_equal(
        arr3[0, 0, 0].get_scalar(0), 10.0, "NDArray constructor with fill value"
    )

    var arr4 = NDArray[f32](List[Int](3, 4, 5))
    assert_true(
        arr4.ndshape[0] == 3, "NDArray constructor with List: shape element 0"
    )
    assert_true(
        arr4.ndshape[1] == 4, "NDArray constructor with List: shape element 1"
    )
    assert_true(
        arr4.ndshape[2] == 5, "NDArray constructor with List: shape element 2"
    )

    var arr5 = NDArray[f32](NDArrayShape(3, 4, 5))
    assert_true(
        arr5.ndshape[0] == 3,
        "NDArray constructor with NDArrayShape: shape element 0",
    )
    assert_true(
        arr5.ndshape[1] == 4,
        "NDArray constructor with NDArrayShape: shape element 1",
    )
    assert_true(
        arr5.ndshape[2] == 5,
        "NDArray constructor with NDArrayShape: shape element 2",
    )

    var arr6 = NDArray[f32](
        data=List[SIMD[f32, 1]](1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        shape=List[Int](2, 5),
    )
    assert_true(
        arr6.ndshape[0] == 2,
        "NDArray constructor with data and shape: shape element 0",
    )
    assert_true(
        arr6.ndshape[1] == 5,
        "NDArray constructor with data and shape: shape element 1",
    )
    assert_equal(
        arr6[1, 4].get_scalar(0),
        10.0,
        "NDArray constructor with data: value check",
    )
