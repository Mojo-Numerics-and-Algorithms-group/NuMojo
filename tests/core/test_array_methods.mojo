from python import Python, PythonObject

from numojo.prelude import *
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close, check_values_close
from testing import TestSuite


def test_constructors():
    # Test NDArray constructor with different input types
    var arr1 = NDArray[f32](Shape(3, 4, 5))
    assert_true(arr1.ndim == 3, "NDArray constructor: ndim")
    assert_true(arr1.shape[0] == 3, "NDArray constructor: shape element 0")
    assert_true(arr1.shape[1] == 4, "NDArray constructor: shape element 1")
    assert_true(arr1.shape[2] == 5, "NDArray constructor: shape element 2")
    assert_true(arr1.size == 60, "NDArray constructor: size")
    assert_true(arr1.dtype == DType.float32, "NDArray constructor: dtype")

    var arr2 = NDArray[f32](VariadicList[Int](3, 4, 5))
    assert_true(
        arr2.shape[0] == 3,
        "NDArray constructor with VariadicList: shape element 0",
    )
    assert_true(
        arr2.shape[1] == 4,
        "NDArray constructor with VariadicList: shape element 1",
    )
    assert_true(
        arr2.shape[2] == 5,
        "NDArray constructor with VariadicList: shape element 2",
    )

    var arr3 = nm.full[f32](Shape(3, 4, 5), fill_value=Scalar[f32](10.0))
    # maybe it's better to return a scalar for arr[0, 0, 0]
    assert_equal(
        arr3[Item(0, 0, 0)], 10.0, "NDArray constructor with fill value"
    )

    var values: List[Int] = [3, 4, 5]
    var arr4 = NDArray[f32](values^)
    assert_true(
        arr4.shape[0] == 3, "NDArray constructor with List: shape element 0"
    )
    assert_true(
        arr4.shape[1] == 4, "NDArray constructor with List: shape element 1"
    )
    assert_true(
        arr4.shape[2] == 5, "NDArray constructor with List: shape element 2"
    )

    var arr5 = NDArray[f32](NDArrayShape(3, 4, 5))
    assert_true(
        arr5.shape[0] == 3,
        "NDArray constructor with NDArrayShape: shape element 0",
    )
    assert_true(
        arr5.shape[1] == 4,
        "NDArray constructor with NDArrayShape: shape element 1",
    )
    assert_true(
        arr5.shape[2] == 5,
        "NDArray constructor with NDArrayShape: shape element 2",
    )


def test_iterator():
    var py = Python.import_module("builtins")
    var np = Python.import_module("numpy")

    var a = nm.arange[i8](24).reshape(Shape(2, 3, 4))
    var anp = a.to_numpy()
    var f = nm.arange[i8](24).reshape(Shape(2, 3, 4), order="F")
    var fnp = f.to_numpy()

    # NDAxisIter
    var a_iter_along_axis = a.iter_along_axis[forward=False](axis=0)
    var b = a_iter_along_axis.__next__() == nm.array[i8]("[11, 23]")
    assert_true(
        b.item(0) == True,
        "`_NDAxisIter` breaks",
    )
    assert_true(
        b.item(1) == True,
        "`_NDAxisIter` breaks",
    )

    # NDArrayIter
    var a_iter_over_dimension = a.__iter__()
    var anp_iter_over_dimension = anp.__iter__()
    for _ in range(a.shape[0]):
        check(
            a_iter_over_dimension.__next__(),
            anp_iter_over_dimension.__next__(),
            "`_NDArrayIter` or `__iter__()` breaks",
        )

    var a_iter_over_dimension_reversed = a.__reversed__()
    var anp_iter_over_dimension_reversed = py.reversed(anp)
    for _ in range(a.shape[0]):
        check(
            a_iter_over_dimension_reversed.__next__(),
            anp_iter_over_dimension_reversed.__next__(),
            "`_NDArrayIter` or `__reversed__()` breaks",
        )

    # NDIter of C-order array
    var a_nditer = a.nditer()
    var anp_nditer = np.nditer(anp)
    for _ in range(a.size):
        check_values_close(
            a_nditer.__next__(),
            anp_nditer.__next__(),
            "`_NDIter` or `nditer()` of C array by order C breaks",
        )

    # NDIter of C-order array
    a_nditer = a.nditer()
    anp_nditer = np.nditer(anp)
    for i in range(a.size):
        check_values_close(
            a_nditer.ith(i),
            anp_nditer.__next__(),
            "`_NDIter.ith()` of C array by order C breaks",
        )

    var a_nditer_f = a.nditer(order="F")
    var anp_nditer_f = np.nditer(anp, order=PythonObject("F"))
    for _ in range(a.size):
        check_values_close(
            a_nditer_f.__next__(),
            anp_nditer_f.__next__(),
            "`_NDIter` or `nditer()` of C array by order F breaks",
        )

    # NDIter of F-order array
    var f_nditer = f.nditer()
    var fnp_nditer = np.nditer(fnp)
    for _ in range(f.size):
        check_values_close(
            f_nditer.__next__(),
            fnp_nditer.__next__(),
            "`_NDIter` or `nditer()` of F array by order C breaks",
        )

    var f_nditer_f = f.nditer(order="F")
    var fnp_nditer_f = np.nditer(fnp, order=PythonObject("F"))
    for _ in range(f.size):
        check_values_close(
            f_nditer_f.__next__(),
            fnp_nditer_f.__next__(),
            "`_NDIter` or `nditer()` of F array by order F breaks",
        )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
