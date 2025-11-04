from math import sqrt
import numojo as nm
from numojo.prelude import *
from python import Python, PythonObject
from utils_for_test import check, check_is_close
from testing.testing import assert_true, assert_almost_equal
from testing import TestSuite


def test_rand():
    """Test random array generation with specified shape."""
    var arr = nm.random.rand[nm.f64](3, 5, 2)
    assert_true(arr.shape[0] == 3, "Shape of random array")
    assert_true(arr.shape[1] == 5, "Shape of random array")
    assert_true(arr.shape[2] == 2, "Shape of random array")


def test_randminmax():
    """Test random array generation with min and max values."""
    var arr_variadic = nm.random.rand[nm.f64](10, 10, 10, min=1, max=2)
    var arr_list = nm.random.rand[nm.f64](List[Int](10, 10, 10), min=3, max=4)
    var arr_variadic_mean = nm.mean(arr_variadic)
    var arr_list_mean = nm.mean(arr_list)
    assert_almost_equal(
        arr_variadic_mean,
        1.5,
        msg="Mean of random array within min and max",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_mean,
        3.5,
        msg="Mean of random array within min and max",
        atol=0.1,
    )


def test_randint():
    """Test random int array generation with min and max values."""
    var arr_low_high = nm.random.randint(Shape(10, 10, 10), 0, 10)
    var arr_high = nm.random.randint(Shape(10, 10, 10), 6)
    var arr_low_high_mean = nm.mean(arr_low_high)
    var arr_high_mean = nm.mean(arr_high)
    assert_almost_equal(
        arr_low_high_mean,
        4.5,
        msg="Mean of `nm.random.randint(Shape(10, 10), 0, 10)` breaks",
        atol=0.1,
    )
    assert_almost_equal(
        arr_high_mean,
        2.5,
        msg="Mean of `nm.random.randint(Shape(10, 10), 6)` breaks",
        atol=0.1,
    )


def test_randn():
    """Test random array generation with normal distribution."""
    var arr_variadic_01 = nm.random.randn[nm.f64](20, 20, 20)
    var arr_variadic_31 = nm.random.randn[nm.f64](
        Shape(20, 20, 20), mean=3, variance=1
    )
    var arr_variadic_12 = nm.random.randn[nm.f64](Shape(20, 20, 20), 1, 2)

    var arr_variadic_mean01 = nm.mean(arr_variadic_01)
    var arr_variadic_mean31 = nm.mean(arr_variadic_31)
    var arr_variadic_mean12 = nm.mean(arr_variadic_12)
    var arr_variadic_var01 = nm.variance(arr_variadic_01)
    var arr_variadic_var31 = nm.variance(arr_variadic_31)
    var arr_variadic_var12 = nm.variance(arr_variadic_12)

    assert_almost_equal(
        arr_variadic_mean01,
        0,
        msg="Mean of random array with mean 0 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_variadic_mean31,
        3,
        msg="Mean of random array with mean 3 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_variadic_mean12,
        1,
        msg="Mean of random array with mean 1 and variance 2",
        atol=0.1,
    )

    assert_almost_equal(
        arr_variadic_var01,
        1,
        msg="Variance of random array with mean 0 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_variadic_var31,
        1,
        msg="Variance of random array with mean 3 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_variadic_var12,
        2,
        msg="Variance of random array with mean 1 and variance 2",
        atol=0.1,
    )


def test_randn_list():
    """Test random array generation with normal distribution."""
    var arr_list_01 = nm.random.randn[nm.f64](Shape(20, 20, 20))
    var arr_list_31 = nm.random.randn[nm.f64](Shape(20, 20, 20)) + 3
    var arr_list_12 = nm.random.randn[nm.f64](Shape(20, 20, 20)) * sqrt(2.0) + 1

    var arr_list_mean01 = nm.mean(arr_list_01)
    var arr_list_mean31 = nm.mean(arr_list_31)
    var arr_list_mean12 = nm.mean(arr_list_12)
    var arr_list_var01 = nm.variance(arr_list_01)
    var arr_list_var31 = nm.variance(arr_list_31)
    var arr_list_var12 = nm.variance(arr_list_12)

    assert_almost_equal(
        arr_list_mean01,
        0,
        msg="Mean of random array with mean 0 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_mean31,
        3,
        msg="Mean of random array with mean 3 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_mean12,
        1,
        msg="Mean of random array with mean 1 and variance 2",
        atol=0.1,
    )

    assert_almost_equal(
        arr_list_var01,
        1,
        msg="Variance of random array with mean 0 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_var31,
        1,
        msg="Variance of random array with mean 3 and variance 1",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_var12,
        2,
        msg="Variance of random array with mean 1 and variance 2",
        atol=0.1,
    )


def test_rand_exponential():
    """Test random array generation with exponential distribution."""
    var arr_variadic = nm.random.exponential[nm.f64](
        Shape(20, 20, 20), scale=2.0
    )
    var arr_list = nm.random.exponential[nm.f64](
        List[Int](20, 20, 20), scale=0.5
    )

    var arr_variadic_mean = nm.mean(arr_variadic)
    var arr_list_mean = nm.mean(arr_list)

    # For exponential distribution, mean = 1 / rate
    assert_almost_equal(
        arr_variadic_mean,
        1 / 2,
        msg="Mean of exponential distribution with rate 2.0",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_mean,
        1 / 0.5,
        msg="Mean of exponential distribution with rate 0.5",
        atol=0.2,
    )

    # For exponential distribution, variance = 1 / (rate^2)
    var arr_variadic_var = nm.variance(arr_variadic)
    var arr_list_var = nm.variance(arr_list)

    assert_almost_equal(
        arr_variadic_var,
        1 / 2**2,
        msg="Variance of exponential distribution with rate 2.0",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_var,
        1 / 0.5**2,
        msg="Variance of exponential distribution with rate 0.5",
        atol=0.5,
    )

    # Test that all values are non-negative
    for i in range(arr_variadic.num_elements()):
        assert_true(
            arr_variadic._buf.ptr[i] >= 0,
            "Exponential distribution should only produce non-negative values",
        )

    for i in range(arr_list.num_elements()):
        assert_true(
            arr_list._buf.ptr[i] >= 0,
            "Exponential distribution should only produce non-negative values",
        )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
