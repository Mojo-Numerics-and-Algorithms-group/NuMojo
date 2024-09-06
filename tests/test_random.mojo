import numojo as nm
from numojo.core import random
from time import now
from python import Python, PythonObject
from utils_for_test import check, check_is_close
from testing.testing import assert_true, assert_almost_equal

def test_rand():
    """Test random array generation with specified shape."""
    var arr = random.rand[nm.f64](3, 5, 2)
    assert_true(arr.ndshape[0] == 3, "Shape of random array")
    assert_true(arr.ndshape[1] == 5, "Shape of random array")
    assert_true(arr.ndshape[2] == 2, "Shape of random array")

def test_randminmax():
    """Test random array generation with min and max values."""
    var arr_variadic = random.rand[nm.f64](10, 10, 10, min=1, max=2)
    var arr_list = random.rand[nm.f64](List[Int](10, 10, 10), min=3, max=4)
    var arr_variadic_mean = nm.cummean(arr_variadic)
    var arr_list_mean = nm.cummean(arr_list)
    assert_almost_equal(arr_variadic_mean, 1.5, msg="Mean of random array within min and max", atol=0.1)
    assert_almost_equal(arr_list_mean, 3.5, msg="Mean of random array within min and max", atol=0.1)

def test_randn():
    """Test random array generation with normal distribution."""
    var arr_variadic_01 = random.randn[nm.f64](20, 20, 20, mean=0.0, variance=1.0)
    var arr_variadic_31 = random.randn[nm.f64](20, 20, 20, mean=3.0, variance=1.0)
    var arr_variadic_12 = random.randn[nm.f64](20, 20, 20, mean=1.0, variance=3.0)

    var arr_variadic_mean01 = nm.cummean(arr_variadic_01)
    var arr_variadic_mean31 = nm.cummean(arr_variadic_31)
    var arr_variadic_mean12 = nm.cummean(arr_variadic_12)
    var arr_variadic_var01 = nm.cumvariance(arr_variadic_01)
    var arr_variadic_var31 = nm.cumvariance(arr_variadic_31)
    var arr_variadic_var12 = nm.cumvariance(arr_variadic_12)

    assert_almost_equal(arr_variadic_mean01, 0, msg="Mean of random array with mean 0 and variance 1", atol=0.1)
    assert_almost_equal(arr_variadic_mean31, 3, msg="Mean of random array with mean 3 and variance 1", atol=0.1)
    assert_almost_equal(arr_variadic_mean12, 1, msg="Mean of random array with mean 1 and variance 2", atol=0.1)

    assert_almost_equal(arr_variadic_var01, 1, msg="Variance of random array with mean 0 and variance 1", atol=0.1)
    assert_almost_equal(arr_variadic_var31, 1, msg="Variance of random array with mean 3 and variance 1", atol=0.1)
    assert_almost_equal(arr_variadic_var12, 3, msg="Variance of random array with mean 1 and variance 2", atol=0.1)

def test_randn_list():
    """Test random array generation with normal distribution."""
    var arr_list_01 = random.randn[nm.f64](List[Int](20, 20, 20), mean=0.0, variance=1.0)
    var arr_list_31 = random.randn[nm.f64](List[Int](20, 20, 20), mean=3.0, variance=1.0)
    var arr_list_12 = random.randn[nm.f64](List[Int](20, 20, 20), mean=1.0, variance=3.0)

    var arr_list_mean01 = nm.cummean(arr_list_01)
    var arr_list_mean31 = nm.cummean(arr_list_31)
    var arr_list_mean12 = nm.cummean(arr_list_12)
    var arr_list_var01 = nm.cumvariance(arr_list_01)
    var arr_list_var31 = nm.cumvariance(arr_list_31)
    var arr_list_var12 = nm.cumvariance(arr_list_12)

    assert_almost_equal(arr_list_mean01, 0, msg="Mean of random array with mean 0 and variance 1", atol=0.1)
    assert_almost_equal(arr_list_mean31, 3, msg="Mean of random array with mean 3 and variance 1", atol=0.1)
    assert_almost_equal(arr_list_mean12, 1, msg="Mean of random array with mean 1 and variance 2", atol=0.1)

    assert_almost_equal(arr_list_var01, 1, msg="Variance of random array with mean 0 and variance 1", atol=0.1)
    assert_almost_equal(arr_list_var31, 1, msg="Variance of random array with mean 3 and variance 1", atol=0.1)
    assert_almost_equal(arr_list_var12, 3, msg="Variance of random array with mean 1 and variance 2", atol=0.1)

def test_rand_exponential():
    """Test random array generation with exponential distribution."""
    var arr_variadic = random.rand_exponential[nm.f64](20, 20, 20, rate=2.0)
    var arr_list = random.rand_exponential[nm.f64](List[Int](20, 20, 20), rate=0.5)

    var arr_variadic_mean = nm.cummean(arr_variadic)
    var arr_list_mean = nm.cummean(arr_list)

    # For exponential distribution, mean = 1 / rate
    assert_almost_equal(arr_variadic_mean, 1/2, msg="Mean of exponential distribution with rate 2.0", atol=0.1)
    assert_almost_equal(arr_list_mean, 1/0.5, msg="Mean of exponential distribution with rate 0.5", atol=0.2)

    # For exponential distribution, variance = 1 / (rate^2)
    var arr_variadic_var = nm.cumvariance(arr_variadic)
    var arr_list_var = nm.cumvariance(arr_list)

    assert_almost_equal(arr_variadic_var, 1/(2), msg="Variance of exponential distribution with rate 2.0", atol=0.1)
    assert_almost_equal(arr_list_var, 1/(0.5), msg="Variance of exponential distribution with rate 0.5", atol=0.5)

    # Test that all values are non-negative
    for i in range(arr_variadic.num_elements()):
        assert_true(arr_variadic.data[i] >= 0, "Exponential distribution should only produce non-negative values")

    for i in range(arr_list.num_elements()):
        assert_true(arr_list.data[i] >= 0, "Exponential distribution should only produce non-negative values")
