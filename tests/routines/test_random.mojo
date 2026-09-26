from std.math import sqrt
from std.python import Python, PythonObject
from utils_for_test import check, check_is_close
from std.testing.testing import assert_true, assert_almost_equal
from std.testing import TestSuite

import numojo as nm
from numojo.prelude import *


def test_rand() raises:
    """Test random array generation with specified shape."""
    var arr = nm.random.rand[nm.f64](3, 5, 2)
    assert_true(arr.shape[0] == 3, "Shape of random array")
    assert_true(arr.shape[1] == 5, "Shape of random array")
    assert_true(arr.shape[2] == 2, "Shape of random array")


def test_randminmax() raises:
    """Test random array generation with min and max values."""
    var arr_variadic = nm.random.rand[nm.f64](10, 10, 10, min=1, max=2)
    var arr_list = nm.random.rand[nm.f64]([Int(10), 10, 10], min=3, max=4)
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


def test_randint() raises:
    """Test random int array generation with min and max values."""
    var arr_low_high = nm.random.randint(Shape(30, 30, 30), 0, 10)
    var arr_high = nm.random.randint(Shape(30, 30, 30), 6)
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


def test_randn() raises:
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


def test_randn_list() raises:
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


def test_rand_exponential() raises:
    """Test random array generation with exponential distribution."""
    var arr_variadic = nm.random.exponential[nm.f64](
        Shape(20, 20, 20), scale=2.0
    )
    var arr_list = nm.random.exponential[nm.f64]([20, 20, 20], scale=0.5)

    var arr_variadic_mean = nm.mean(arr_variadic)
    var arr_list_mean = nm.mean(arr_list)

    # For exponential distribution, mean = 1 / rate
    assert_almost_equal(
        arr_variadic_mean,
        0.5,
        msg="Mean of exponential distribution with rate 2.0",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_mean,
        2.0,
        msg="Mean of exponential distribution with rate 0.5",
        atol=0.2,
    )

    # For exponential distribution, variance = 1 / (rate^2)
    var arr_variadic_var = nm.variance(arr_variadic)
    var arr_list_var = nm.variance(arr_list)

    assert_almost_equal(
        arr_variadic_var,
        1.0 / 2.0**2,
        msg="Variance of exponential distribution with rate 2.0",
        atol=0.1,
    )
    assert_almost_equal(
        arr_list_var,
        1.0 / 0.5**2,
        msg="Variance of exponential distribution with rate 0.5",
        atol=0.5,
    )

    # Test that all values are non-negative
    for i in range(arr_variadic.size):
        assert_true(
            arr_variadic.unsafe_get(i) >= 0,
            "Exponential distribution should only produce non-negative values",
        )

    for i in range(arr_list.size):
        assert_true(
            arr_list.unsafe_get(i) >= 0,
            "Exponential distribution should only produce non-negative values",
        )


def test_randbool() raises:
    """Test random boolean array generation."""
    # Shape is preserved
    var arr = nm.random.randbool(3, 5, 2)
    assert_true(arr.shape[0] == 3, "Shape[0] of randbool array")
    assert_true(arr.shape[1] == 5, "Shape[1] of randbool array")
    assert_true(arr.shape[2] == 2, "Shape[2] of randbool array")

    # All values are True when p=1.0
    var all_true = nm.random.randbool(10, 10, p=1.0)
    for i in range(all_true.size):
        assert_true(
            all_true.unsafe_get(i) == True,
            "All values should be True with p=1.0",
        )

    # All values are False when p=0.0
    var all_false = nm.random.randbool(10, 10, p=0.0)
    for i in range(all_false.size):
        assert_true(
            all_false.unsafe_get(i) == False,
            "All values should be False with p=0.0",
        )

    # Fraction of True values with p=0.5 should be near 0.5
    var big = nm.random.randbool(Shape(50, 50), p=0.5)
    var count_true: Int = 0
    for i in range(big.size):
        if big.unsafe_get(i):
            count_true += 1
    var frac = Float64(count_true) / Float64(big.size)
    assert_true(
        frac > 0.35 and frac < 0.65,
        "Fraction of True values should be near 0.5 for p=0.5",
    )


def test_seed() raises:
    """Test that seeding makes results reproducible."""
    nm.random.seed(42)
    var arr1 = nm.random.rand[nm.f64](Shape(20))
    nm.random.seed(42)
    var arr2 = nm.random.rand[nm.f64](Shape(20))
    for i in range(arr1.size):
        assert_true(
            arr1.item(i) == arr2.item(i),
            "Same seed should produce identical sequences",
        )

    nm.random.seed(1)
    var arr3 = nm.random.rand[nm.f64](Shape(20))
    nm.random.seed(2)
    var arr4 = nm.random.rand[nm.f64](Shape(20))
    var all_equal = True
    for i in range(arr3.size):
        if arr3.item(i) != arr4.item(i):
            all_equal = False
    assert_true(
        not all_equal, "Different seeds should (almost certainly) differ"
    )

    # Restore a time-based seed so later tests in this file aren't
    # accidentally correlated with the fixed seeds above.
    nm.random.seed()


def test_uniform() raises:
    """Test the `uniform` alias for `rand`."""
    var arr = nm.random.uniform[nm.f64](Shape(20, 20), low=-1.0, high=1.0)
    assert_almost_equal(
        nm.mean(arr), 0.0, msg="Mean of uniform(-1, 1)", atol=0.1
    )
    for i in range(arr.size):
        var v = arr.unsafe_get(i)
        assert_true(v >= -1.0 and v < 1.0, "uniform values within [low, high)")


def test_normal() raises:
    """Test the `normal` alias for `randn`."""
    var arr = nm.random.normal[nm.f64](Shape(30, 30), loc=5.0, scale=2.0)
    assert_almost_equal(
        nm.mean(arr), 5.0, msg="Mean of normal(loc=5, scale=2)", atol=0.2
    )
    assert_almost_equal(
        nm.variance(arr),
        4.0,
        msg="Variance of normal(loc=5, scale=2)",
        atol=0.5,
    )


def test_binomial() raises:
    """Test the binomial distribution."""
    var arr = nm.random.binomial(Shape(3000), n=20, p=0.3)
    assert_almost_equal(
        nm.mean(arr).cast[nm.f64](),
        6.0,
        msg="Mean of Binomial(20, 0.3)",
        atol=0.3,
    )
    for i in range(arr.size):
        var v = arr.unsafe_get(i)
        assert_true(v >= 0 and v <= 20, "binomial values within [0, n]")


def test_poisson() raises:
    """Test the Poisson distribution."""
    var arr = nm.random.poisson(Shape(3000), lam=4.0)
    assert_almost_equal(
        nm.mean(arr).cast[nm.f64](),
        4.0,
        msg="Mean of Poisson(4.0)",
        atol=0.3,
    )
    for i in range(arr.size):
        assert_true(arr.unsafe_get(i) >= 0, "poisson values are non-negative")


def test_gamma() raises:
    """Test the gamma distribution."""
    var arr = nm.random.gamma[nm.f64](Shape(3000), k=2.0, scale=2.0)
    # Gamma(k, scale) has mean k * scale and variance k * scale^2.
    assert_almost_equal(
        nm.mean(arr), 4.0, msg="Mean of Gamma(k=2, scale=2)", atol=0.3
    )
    assert_almost_equal(
        nm.variance(arr), 8.0, msg="Variance of Gamma(k=2, scale=2)", atol=1.5
    )
    for i in range(arr.size):
        assert_true(arr.unsafe_get(i) > 0, "gamma values are positive")


def test_beta() raises:
    """Test the beta distribution."""
    var arr = nm.random.beta[nm.f64](Shape(3000), a=2.0, b=5.0)
    # Beta(a, b) has mean a / (a + b).
    assert_almost_equal(
        nm.mean(arr), 2.0 / 7.0, msg="Mean of Beta(2, 5)", atol=0.05
    )
    for i in range(arr.size):
        var v = arr.unsafe_get(i)
        assert_true(v >= 0.0 and v <= 1.0, "beta values within [0, 1]")


def test_chisquare() raises:
    """Test the chi-square distribution."""
    var arr = nm.random.chisquare[nm.f64](Shape(3000), df=4.0)
    # ChiSquare(df) has mean df and variance 2 * df.
    assert_almost_equal(nm.mean(arr), 4.0, msg="Mean of ChiSquare(4)", atol=0.3)
    assert_almost_equal(
        nm.variance(arr), 8.0, msg="Variance of ChiSquare(4)", atol=1.5
    )
    for i in range(arr.size):
        assert_true(arr.unsafe_get(i) > 0, "chisquare values are positive")


def test_shuffle() raises:
    """Test in-place shuffling of an array."""
    var arr = nm.arange[nm.f64](20)
    nm.random.shuffle(arr)
    assert_true(arr.size == 20, "shuffle preserves size")

    # The shuffled array should be a permutation: same elements, sum unchanged.
    var expected_sum: Float64 = 0.0
    for i in range(20):
        expected_sum += Float64(i)
    var actual_sum: Float64 = 0.0
    for i in range(arr.size):
        actual_sum += arr.unsafe_get(i)
    assert_almost_equal(
        actual_sum, expected_sum, msg="shuffle preserves the set of elements"
    )


def test_permutation() raises:
    """Test `permutation` for both an integer range and an array copy."""
    var perm = nm.random.permutation(20)
    assert_true(perm.size == 20, "permutation(n) has size n")
    var seen = List[Bool](capacity=20)
    for _ in range(20):
        seen.append(False)
    for i in range(perm.size):
        var v = Int(perm.unsafe_get(i))
        assert_true(v >= 0 and v < 20, "permutation(n) values within [0, n)")
        seen[v] = True
    for i in range(20):
        assert_true(seen[i], "permutation(n) contains every value exactly once")

    var original = nm.arange[nm.f64](10)
    var permuted = nm.random.permutation(original)
    for i in range(original.size):
        assert_true(
            original.unsafe_get(i) == Float64(i),
            "permutation(array) does not modify the input",
        )
    var permuted_sum: Float64 = 0.0
    for i in range(permuted.size):
        permuted_sum += permuted.unsafe_get(i)
    assert_almost_equal(
        permuted_sum, 45.0, msg="permutation(array) preserves the elements"
    )


def test_choice() raises:
    """Test sampling from an array, with and without replacement."""
    var pool = nm.arange[nm.f64](10)

    var with_replacement = nm.random.choice(Shape(100), pool, replace=True)
    assert_true(
        with_replacement.size == 100, "choice output has requested size"
    )
    for i in range(with_replacement.size):
        var v = with_replacement.unsafe_get(i)
        assert_true(v >= 0.0 and v < 10.0, "choice values come from the pool")

    var without_replacement = nm.random.choice(Shape(10), pool, replace=False)
    var seen = List[Bool](capacity=10)
    for _ in range(10):
        seen.append(False)
    for i in range(without_replacement.size):
        var v = Int(without_replacement.unsafe_get(i))
        assert_true(
            not seen[v], "choice without replacement does not repeat values"
        )
        seen[v] = True


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
