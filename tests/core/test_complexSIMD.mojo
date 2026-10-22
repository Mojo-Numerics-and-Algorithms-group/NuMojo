from std.python import Python
from std.testing import assert_almost_equal, assert_equal, assert_true
from std.testing import TestSuite

from numojo import *


def test_complex_init() raises:
    """Test initialization of ComplexSIMD."""
    var c1 = ComplexSIMD[cf32](1.0, 2.0)
    assert_equal(c1.re, 1.0, "init failed")
    assert_equal(c1.im, 2.0, "init failed")

    var c2 = ComplexSIMD[cf32](c1)
    assert_equal(c2.re, c1.re, "init failed")
    assert_equal(c2.im, c1.im, "init failed")

    var c3 = ComplexSIMD[cf32, 2](1.0)
    assert_equal(c3.re[0], 1.0, "init failed")
    assert_equal(c3.im[0], 1.0, "init failed")


def test_complex_add() raises:
    """Test addition of ComplexSIMD numbers."""
    var c1 = ComplexSIMD[cf32](1.0, 2.0)
    var c2 = ComplexSIMD[cf32](3.0, 4.0)

    var sum = c1 + c2
    assert_equal(sum.re, 4.0, "addition failed")
    assert_equal(sum.im, 6.0, "addition failed")

    var c3 = c1
    c3 += c2
    assert_equal(c3.re, 4.0, "addition failed")
    assert_equal(c3.im, 6.0, "addition failed")


def test_complex_sub() raises:
    """Test subtraction of ComplexSIMD numbers."""
    var c1 = ComplexSIMD[cf32](3.0, 4.0)
    var c2 = ComplexSIMD[cf32](1.0, 2.0)

    var diff = c1 - c2
    assert_equal(diff.re, 2.0, "subtraction failed")
    assert_equal(diff.im, 2.0, "subtraction failed")

    var c3 = c1
    c3 -= c2
    assert_equal(c3.re, 2.0, "subtraction failed")
    assert_equal(c3.im, 2.0, "subtraction failed")


def test_complex_mul() raises:
    """Test multiplication of ComplexSIMD numbers."""
    var c1 = ComplexSIMD[cf32](1.0, 2.0)
    var c2 = ComplexSIMD[cf32](3.0, 4.0)

    # (1 + 2i)(3 + 4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
    var prod = c1 * c2
    assert_equal(prod.re, -5.0, "multiplication failed")
    assert_equal(prod.im, 10.0, "multiplication failed")

    var c3 = c1
    c3 *= c2
    assert_equal(c3.re, -5.0, "multiplication failed")
    assert_equal(c3.im, 10.0, "multiplication failed")


def test_complex_div() raises:
    """Test division of ComplexSIMD numbers."""
    var c1 = ComplexSIMD[cf32](1.0, 2.0)
    var c2 = ComplexSIMD[cf32](3.0, 4.0)

    # (1 + 2i)/(3 + 4i) = (1*3 + 2*4 + (2*3 - 1*4)i)/(3^2 + 4^2)
    # = (11 + 2i)/25
    var quot = c1 / c2
    assert_almost_equal(quot.re, 0.44, " division failed")
    assert_almost_equal(quot.im, 0.08, " division failed")


def test_complex_pow_matches_numpy() raises:
    """Complex and real powers use NumPy's principal-branch semantics."""
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var base = ComplexSIMD[cf64](1.5, -2.0)
    var exponent = ComplexSIMD[cf64](0.75, 0.5)

    var complex_result = base**exponent
    var complex_expected = np.power(
        builtins.complex(1.5, -2.0), builtins.complex(0.75, 0.5)
    )
    assert_true(np.isclose(complex_result.re, complex_expected.real))
    assert_true(np.isclose(complex_result.im, complex_expected.imag))

    var real_result = base ** Scalar[DType.float64](2.5)
    var real_expected = np.power(builtins.complex(1.5, -2.0), 2.5)
    assert_true(np.isclose(real_result.re, real_expected.real))
    assert_true(np.isclose(real_result.im, real_expected.imag))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
