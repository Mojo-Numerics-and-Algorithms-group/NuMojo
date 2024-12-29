from testing import assert_equal, assert_almost_equal
from numojo import *


fn test_complex_init() raises:
    """Test initialization of ComplexSIMD."""
    var c1 = ComplexSIMD[cf32, f32](1.0, 2.0)
    assert_equal(c1.re, 1.0, "init failed")
    assert_equal(c1.im, 2.0, "init failed")

    var c2 = ComplexSIMD[cf32, f32](c1)
    assert_equal(c2.re, c1.re)
    assert_equal(c2.im, c1.im)


fn test_complex_add() raises:
    """Test addition of ComplexSIMD numbers."""
    var c1 = ComplexSIMD[cf32, f32](1.0, 2.0)
    var c2 = ComplexSIMD[cf32, f32](3.0, 4.0)

    var sum = c1 + c2
    assert_equal(sum.re, 4.0, "addition failed")
    assert_equal(sum.im, 6.0, "addition failed")

    var c3 = c1
    c3 += c2
    assert_equal(c3.re, 4.0, "addition failed")
    assert_equal(c3.im, 6.0, "addition failed")


fn test_complex_sub() raises:
    """Test subtraction of ComplexSIMD numbers."""
    var c1 = ComplexSIMD[cf32, f32](3.0, 4.0)
    var c2 = ComplexSIMD[cf32, f32](1.0, 2.0)

    var diff = c1 - c2
    assert_equal(diff.re, 2.0, "subtraction failed")
    assert_equal(diff.im, 2.0, "subtraction failed")

    var c3 = c1
    c3 -= c2
    assert_equal(c3.re, 2.0, "subtraction failed")
    assert_equal(c3.im, 2.0, "subtraction failed")


fn test_complex_mul() raises:
    """Test multiplication of ComplexSIMD numbers."""
    var c1 = ComplexSIMD[cf32, f32](1.0, 2.0)
    var c2 = ComplexSIMD[cf32, f32](3.0, 4.0)

    # (1 + 2i)(3 + 4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
    var prod = c1 * c2
    assert_equal(prod.re, -5.0, "multiplication failed")
    assert_equal(prod.im, 10.0, "multiplication failed")

    var c3 = c1
    c3 *= c2
    assert_equal(c3.re, -5.0, "multiplication failed")
    assert_equal(c3.im, 10.0, "multiplication failed")


fn test_complex_div() raises:
    """Test division of ComplexSIMD numbers."""
    var c1 = ComplexSIMD[cf32, f32](1.0, 2.0)
    var c2 = ComplexSIMD[cf32, f32](3.0, 4.0)

    # (1 + 2i)/(3 + 4i) = (1*3 + 2*4 + (2*3 - 1*4)i)/(3^2 + 4^2)
    # = (11 + 2i)/25
    var quot = c1 / c2
    assert_almost_equal(quot.re, 0.44, " division failed")
    assert_almost_equal(quot.im, 0.08, " division failed")
