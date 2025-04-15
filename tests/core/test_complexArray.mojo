from testing import assert_equal, assert_almost_equal
from numojo import *

# TODO: Added getter and setter tests


fn test_complex_array_init() raises:
    """Test initialization of ComplexArray."""
    var c1 = ComplexNDArray[f32](Shape(2, 2))
    c1.itemset(0, ComplexSIMD[f32](1.0, 2.0))
    c1.itemset(1, ComplexSIMD[f32](3.0, 4.0))
    c1.itemset(2, ComplexSIMD[f32](5.0, 6.0))
    c1.itemset(3, ComplexSIMD[f32](7.0, 8.0))
    assert_almost_equal(c1.item(0).re, 1.0, "init failed")
    assert_almost_equal(c1.item(0).im, 2.0, "init failed")


fn test_complex_array_add() raises:
    """Test addition of ComplexArray numbers."""
    var c1 = ComplexNDArray[f32](Shape(2, 2))
    var c2 = ComplexNDArray[f32](Shape(2, 2))
    c1.itemset(0, ComplexSIMD[f32](1.0, 2.0))
    c1.itemset(1, ComplexSIMD[f32](3.0, 4.0))
    c1.itemset(2, ComplexSIMD[f32](5.0, 6.0))
    c1.itemset(3, ComplexSIMD[f32](7.0, 8.0))
    c2.itemset(0, ComplexSIMD[f32](1.0, 2.0))
    c2.itemset(1, ComplexSIMD[f32](3.0, 4.0))
    c2.itemset(2, ComplexSIMD[f32](5.0, 6.0))
    c2.itemset(3, ComplexSIMD[f32](7.0, 8.0))

    var sum = c1 + c2

    assert_almost_equal(sum.item(0).re, 2.0, "add failed")
    assert_almost_equal(sum.item(0).im, 4.0, "add failed")
    assert_almost_equal(sum.item(1).re, 6.0, "add failed")
    assert_almost_equal(sum.item(1).im, 8.0, "add failed")
    assert_almost_equal(sum.item(2).re, 10.0, "add failed")
    assert_almost_equal(sum.item(2).im, 12.0, "add failed")
    assert_almost_equal(sum.item(3).re, 14.0, "add failed")
    assert_almost_equal(sum.item(3).im, 16.0, "add failed")


fn test_complex_array_sub() raises:
    """Test subtraction of ComplexArray numbers."""
    var c1 = ComplexNDArray[f32](Shape(2, 2))
    var c2 = ComplexNDArray[f32](Shape(2, 2))
    c1.itemset(0, ComplexSIMD[f32](1.0, 2.0))
    c1.itemset(1, ComplexSIMD[f32](3.0, 4.0))
    c1.itemset(2, ComplexSIMD[f32](5.0, 6.0))
    c1.itemset(3, ComplexSIMD[f32](7.0, 8.0))

    c2.itemset(0, ComplexSIMD[f32](3.0, 4.0))
    c2.itemset(1, ComplexSIMD[f32](5.0, 6.0))
    c2.itemset(2, ComplexSIMD[f32](7.0, 8.0))
    c2.itemset(3, ComplexSIMD[f32](9.0, 10.0))

    var diff = c1 - c2

    assert_almost_equal(diff.item(0).re, -2.0, "sub failed")
    assert_almost_equal(diff.item(0).im, -2.0, "sub failed")
    assert_almost_equal(diff.item(1).re, -2.0, "sub failed")
    assert_almost_equal(diff.item(1).im, -2.0, "sub failed")
    assert_almost_equal(diff.item(2).re, -2.0, "sub failed")
    assert_almost_equal(diff.item(2).im, -2.0, "sub failed")
    assert_almost_equal(diff.item(3).re, -2.0, "sub failed")
    assert_almost_equal(diff.item(3).im, -2.0, "sub failed")


fn test_complex_array_mul() raises:
    """Test multiplication of ComplexArray numbers."""
    var c1 = ComplexNDArray[f32](Shape(2, 2))
    var c2 = ComplexNDArray[f32](Shape(2, 2))
    c1.itemset(0, ComplexSIMD[f32](1.0, 2.0))
    c1.itemset(1, ComplexSIMD[f32](3.0, 4.0))
    c1.itemset(2, ComplexSIMD[f32](5.0, 6.0))
    c1.itemset(3, ComplexSIMD[f32](7.0, 8.0))

    c2.itemset(0, ComplexSIMD[f32](1.0, 2.0))
    c2.itemset(1, ComplexSIMD[f32](3.0, 4.0))
    c2.itemset(2, ComplexSIMD[f32](5.0, 6.0))
    c2.itemset(3, ComplexSIMD[f32](7.0, 8.0))

    var prod = c1 * c2

    assert_almost_equal(prod.item(0).re, -3.0, "mul failed")
    assert_almost_equal(prod.item(0).im, 4.0, "mul failed")


fn test_complex_array_div() raises:
    """Test division of ComplexArray numbers."""
    var c1 = ComplexNDArray[f32](Shape(2, 2))
    var c2 = ComplexNDArray[f32](Shape(2, 2))
    c1.itemset(0, ComplexSIMD[f32](1.0, 2.0))
    c1.itemset(1, ComplexSIMD[f32](3.0, 4.0))
    c1.itemset(2, ComplexSIMD[f32](5.0, 6.0))
    c1.itemset(3, ComplexSIMD[f32](7.0, 8.0))

    c2.itemset(0, ComplexSIMD[f32](3.0, 4.0))
    c2.itemset(1, ComplexSIMD[f32](5.0, 6.0))
    c2.itemset(2, ComplexSIMD[f32](7.0, 8.0))
    c2.itemset(3, ComplexSIMD[f32](9.0, 10.0))

    var quot = c1 / c2

    assert_almost_equal(quot.item(0).re, 0.44, "div failed")
    assert_almost_equal(quot.item(0).im, 0.08, "div failed")
