from std.testing import assert_equal, assert_almost_equal
from numojo import *
from std.testing import TestSuite

# TODO: Added getter and setter tests


def test_complex_array_init() raises:
    """Test initialization of ComplexArray."""
    var c1 = ComplexNDArray[cf32](Shape(2, 2))
    c1.itemset(0, ComplexSIMD[cf32](1.0, 2.0))
    c1.itemset(1, ComplexSIMD[cf32](3.0, 4.0))
    c1.itemset(2, ComplexSIMD[cf32](5.0, 6.0))
    c1.itemset(3, ComplexSIMD[cf32](7.0, 8.0))
    assert_almost_equal(c1.item(0).re, 1.0, "init failed")
    assert_almost_equal(c1.item(0).im, 2.0, "init failed")


def test_complex_array_itemset() raises:
    """Test itemset with List[Int] coordinates."""
    var c1 = ComplexNDArray[cf32](Shape(3, 3))
    c1.itemset([0, 0], ComplexSIMD[cf32](1.0, 2.0))
    c1.itemset([1, 1], ComplexSIMD[cf32](3.0, 4.0))
    c1.itemset([2, 2], ComplexSIMD[cf32](5.0, 6.0))
    assert_almost_equal(c1.item(0, 0).re, 1.0, "itemset List failed")
    assert_almost_equal(c1.item(0, 0).im, 2.0, "itemset List failed")
    assert_almost_equal(c1.item(1, 1).re, 3.0, "itemset List failed")
    assert_almost_equal(c1.item(1, 1).im, 4.0, "itemset List failed")
    assert_almost_equal(c1.item(2, 2).re, 5.0, "itemset List failed")
    assert_almost_equal(c1.item(2, 2).im, 6.0, "itemset List failed")


def test_complex_array_view() raises:
    """Test view() method."""
    var c1 = ComplexNDArray[cf32](Shape(2, 2))
    c1.itemset(0, ComplexSIMD[cf32](1.0, 2.0))
    var v = c1.view()
    assert_almost_equal(v.item(0).re, 1.0, "view failed")
    assert_almost_equal(v.item(0).im, 2.0, "view failed")

def test_complex_array_view() raises:
    """Test view() method."""
    var c1 = ComplexNDArray[cf32](Shape(2, 2))
    c1.itemset(0, ComplexSIMD[cf32](1.0, 2.0))
    var v = c1.view()
    assert_almost_equal(v.item(0).re, 1.0, "view failed")
    assert_almost_equal(v.item(0).im, 2.0, "view failed")


def test_complex_array_itemset_negative_index() raises:
    """Test itemset with negative indices."""
    var c1 = ComplexNDArray[cf32](Shape(3, 3))
    c1.itemset(-1, ComplexSIMD[cf32](7.0, 8.0))
    assert_almost_equal(c1.item(2, 2).re, 7.0, "negative index failed")
    assert_almost_equal(c1.item(2, 2).im, 8.0, "negative index failed")
    c1.itemset([-1, -1], ComplexSIMD[cf32](9.0, 10.0))
    assert_almost_equal(c1.item(2, 2).re, 9.0, "negative index List failed")
    assert_almost_equal(c1.item(2, 2).im, 10.0, "negative index List failed")


def test_complex_array_add() raises:
    """Test addition of ComplexArray numbers."""
    var c1 = ComplexNDArray[cf32](Shape(2, 2))
    var c2 = ComplexNDArray[cf32](Shape(2, 2))
    c1.itemset(0, ComplexSIMD[cf32](1.0, 2.0))
    c1.itemset(1, ComplexSIMD[cf32](3.0, 4.0))
    c1.itemset(2, ComplexSIMD[cf32](5.0, 6.0))
    c1.itemset(3, ComplexSIMD[cf32](7.0, 8.0))
    c2.itemset(0, ComplexSIMD[cf32](1.0, 2.0))
    c2.itemset(1, ComplexSIMD[cf32](3.0, 4.0))
    c2.itemset(2, ComplexSIMD[cf32](5.0, 6.0))
    c2.itemset(3, ComplexSIMD[cf32](7.0, 8.0))

    var sum = c1 + c2

    assert_almost_equal(sum.item(0).re, 2.0, "add failed")
    assert_almost_equal(sum.item(0).im, 4.0, "add failed")
    assert_almost_equal(sum.item(1).re, 6.0, "add failed")
    assert_almost_equal(sum.item(1).im, 8.0, "add failed")
    assert_almost_equal(sum.item(2).re, 10.0, "add failed")
    assert_almost_equal(sum.item(2).im, 12.0, "add failed")
    assert_almost_equal(sum.item(3).re, 14.0, "add failed")
    assert_almost_equal(sum.item(3).im, 16.0, "add failed")


def test_complex_array_sub() raises:
    """Test subtraction of ComplexArray numbers."""
    var c1 = ComplexNDArray[cf32](Shape(2, 2))
    var c2 = ComplexNDArray[cf32](Shape(2, 2))
    c1.itemset(0, ComplexSIMD[cf32](1.0, 2.0))
    c1.itemset(1, ComplexSIMD[cf32](3.0, 4.0))
    c1.itemset(2, ComplexSIMD[cf32](5.0, 6.0))
    c1.itemset(3, ComplexSIMD[cf32](7.0, 8.0))

    c2.itemset(0, ComplexSIMD[cf32](3.0, 4.0))
    c2.itemset(1, ComplexSIMD[cf32](5.0, 6.0))
    c2.itemset(2, ComplexSIMD[cf32](7.0, 8.0))
    c2.itemset(3, ComplexSIMD[cf32](9.0, 10.0))

    var diff = c1 - c2

    assert_almost_equal(diff.item(0).re, -2.0, "sub failed")
    assert_almost_equal(diff.item(0).im, -2.0, "sub failed")
    assert_almost_equal(diff.item(1).re, -2.0, "sub failed")
    assert_almost_equal(diff.item(1).im, -2.0, "sub failed")
    assert_almost_equal(diff.item(2).re, -2.0, "sub failed")
    assert_almost_equal(diff.item(2).im, -2.0, "sub failed")
    assert_almost_equal(diff.item(3).re, -2.0, "sub failed")
    assert_almost_equal(diff.item(3).im, -2.0, "sub failed")


def test_complex_array_mul() raises:
    """Test multiplication of ComplexArray numbers."""
    var c1 = ComplexNDArray[cf32](Shape(2, 2))
    var c2 = ComplexNDArray[cf32](Shape(2, 2))
    c1.itemset(0, ComplexSIMD[cf32](1.0, 2.0))
    c1.itemset(1, ComplexSIMD[cf32](3.0, 4.0))
    c1.itemset(2, ComplexSIMD[cf32](5.0, 6.0))
    c1.itemset(3, ComplexSIMD[cf32](7.0, 8.0))

    c2.itemset(0, ComplexSIMD[cf32](1.0, 2.0))
    c2.itemset(1, ComplexSIMD[cf32](3.0, 4.0))
    c2.itemset(2, ComplexSIMD[cf32](5.0, 6.0))
    c2.itemset(3, ComplexSIMD[cf32](7.0, 8.0))

    var prod = c1 * c2

    assert_almost_equal(prod.item(0).re, -3.0, "mul failed")
    assert_almost_equal(prod.item(0).im, 4.0, "mul failed")


def test_complex_array_div() raises:
    """Test division of ComplexArray numbers."""
    var c1 = ComplexNDArray[cf32](Shape(2, 2))
    var c2 = ComplexNDArray[cf32](Shape(2, 2))
    c1.itemset(0, ComplexSIMD[cf32](1.0, 2.0))
    c1.itemset(1, ComplexSIMD[cf32](3.0, 4.0))
    c1.itemset(2, ComplexSIMD[cf32](5.0, 6.0))
    c1.itemset(3, ComplexSIMD[cf32](7.0, 8.0))

    c2.itemset(0, ComplexSIMD[cf32](3.0, 4.0))
    c2.itemset(1, ComplexSIMD[cf32](5.0, 6.0))
    c2.itemset(2, ComplexSIMD[cf32](7.0, 8.0))
    c2.itemset(3, ComplexSIMD[cf32](9.0, 10.0))

    var quot = c1 / c2

    assert_almost_equal(quot.item(0).re, 0.44, "div failed")
    assert_almost_equal(quot.item(0).im, 0.08, "div failed")


def _make_complex_2x3() raises -> ComplexNDArray[cf32]:
    var c = ComplexNDArray[cf32](Shape(2, 3))
    # Row 0
    c.itemset(0, ComplexSIMD[cf32](1.0, 0.0))
    c.itemset(1, ComplexSIMD[cf32](2.0, 1.0))
    c.itemset(2, ComplexSIMD[cf32](3.0, -1.0))
    # Row 1
    c.itemset(3, ComplexSIMD[cf32](0.0, 2.0))
    c.itemset(4, ComplexSIMD[cf32](2.0, -3.0))
    c.itemset(5, ComplexSIMD[cf32](5.0, 4.0))
    return c^


def test_complex_array_deep_copy_independent() raises:
    var c = _make_complex_2x3()
    var d = c.deep_copy()
    d.itemset(0, ComplexSIMD[cf32](99.0, 88.0))
    assert_almost_equal(c.item(0).re, 1.0, "deep_copy modified source")
    assert_almost_equal(c.item(0).im, 0.0, "deep_copy modified source")
    assert_almost_equal(d.item(0).re, 99.0, "deep_copy write failed")
    assert_almost_equal(d.item(0).im, 88.0, "deep_copy write failed")


def test_complex_array_setitem_bool_mask_scalar() raises:
    var c = _make_complex_2x3()
    var idx = arange[i32](0, 6, step=1).reshape(Shape(2, 3))
    var mask = idx > 2
    var rhs = ComplexNDArray[cf32](Shape(2, 3))
    rhs.fill(ComplexSIMD[cf32](9.0, -9.0))
    c[mask] = rhs
    assert_almost_equal(c.item(0).re, 1.0, "mask scalar write incorrect")
    assert_almost_equal(c.item(3).re, 9.0, "mask scalar write failed")
    assert_almost_equal(c.item(5).im, -9.0, "mask scalar write failed")


def test_complex_array_lexicographic_comparisons() raises:
    var a = ComplexNDArray[cf32](Shape(3))
    a.itemset(0, ComplexSIMD[cf32](1.0, 2.0))
    a.itemset(1, ComplexSIMD[cf32](2.0, -1.0))
    a.itemset(2, ComplexSIMD[cf32](2.0, 3.0))

    var b = ComplexNDArray[cf32](Shape(3))
    b.itemset(0, ComplexSIMD[cf32](1.0, 3.0))
    b.itemset(1, ComplexSIMD[cf32](2.0, -1.0))
    b.itemset(2, ComplexSIMD[cf32](1.0, 9.0))

    var lt = a < b
    var gt = a > b
    var ne = a != b
    var eq = a == b

    assert_equal(Int(lt.load(0)), 1, "lexicographic < failed at 0")
    assert_equal(Int(lt.load(1)), 0, "lexicographic < failed at 1")
    assert_equal(Int(gt.load(2)), 1, "lexicographic > failed at 2")
    assert_equal(Int(ne.load(0)), 1, "!= failed at 0")
    assert_equal(Int(ne.load(1)), 0, "!= failed at 1")
    assert_equal(Int(eq.load(1)), 1, "== failed at 1")


def test_complex_array_axis_reductions() raises:
    var c = _make_complex_2x3()

    var s0 = c.sum(axis=0)
    assert_almost_equal(s0.item(0).re, 1.0, "sum(axis=0) re[0] failed")
    assert_almost_equal(s0.item(0).im, 2.0, "sum(axis=0) im[0] failed")
    assert_almost_equal(s0.item(1).re, 4.0, "sum(axis=0) re[1] failed")
    assert_almost_equal(s0.item(1).im, -2.0, "sum(axis=0) im[1] failed")

    var s1 = c.sum(axis=1)
    assert_almost_equal(s1.item(0).re, 6.0, "sum(axis=1) re[0] failed")
    assert_almost_equal(s1.item(0).im, 0.0, "sum(axis=1) im[0] failed")
    assert_almost_equal(s1.item(1).re, 7.0, "sum(axis=1) re[1] failed")
    assert_almost_equal(s1.item(1).im, 3.0, "sum(axis=1) im[1] failed")

    var m1 = c.mean(axis=1)
    assert_almost_equal(m1.item(0).re, 2.0, "mean(axis=1) re[0] failed")
    assert_almost_equal(m1.item(0).im, 0.0, "mean(axis=1) im[0] failed")


def test_complex_array_axis_prod_and_cumprod() raises:
    var c = _make_complex_2x3()

    var p1 = c.prod(axis=1)
    # (1+0i)*(2+1i)*(3-1i) = 7+1i
    assert_almost_equal(p1.item(0).re, 7.0, "prod(axis=1) row0 re failed")
    assert_almost_equal(p1.item(0).im, 1.0, "prod(axis=1) row0 im failed")

    var cp1 = c.cumprod(axis=1)
    # Row 0 cumulative: [1+0i, 2+1i, 7+1i]
    assert_almost_equal(cp1.item(0, 0).re, 1.0, "cumprod(axis=1) [0,0] re")
    assert_almost_equal(cp1.item(0, 1).im, 1.0, "cumprod(axis=1) [0,1] im")
    assert_almost_equal(cp1.item(0, 2).re, 7.0, "cumprod(axis=1) [0,2] re")
    assert_almost_equal(cp1.item(0, 2).im, 1.0, "cumprod(axis=1) [0,2] im")


def test_complex_array_axis_arg_and_extrema() raises:
    var c = _make_complex_2x3()

    var am1 = c.argmax(axis=1)
    var an1 = c.argmin(axis=1)
    assert_equal(Int(am1.load(0)), 2, "argmax(axis=1) row0 failed")
    assert_equal(Int(am1.load(1)), 2, "argmax(axis=1) row1 failed")
    assert_equal(Int(an1.load(0)), 0, "argmin(axis=1) row0 failed")
    assert_equal(Int(an1.load(1)), 0, "argmin(axis=1) row1 failed")

    var mx0 = c.max(axis=0)
    var mn0 = c.min(axis=0)
    assert_almost_equal(mx0.item(0).re, 1.0, "max(axis=0) col0 re failed")
    assert_almost_equal(mn0.item(0).re, 0.0, "min(axis=0) col0 re failed")


def test_complex_array_argsort_sort_median() raises:
    var c = ComplexNDArray[cf32](Shape(4))
    c.itemset(0, ComplexSIMD[cf32](2.0, 1.0))
    c.itemset(1, ComplexSIMD[cf32](1.0, 9.0))
    c.itemset(2, ComplexSIMD[cf32](2.0, -3.0))
    c.itemset(3, ComplexSIMD[cf32](0.0, 4.0))

    var idx = c.argsort(axis=0)
    assert_equal(Int(idx.load(0)), 3, "argsort index 0 failed")
    assert_equal(Int(idx.load(1)), 1, "argsort index 1 failed")
    assert_equal(Int(idx.load(2)), 2, "argsort index 2 failed")
    assert_equal(Int(idx.load(3)), 0, "argsort index 3 failed")

    c.sort(axis=0)
    assert_almost_equal(c.item(0).re, 0.0, "sort first re failed")
    assert_almost_equal(c.item(1).re, 1.0, "sort second re failed")
    assert_almost_equal(c.item(2).im, -3.0, "sort third im failed")

    var med = c.median()
    # middle two are (1+9i) and (2-3i) -> (1.5+3i)
    assert_almost_equal(med.re, 1.5, "median re failed")
    assert_almost_equal(med.im, 3.0, "median im failed")


def test_complex_array_tolist_respects_view_order() raises:
    var c = _make_complex_2x3()
    var t = c.T([1, 0])
    var vals = t.tolist()
    # transposed first column should be original [0,0], [1,0]
    assert_almost_equal(vals[0].re, 1.0, "tolist transposed value[0] re")
    assert_almost_equal(vals[0].im, 0.0, "tolist transposed value[0] im")
    assert_almost_equal(vals[1].re, 0.0, "tolist transposed value[1] re")
    assert_almost_equal(vals[1].im, 2.0, "tolist transposed value[1] im")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
