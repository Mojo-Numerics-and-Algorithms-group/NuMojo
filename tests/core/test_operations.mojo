import numojo as nm
from numojo.prelude import *
from std.testing.testing import assert_true
from std.testing import TestSuite


def test_ndarray_multiplication_commutes() raises:
    var A = nm.ones[nm.f64](nm.Shape(2, 2))
    var B = nm.ones[nm.f64](nm.Shape(2, 2))
    var L = 2.0 * (A @ B)
    var R = (A @ B) * 2.0
    assert_true((L == R).all())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
