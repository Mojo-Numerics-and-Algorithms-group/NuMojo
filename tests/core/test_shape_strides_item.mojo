from numojo.prelude import *
from std.testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close
from std.testing import TestSuite


def test_shape() raises:
    var A = nm.NDArrayShape(2, 3, 4)
    assert_true(
        A[-1] == 4,
        msg=String("`NDArrayShape.__getitem__()` fails: may overflow"),
    )


def test_strides() raises:
    var A = nm.NDArrayStrides(2, 3, 4)
    assert_true(
        A[-1] == 4,
        msg=String("`NDArrayStrides.__getitem__()` fails: may overflow"),
    )
    assert_true(
        A[-2] == 3,
        msg=String("`NDArrayStrides.__getitem__()` fails: may overflow"),
    )


def test_item() raises:
    var A = nm.Item(2, 3, 4)
    assert_true(
        A[-1] == 4,
        msg=String("`NDArrayStrides.__getitem__()` fails: may overflow"),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
