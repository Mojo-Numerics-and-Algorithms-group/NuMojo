from numojo.prelude import *
from testing.testing import assert_true, assert_almost_equal, assert_equal
from utils_for_test import check, check_is_close
from testing import TestSuite


def test_shape():
    var A = nm.NDArrayShape(2, 3, 4)
    assert_true(
        A[-1] == 4,
        msg=String("`NDArrayShape.__getitem__()` fails: may overflow"),
    )


def test_strides():
    var A = nm.NDArrayStrides(2, 3, 4)
    assert_true(
        A[-1] == 4,
        msg=String("`NDArrayStrides.__getitem__()` fails: may overflow"),
    )
    assert_true(
        A[-2] == 3,
        msg=String("`NDArrayStrides.__getitem__()` fails: may overflow"),
    )


def test_item():
    var A = nm.Item(2, 3, 4)
    assert_true(
        A[-1] == 4,
        msg=String("`NDArrayStrides.__getitem__()` fails: may overflow"),
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
