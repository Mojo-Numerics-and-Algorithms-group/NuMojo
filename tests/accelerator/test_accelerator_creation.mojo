from numojo.core.accelerator.device import Device
from numojo.core.accelerator_ndarray import (
    arange,
    full,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from numojo.prelude import Shape, f32
from std.testing import TestSuite
from std.testing.testing import assert_equal


def test_zeros_cpu() raises:
    var a = zeros[f32, Device.CPU](Shape(2, 3))
    assert_equal(a.shape[0], 2)
    assert_equal(a.shape[1], 3)
    assert_equal(a.item(0, 0), 0.0)
    assert_equal(a.item(1, 2), 0.0)
    assert_equal(a.device.device_name(), "cpu")


def test_ones_cpu() raises:
    var a = ones[f32, Device.CPU](Shape(2, 2))
    assert_equal(a.item(0, 0), 1.0)
    assert_equal(a.item(1, 1), 1.0)


def test_full_cpu() raises:
    var a = full[f32, Device.CPU](Shape(2, 2), 7.0)
    assert_equal(a.item(0, 0), 7.0)
    assert_equal(a.item(1, 1), 7.0)


def test_arange_cpu() raises:
    var a = arange[f32, Device.CPU](0.0, 5.0, 1.0)
    assert_equal(a.shape[0], 5)
    assert_equal(a.item(0), 0.0)
    assert_equal(a.item(4), 4.0)


def test_like_cpu() raises:
    var base = full[f32, Device.CPU](Shape(2, 2), 3.0)
    var z = zeros_like[f32, Device.CPU](base)
    var o = ones_like[f32, Device.CPU](base)
    assert_equal(z.shape[0], 2)
    assert_equal(z.shape[1], 2)
    assert_equal(z.item(0, 0), 0.0)
    assert_equal(o.item(1, 1), 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
