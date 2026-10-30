# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator array creation tests
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Accelerator creation tests (tests.accelerator.test_accelerator_creation).
=========================================================================
Tests accelerator creation routines and compact layout metadata.

Exports
-------
- `test_0d_array_cpu`, `test_zeros_cpu`, `test_ones_cpu`, `test_full_cpu`.
- `test_arange_cpu`: Tests range creation.
- `test_like_cpu`, `test_layout_metadata_cpu`, `test_high_rank_layout_cpu`.
- `test_layout_transform_views_cpu`: Tests layout-backed view transforms.
- `main`: Runs the discovered tests.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.testing import assert_false, assert_true, TestSuite
from std.testing.testing import assert_equal

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.accelerator.device import Device
from numojo.core.accelerator_ndarray import (
    AcceleratorNDArray,
    arange,
    full,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from numojo.core.layout.ndlayout import Shape
from numojo.prelude import f32


def test_0d_array_cpu() raises:
    """Tests creation and scalar access for a rank-zero accelerator array."""
    var scalar = zeros[f32, Device.CPU](Shape())
    assert_equal(scalar.ndim(), 0)
    assert_equal(scalar.size(), 1)
    assert_equal(len(scalar.shape()), 0)
    assert_equal(len(scalar.strides()), 0)
    assert_equal(scalar[], 0.0)

    scalar.itemset(0, 42.0)
    assert_equal(scalar[], 42.0)
    assert_true(scalar.flags.C_CONTIGUOUS)
    assert_true(scalar.flags.F_CONTIGUOUS)


def test_zeros_cpu() raises:
    var a = zeros[f32, Device.CPU](Shape([2, 3]))
    assert_equal(a.shape()[0], 2)
    assert_equal(a.shape()[1], 3)
    assert_equal(a.item(0, 0), 0.0)
    assert_equal(a.item(1, 2), 0.0)
    assert_equal(a.device.device_name(), "cpu")


def test_ones_cpu() raises:
    var a = ones[f32, Device.CPU](Shape([2, 2]))
    assert_equal(a.item(0, 0), 1.0)
    assert_equal(a.item(1, 1), 1.0)


def test_full_cpu() raises:
    var a = full[f32, Device.CPU](Shape([2, 2]), 7.0)
    assert_equal(a.item(0, 0), 7.0)
    assert_equal(a.item(1, 1), 7.0)


def test_arange_cpu() raises:
    var a = arange[f32, Device.CPU](0.0, 5.0, 1.0)
    assert_equal(a.shape()[0], 5)
    assert_equal(a.item(0), 0.0)
    assert_equal(a.item(4), 4.0)


def test_like_cpu() raises:
    var base = full[f32, Device.CPU](Shape([2, 2]), 3.0)
    var z = zeros_like[f32, Device.CPU](base)
    var o = ones_like[f32, Device.CPU](base)
    assert_equal(z.shape()[0], 2)
    assert_equal(z.shape()[1], 2)
    assert_equal(z.item(0, 0), 0.0)
    assert_equal(o.item(1, 1), 1.0)


def test_layout_metadata_cpu() raises:
    """Tests accelerator metadata access and view layout copies."""
    var a = zeros[f32, Device.CPU](Shape([2, 3]))
    assert_equal(a.ndim(), 2)
    assert_equal(a.size(), 6)
    assert_equal(a.offset(), 0)
    assert_equal(a.strides()[0], 3)
    assert_equal(a.strides()[1], 1)
    assert_true(a.flags.C_CONTIGUOUS)

    var row = a[1]
    assert_equal(row.ndim(), 1)
    assert_equal(row.shape()[0], 3)
    assert_equal(row.strides()[0], 1)
    assert_equal(row.offset(), 3)
    assert_false(row.flags.OWNDATA)


def test_layout_transform_views_cpu() raises:
    """Tests layout-backed slice, select, transpose, and reshape views."""
    var array = zeros[f32, Device.CPU](Shape(2, 3))
    for i in range(array.size()):
        array.itemset(i, Float32(i))

    var selected = array.select(0, 1)
    assert_equal(selected.shape().tolist(), [3])
    assert_equal(selected.offset(), 3)
    assert_equal(selected.item(2), 5.0)
    assert_false(selected.flags.OWNDATA)

    var transposed = array.transpose()
    assert_equal(transposed.shape().tolist(), [3, 2])
    assert_equal(transposed.strides().tolist(), [1, 3])
    assert_equal(transposed.item(2, 1), 5.0)
    assert_true(transposed.flags.F_CONTIGUOUS)

    var reshaped = array.reshape(Shape(3, 2))
    assert_equal(reshaped.shape().tolist(), [3, 2])
    assert_equal(reshaped.strides().tolist(), [2, 1])
    assert_equal(reshaped.item(2, 1), 5.0)

    var reversed = array.slice(1, 2, 3, -1)
    assert_equal(reversed.shape().tolist(), [2, 3])
    assert_equal(reversed.strides().tolist(), [3, -1])
    assert_equal(reversed.offset(), 2)
    assert_equal(reversed.item(0, 0), 2.0)
    assert_equal(reversed.item(1, 2), 3.0)

    var empty = array.slice(0, 0, 0)
    assert_equal(empty.shape().tolist(), [0, 3])
    assert_equal(empty.size(), 0)
    assert_true(empty.flags.C_CONTIGUOUS)


def test_high_rank_layout_cpu() raises:
    """Tests metadata spilling beyond the rank-eight inline capacity."""
    var shape = List[Int](length=9, fill=1)
    shape[8] = 2
    var a = AcceleratorNDArray[f32, Device.CPU](shape)
    var copied = a.copy()
    var view = a.view()

    assert_equal(a.ndim(), 9)
    assert_equal(a.size(), 2)
    assert_equal(a.shape()[8], 2)
    assert_equal(a.strides()[8], 1)
    assert_equal(copied.shape().tolist(), a.shape().tolist())
    assert_equal(view.shape().tolist(), a.shape().tolist())
    assert_false(view.flags.OWNDATA)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
