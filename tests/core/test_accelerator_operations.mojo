"""
Tests for `AcceleratorNDArray` elementwise operation dispatch.
"""

from numojo.core.accelerator.device import Device
from numojo.core.accelerator_ndarray import full, zeros
from numojo.core.layout.ndshape import NDArrayShape as Shape
from std.testing.testing import assert_true, assert_equal
from std.testing import TestSuite
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)


def test_add_cpu_uniform() raises:
    var a = full[DType.float32, Device.CPU](Shape(8), 3.0)
    var b = full[DType.float32, Device.CPU](Shape(8), 4.0)
    var c = a + b
    for i in range(8):
        assert_equal(c.item(i), 7.0, "cpu add uniform")


def test_add_cpu_distinct_values() raises:
    var a = zeros[DType.float32, Device.CPU](Shape(6))
    var b = zeros[DType.float32, Device.CPU](Shape(6))
    for i in range(6):
        a.itemset(i, Float32(i))
        b.itemset(i, Float32(100 * i))
    var c = a + b
    for i in range(6):
        assert_equal(
            c.item(i), Float32(i) + Float32(100 * i), "cpu add distinct"
        )


def test_add_cpu_shape_mismatch_raises() raises:
    var a = full[DType.float32, Device.CPU](Shape(4), 1.0)
    var b = full[DType.float32, Device.CPU](Shape(5), 1.0)
    var raised = False
    try:
        _ = a + b
    except:
        raised = True
    assert_true(raised, "shape mismatch should raise")


def _run_gpu_add_test[device: Device]() raises:
    var a_host = zeros[DType.float32, Device.CPU](Shape(6))
    var b_host = zeros[DType.float32, Device.CPU](Shape(6))
    for i in range(6):
        a_host.itemset(i, Float32(i))
        b_host.itemset(i, Float32(100 * i))

    var a_gpu = a_host.to[device]()
    var b_gpu = b_host.to[device]()
    var c_gpu = a_gpu + b_gpu
    var c_host = c_gpu.to_host()

    for i in range(6):
        assert_equal(
            c_host.item(i),
            Float32(i) + Float32(100 * i),
            "gpu add matches expected (device=" + String(device) + ")",
        )


def test_add_gpu_matches_cpu_when_available() raises:
    comptime if has_nvidia_gpu_accelerator():
        _run_gpu_add_test[Device.CUDA]()
    elif has_amd_gpu_accelerator():
        _run_gpu_add_test[Device.ROCM]()
    elif has_apple_gpu_accelerator():
        _run_gpu_add_test[Device.MPS]()
    else:
        assert_true(True, "no gpu available; gpu add test skipped")


def main() raises:
    if (
        has_nvidia_gpu_accelerator()
        or has_amd_gpu_accelerator()
        or has_apple_gpu_accelerator()
    ):
        TestSuite.discover_tests[__functions_in_module()]().run()
    else:
        print("No GPU available; skipping GPU tests.")
