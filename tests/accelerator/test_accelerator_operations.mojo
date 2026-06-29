"""
Tests for `AcceleratorNDArray` elementwise operation dispatch.
"""

from numojo.core.accelerator.device import Device
from numojo.core.accelerator_ndarray import AcceleratorNDArray, full, zeros
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


def _make_distinct_operands() raises -> (
    Tuple[
        AcceleratorNDArray[DType.float32, Device.CPU],
        AcceleratorNDArray[DType.float32, Device.CPU],
    ]
):
    var a = zeros[DType.float32, Device.CPU](Shape(6))
    var b = zeros[DType.float32, Device.CPU](Shape(6))
    for i in range(6):
        a.itemset(i, Float32(i + 1))
        b.itemset(i, Float32(2 * (i + 1)))
    return (a^, b^)


def test_add_cpu_distinct_values() raises:
    var ab = _make_distinct_operands()
    var c = ab[0] + ab[1]
    for i in range(6):
        assert_equal(
            c.item(i), ab[0].item(i) + ab[1].item(i), "cpu add distinct"
        )


def test_sub_cpu_distinct_values() raises:
    var ab = _make_distinct_operands()
    var c = ab[0] - ab[1]
    for i in range(6):
        assert_equal(
            c.item(i), ab[0].item(i) - ab[1].item(i), "cpu sub distinct"
        )


def test_mul_cpu_distinct_values() raises:
    var ab = _make_distinct_operands()
    var c = ab[0] * ab[1]
    for i in range(6):
        assert_equal(
            c.item(i), ab[0].item(i) * ab[1].item(i), "cpu mul distinct"
        )


def test_div_cpu_distinct_values() raises:
    var ab = _make_distinct_operands()
    var c = ab[0] / ab[1]
    for i in range(6):
        assert_equal(
            c.item(i), ab[0].item(i) / ab[1].item(i), "cpu div distinct"
        )


def test_neg_cpu_distinct_values() raises:
    var ab = _make_distinct_operands()
    var c = -ab[0]
    for i in range(6):
        assert_equal(c.item(i), -ab[0].item(i), "cpu neg distinct")


def test_sum_cpu_distinct_values() raises:
    var ab = _make_distinct_operands()
    var a = ab[0].copy()
    # a = [1, 2, 3, 4, 5, 6] -> expected sum 21
    var expected = Float32(0)
    for i in range(6):
        expected += a.item(i)
    assert_equal(a.sum(), expected, "cpu sum distinct")
    assert_equal(a.sum(), Float32(21), "cpu sum distinct matches hand-computed value")


def test_add_cpu_shape_mismatch_raises() raises:
    var a = full[DType.float32, Device.CPU](Shape(4), 1.0)
    var b = full[DType.float32, Device.CPU](Shape(5), 1.0)
    var raised = False
    try:
        _ = a + b
    except:
        raised = True
    assert_true(raised, "shape mismatch should raise")


def _run_gpu_elementwise_op_test[device: Device]() raises:
    var ab = _make_distinct_operands()
    var a_gpu = ab[0].to[device]()
    var b_gpu = ab[1].to[device]()

    var add_host = (a_gpu + b_gpu).to_host()
    var sub_host = (a_gpu - b_gpu).to_host()
    var mul_host = (a_gpu * b_gpu).to_host()
    var div_host = (a_gpu / b_gpu).to_host()
    var neg_host = (-a_gpu).to_host()

    for i in range(6):
        var av = ab[0].item(i)
        var bv = ab[1].item(i)
        assert_equal(add_host.item(i), av + bv, "gpu add matches cpu")
        assert_equal(sub_host.item(i), av - bv, "gpu sub matches cpu")
        assert_equal(mul_host.item(i), av * bv, "gpu mul matches cpu")
        assert_equal(div_host.item(i), av / bv, "gpu div matches cpu")
        assert_equal(neg_host.item(i), -av, "gpu neg matches cpu")


def test_elementwise_ops_gpu_match_cpu_when_available() raises:
    comptime if has_nvidia_gpu_accelerator():
        _run_gpu_elementwise_op_test[Device.CUDA]()
    elif has_amd_gpu_accelerator():
        _run_gpu_elementwise_op_test[Device.ROCM]()
    elif has_apple_gpu_accelerator():
        _run_gpu_elementwise_op_test[Device.MPS]()
    else:
        assert_true(True, "no gpu available; gpu binary op tests skipped")


def _run_gpu_sum_test[device: Device]() raises:
    # Larger, non-uniform array spanning multiple thread blocks so the
    # reduction kernel's block-tree-reduction + host-combine path is
    # actually exercised (not just a single-block edge case).
    var size = 1000
    var a = zeros[DType.float32, Device.CPU](Shape(size))
    var expected = Float32(0)
    for i in range(size):
        var v = Float32((i % 53) - 26)
        a.itemset(i, v)
        expected += v

    var a_gpu = a.to[device]()
    var gpu_sum = a_gpu.sum()
    assert_equal(gpu_sum, expected, "gpu sum matches hand-accumulated cpu sum")
    assert_equal(gpu_sum, a.sum(), "gpu sum matches AcceleratorNDArray cpu sum")


def test_sum_gpu_matches_cpu_when_available() raises:
    comptime if has_nvidia_gpu_accelerator():
        _run_gpu_sum_test[Device.CUDA]()
    elif has_amd_gpu_accelerator():
        _run_gpu_sum_test[Device.ROCM]()
    elif has_apple_gpu_accelerator():
        _run_gpu_sum_test[Device.MPS]()
    else:
        assert_true(True, "no gpu available; gpu sum test skipped")


def main() raises:
    if (
        has_nvidia_gpu_accelerator()
        or has_amd_gpu_accelerator()
        or has_apple_gpu_accelerator()
    ):
        TestSuite.discover_tests[__functions_in_module()]().run()
    else:
        print("No GPU available; skipping GPU tests.")
