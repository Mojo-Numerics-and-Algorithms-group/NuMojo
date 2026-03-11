from numojo.core.accelerator.device import Device
from std.testing.testing import assert_true, assert_equal
from std.testing import TestSuite


def test_default_init() raises:
    var d = Device()
    assert_equal(d.type, "cpu", "default init type")
    assert_equal(d.name, "", "default init name")
    assert_equal(d.id, 0, "default init id")


def test_cpu_constant() raises:
    var d = Device.CPU
    assert_equal(d.type, "cpu", "CPU constant type")
    assert_equal(d.name, "", "CPU constant name")
    assert_equal(d.id, 0, "CPU constant id")


def test_cuda_constant() raises:
    var d = Device.CUDA
    assert_equal(d.type, "gpu", "CUDA constant type")
    assert_equal(d.name, "cuda", "CUDA constant name")
    assert_equal(d.id, 0, "CUDA constant id")


def test_rocm_constant() raises:
    var d = Device.ROCM
    assert_equal(d.type, "gpu", "ROCM constant type")
    assert_equal(d.name, "rocm", "ROCM constant name")
    assert_equal(d.id, 0, "ROCM constant id")


def test_mps_constant() raises:
    var d = Device.MPS
    assert_equal(d.type, "gpu", "MPS constant type")
    assert_equal(d.name, "mps", "MPS constant name")
    assert_equal(d.id, 0, "MPS constant id")


def test_cpu_explicit_init() raises:
    var d = Device(type="cpu", name="", id=0)
    assert_equal(d.type, "cpu", "explicit cpu init type")
    assert_equal(d.name, "", "explicit cpu init name")
    assert_equal(d.id, 0, "explicit cpu init id")


def test_invalid_type_falls_back_to_cpu() raises:
    var d = Device(type="tpu", name="", id=0)
    assert_equal(d.type, "cpu", "invalid type fallback type")
    assert_equal(d.name, "", "invalid type fallback name")
    assert_equal(d.id, 0, "invalid type fallback id")


def test_gpu_empty_name_falls_back_to_cpu() raises:
    var d = Device(type="gpu", name="", id=0)
    assert_equal(d.type, "cpu", "gpu empty name fallback type")
    assert_equal(d.name, "", "gpu empty name fallback name")


def test_cpu_with_nonzero_id_falls_back() raises:
    var d = Device(type="cpu", name="", id=5)
    assert_equal(d.type, "cpu", "cpu nonzero id fallback type")
    assert_equal(d.id, 0, "cpu nonzero id fallback id")


def test_cpu_with_name_falls_back() raises:
    var d = Device(type="cpu", name="something", id=0)
    assert_equal(d.type, "cpu", "cpu with name fallback type")
    assert_equal(d.name, "", "cpu with name fallback name")


def test_invalid_gpu_backend_falls_back() raises:
    var d = Device(type="gpu", name="vulkan", id=0)
    assert_equal(d.type, "cpu", "invalid gpu backend fallback type")
    assert_equal(d.name, "", "invalid gpu backend fallback name")


def test_negative_gpu_id_falls_back() raises:
    var d = Device(type="gpu", name="cuda", id=-1)
    assert_equal(d.type, "cpu", "negative gpu id fallback type")
    assert_equal(d.name, "", "negative gpu id fallback name")


def test_parse_cpu_string() raises:
    var d = Device("cpu")
    assert_equal(d.type, "cpu", "parse 'cpu' type")
    assert_equal(d.name, "", "parse 'cpu' name")
    assert_equal(d.id, 0, "parse 'cpu' id")


def test_parse_cpu_string_variant() raises:
    var d = Device("cpu:0")
    assert_equal(d.type, "cpu", "parse 'cpu:0' type")


def test_parse_empty_falls_back() raises:
    var d = Device.parse_device_string("")
    assert_equal(d.type, "cpu", "parse empty string type")


def test_parse_garbage_falls_back() raises:
    var d = Device.parse_device_string("foobar")
    assert_equal(d.type, "cpu", "parse garbage type")
    assert_equal(d.name, "", "parse garbage name")


def test_parse_colon_no_id_falls_back() raises:
    var d = Device.parse_device_string("cuda:")
    assert_equal(d.type, "cpu", "parse 'cuda:' fallback type")


def test_parse_negative_id_falls_back() raises:
    var d = Device.parse_device_string("cuda:-1")
    assert_equal(d.type, "cpu", "parse negative id fallback type")


def test_parse_non_numeric_id_falls_back() raises:
    var d = Device.parse_device_string("cuda:abc")
    assert_equal(d.type, "cpu", "parse non-numeric id fallback type")


def test_parse_string_containing_cpu_not_matched() raises:
    """Ensure that strings like 'mycpu' or 'notcpu' are not matched as CPU."""
    var d = Device.parse_device_string("mycpu")
    assert_equal(d.type, "cpu", "parse 'mycpu' fallback type")
    assert_equal(d.name, "", "parse 'mycpu' fallback name")
    # 'mycpu' is not a valid backend, so it should fall back to CPU
    # but importantly, it should NOT be matched by the "cpu" in text check


def test_default_init_is_cpu() raises:
    """Default-constructed device should behave as CPU."""
    var d = Device()
    assert_true(d.is_cpu(), "default device is_cpu()")
    assert_true(d.is_available(), "default device is_available()")
    assert_true(d == Device.CPU, "default device equals CPU constant")


def test_unchecked_init() raises:
    var d = Device._unchecked_init(type="gpu", name="cuda", id=3)
    assert_equal(d.type, "gpu", "_unchecked_init type")
    assert_equal(d.name, "cuda", "_unchecked_init name")
    assert_equal(d.id, 3, "_unchecked_init id")


def test_unchecked_init_arbitrary() raises:
    var d = Device._unchecked_init(type="xpu", name="whatever", id=99)
    assert_equal(d.type, "xpu", "_unchecked_init arbitrary type")
    assert_equal(d.name, "whatever", "_unchecked_init arbitrary name")
    assert_equal(d.id, 99, "_unchecked_init arbitrary id")


def test_cpu_fallback() raises:
    var d = Device._cpu_fallback()
    assert_equal(d.type, "cpu", "_cpu_fallback type")
    assert_equal(d.name, "", "_cpu_fallback name")
    assert_equal(d.id, 0, "_cpu_fallback id")


def test_equality_same() raises:
    assert_true(Device.CPU == Device.CPU, "CPU == CPU")
    assert_true(Device.CUDA == Device.CUDA, "CUDA == CUDA")
    assert_true(Device.ROCM == Device.ROCM, "ROCM == ROCM")
    assert_true(Device.MPS == Device.MPS, "MPS == MPS")


def test_equality_equivalent() raises:
    var a = Device(type="cpu", name="", id=0)
    assert_true(a == Device.CPU, "explicit cpu == CPU constant")


def test_inequality_different_type() raises:
    var a = Device._unchecked_init(type="cpu", name="", id=0)
    var b = Device._unchecked_init(type="gpu", name="cuda", id=0)
    assert_true(a != b, "cpu != cuda")


def test_inequality_different_name() raises:
    var a = Device._unchecked_init(type="gpu", name="cuda", id=0)
    var b = Device._unchecked_init(type="gpu", name="rocm", id=0)
    assert_true(a != b, "cuda != rocm")


def test_inequality_different_id() raises:
    var a = Device._unchecked_init(type="gpu", name="cuda", id=0)
    var b = Device._unchecked_init(type="gpu", name="cuda", id=1)
    assert_true(a != b, "cuda:0 != cuda:1")


def test_str_cpu() raises:
    var s = String(Device.CPU)
    assert_equal(s, "Device(type='cpu', name='', id=0)", "__str__ CPU")


def test_str_cuda() raises:
    var s = String(Device.CUDA)
    assert_equal(s, "Device(type='gpu', name='cuda', id=0)", "__str__ CUDA")


def test_str_rocm() raises:
    var s = String(Device.ROCM)
    assert_equal(s, "Device(type='gpu', name='rocm', id=0)", "__str__ ROCM")


def test_str_mps() raises:
    var s = String(Device.MPS)
    assert_equal(s, "Device(type='gpu', name='mps', id=0)", "__str__ MPS")


def test_repr_equals_str() raises:
    var d = Device.CPU
    assert_equal(repr(d), String(d), "__repr__ == __str__")


def test_is_cpu() raises:
    assert_true(Device.CPU.is_cpu(), "CPU.is_cpu()")
    assert_true(not Device.CPU.is_gpu(), "not CPU.is_gpu()")


def test_is_gpu() raises:
    assert_true(Device.CUDA.is_gpu(), "CUDA.is_gpu()")
    assert_true(not Device.CUDA.is_cpu(), "not CUDA.is_cpu()")
    assert_true(Device.ROCM.is_gpu(), "ROCM.is_gpu()")
    assert_true(Device.MPS.is_gpu(), "MPS.is_gpu()")


def test_cpu_is_available() raises:
    assert_true(Device.CPU.is_available(), "CPU.is_available()")


def test_default_device_is_available() raises:
    var d = Device.default_device()
    assert_true(d.is_available(), "default_device().is_available()")


def test_default_device_is_valid_type() raises:
    var d = Device.default_device()
    assert_true(
        d.type == "cpu" or d.type == "gpu",
        "default_device type is cpu or gpu",
    )


def test_available_devices_contains_cpu() raises:
    var listing = Device.available_devices()
    assert_true("cpu" in listing, "available_devices contains cpu")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
