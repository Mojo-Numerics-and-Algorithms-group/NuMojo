# ===----------------------------------------------------------------------=== #
# NuMojo: compile-fail case, run by tests/test_accelerator_compile_fail.sh.
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# ===----------------------------------------------------------------------=== #
# A read-only view cannot hand out a mutable pointer.
# EXPECT-ERROR: cannot mutate a read-only view

from numojo.core.accelerator.device import Device
from numojo.core.accelerator_ndarray import AcceleratorNDArray, zeros
from numojo.core.accelerator.ndarray_view import AcceleratorNDArrayView
from numojo.core.layout.ndlayout import Shape


def main() raises:
    var array = zeros[DType.float32, Device.CPU](Shape(2))
    var view = array.view()
    view.unsafe_mut_ptr()[] = 1.0
