# ===----------------------------------------------------------------------=== #
# NuMojo: compile-fail case, run by tests/test_accelerator_compile_fail.sh.
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# ===----------------------------------------------------------------------=== #
# A read-only view cannot write an element.
# EXPECT-ERROR: violated constraint

from numojo.core.accelerator.device import Device
from numojo.core.accelerator_ndarray import AcceleratorNDArray, zeros
from numojo.core.accelerator.ndarray_view import AcceleratorNDArrayView
from numojo.core.layout.ndlayout import Shape


def main() raises:
    var array = zeros[DType.float32, Device.CPU](Shape(2))
    var view = array.view()
    view.itemset(0, 1.0)
