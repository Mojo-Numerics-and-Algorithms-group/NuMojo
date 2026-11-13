# ===----------------------------------------------------------------------=== #
# NuMojo: compile-fail case, run by tests/test_accelerator_compile_fail.sh.
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# ===----------------------------------------------------------------------=== #
# A read-only owner borrow cannot create a mutable view.
# EXPECT-ERROR: invalid call to 'view_mut'

from numojo.core.accelerator.device import Device
from numojo.core.accelerator_ndarray import AcceleratorNDArray, zeros
from numojo.core.accelerator.ndarray_view import AcceleratorNDArrayView
from numojo.core.layout.ndlayout import Shape


def write_first(array: AcceleratorNDArray[DType.float32, Device.CPU]) raises:
    var view = array.view_mut()
    view.itemset(0, 1.0)


def main() raises:
    var array = zeros[DType.float32, Device.CPU](Shape(2))
    write_first(array)
