# ===----------------------------------------------------------------------=== #
# NuMojo: compile-fail case, run by tests/test_accelerator_compile_fail.sh.
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# ===----------------------------------------------------------------------=== #
# An owner passed mutably cannot alias a view passed in the same call.
# EXPECT-ERROR: aliasing values passed mutably

from numojo.core.accelerator.device import Device
from numojo.core.accelerator_ndarray import AcceleratorNDArray, zeros
from numojo.core.accelerator.ndarray_view import AcceleratorNDArrayView
from numojo.core.layout.ndlayout import Shape


def add_first(
    mut target: AcceleratorNDArray[DType.float32, Device.CPU],
    source: AcceleratorNDArrayView[DType.float32, Device.CPU, _],
) raises:
    target.itemset(0, target.item(0) + source.item(0))


def main() raises:
    var array = zeros[DType.float32, Device.CPU](Shape(2))
    add_first(array, array.view())
