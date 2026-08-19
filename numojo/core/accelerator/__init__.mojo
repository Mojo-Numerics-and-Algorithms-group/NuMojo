# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator (GPU) support
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Accelerator (numojo.core.accelerator)
======================================

Accelerator (GPU) support namespace for NuMojo.

Exports
-------
- `Device`: Abstract device interface.
- `DeviceHandle`: Handle for device instances.
- `DeviceSpec`: Device specification and capabilities.
- `cpu`: CPU device instance.
- `cuda`: CUDA device instance.
- `mps`: Metal Performance Shaders device instance.
- `rocm`: ROCm device instance.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from .device import (
    cpu,
    cuda,
    Device,
    DeviceHandle,
    DeviceSpec,
    mps,
    rocm,
)
