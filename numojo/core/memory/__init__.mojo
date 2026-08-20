# ===----------------------------------------------------------------------=== #
# NuMojo: Memory management and storage
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Memory (numojo.core.memory).
============================

Low-level memory and storage utilities used by NuMojo core containers.

Exports
-------
- `DataContainer`: Abstract data container interface.
- `HostStorage`: Host (CPU) memory storage.
- `DeviceStorage`: Device (GPU) memory storage.
- `AcceleratorDataContainer`: Accelerator-aware data container.
- `from_dlpack`: DLPack interoperability function.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from .data_container import DataContainer
from .dlpack import from_dlpack
from .storage import (
    AcceleratorDataContainer,
    DeviceStorage,
    HostStorage,
)
