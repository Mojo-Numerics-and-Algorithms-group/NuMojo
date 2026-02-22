# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator submodule
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
=====================================
Accelerator (numojo.core.accelerator)
=====================================

Accelerator (GPU) support namespace for NuMojo.
"""

from .device import Device, cpu, cuda, mps, rocm
