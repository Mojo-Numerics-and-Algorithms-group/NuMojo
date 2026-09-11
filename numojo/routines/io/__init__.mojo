# ===----------------------------------------------------------------------=== #
# NuMojo: I/O and file operations
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
I/O routines (numojo.routines.io).
==================================
File I/O operations and array formatting for NuMojo.

Exports
-------
- `load`, `loadtxt`: Functions for reading arrays from files.
- `save`, `savetxt`: Functions for writing arrays to files.
- `load_npy`, `save_npy`: Pure-Mojo NumPy `.npy` reader/writer (no Python).
- `load_torch`, `save_torch`: Pure-Mojo PyTorch `.pt` reader/writer for a
  single plain tensor (no Python).
- `set_printoptions`: Configure array printing options.
- `PrintOptions`: Array printing configuration.
- `format_floating_scientific`: Scientific notation formatting.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from .files import (
    load,
    loadtxt,
    save,
    savetxt,
)
from .formatting import (
    format_floating_scientific,
    PrintOptions,
    set_printoptions,
)
from .npy import load_npy, save_npy
from .torch import load_torch, save_torch
