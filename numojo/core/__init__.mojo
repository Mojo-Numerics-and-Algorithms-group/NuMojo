# ===----------------------------------------------------------------------=== #
# NuMojo: Core types and utilities (numojo.core)
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Core (numojo.core).
===================
Foundational data structures and utilities for NuMojo: arrays, memory
layouts, dtype aliases, error handling, and complex number support.

Exports
-------
- `NDArray`, `AcceleratorNDArray`: Core array containers.
- `ComplexNDArray`, `ComplexSIMD`: Complex number support.
- `NDArrayShape`, `NDArrayStrides`, `Flags`, `newaxis`: Layout metadata.
- `IndexMethods`, `Item`, `TraverseMethods`, `Validator`: Indexing helpers.
- `DataContainer`, `HostStorage`, `DeviceStorage`, `AcceleratorDataContainer`:
  Memory storage types.
- `NumojoError`, `terminate`: Error handling.
- Dtype aliases (`i8`, `f32`, `boolean`, ...) and complex counterparts.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from . import (
    accelerator,
    complex,
    dtype,
    layout,
    memory,
    traits,
)
from .accelerator import (
    cpu,
    cuda,
    Device,
    mps,
    rocm,
)
from .accelerator_ndarray import AcceleratorNDArray
from .complex import (
    ComplexNDArray,
    ComplexSIMD,
)
from .dtype import (
    bf16,
    boolean,
    cbf16,
    cboolean,
    cf16,
    cf32,
    cf64,
    ci128,
    ci16,
    ci256,
    ci32,
    ci64,
    ci8,
    cint,
    cinvalid,
    ComplexDType,
    cu128,
    cu16,
    cu256,
    cu32,
    cu64,
    cu8,
    cuint,
    f16,
    f32,
    f64,
    i128,
    i16,
    i256,
    i32,
    i64,
    i8,
    int,
    u128,
    u16,
    u256,
    u32,
    u64,
    u8,
    uint,
)
from .error import (
    NumojoError,
    terminate,
)
from .indexing import (
    IndexMethods,
    Item,
    TraverseMethods,
    Validator,
)
from .layout import (
    Flags,
    NDArrayShape,
    NDArrayStrides,
    newaxis,
)
from .memory import (
    AcceleratorDataContainer,
    DataContainer,
    DeviceStorage,
    HostStorage,
)
from .ndarray import NDArray
from .type_aliases import (
    `1j`,
    ComplexScalar,
    CScalar,
    Shape,
    Strides,
)
