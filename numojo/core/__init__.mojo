# ===----------------------------------------------------------------------=== #
# NuMojo: Core types and utilities (numojo.core)
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Core (numojo.core)
---------------------
This sub module provides the core types and utilities for NuMojo, including fundamental data structures
like `NDArray` and `Matrix`, dtype aliases, memory layout definitions, error handling utilities, and complex number support.
It serves as the foundational layer upon which higher-level routines and algorithms are built.
Fundamental types and utilities for NuMojo: arrays, matrices, memory layouts, data types, and error handling.
"""
# ===----------------------------------------------------------------------===#
# Local
# ===----------------------------------------------------------------------===#
from . import (
    accelerator,
    complex,
    dtype,
    layout,
    matrix,
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
from .matrix import Matrix
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


