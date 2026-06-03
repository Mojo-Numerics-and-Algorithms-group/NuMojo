# ===----------------------------------------------------------------------=== #
# NuMojo: Core types and utilities (numojo.core)
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""Core (numojo.core)

This sub module provides the core types and utilities for NuMojo, including fundamental data structures
like `NDArray` and `Matrix`, dtype aliases, memory layout definitions, error handling utilities, and complex number support.
It serves as the foundational layer upon which higher-level routines and algorithms are built.
Fundamental types and utilities for NuMojo: arrays, matrices, memory layouts, data types, and error handling.
"""

from numojo.core.ndarray import NDArray

from numojo.core.type_aliases import (
    Shape,
    Strides,
    ComplexScalar,
    CScalar,
    `1j`,
)

from numojo.core.error import (
    terminate,
    NumojoError,
)

from numojo.core.matrix import Matrix

from numojo.core.layout import (
    NDArrayShape,
    NDArrayStrides,
    Flags,
)

from numojo.core.dtype import (
    i8,
    i16,
    i32,
    i64,
    i128,
    i256,
    int,
    u8,
    u16,
    u32,
    u64,
    u128,
    u256,
    uint,
    bf16,
    f16,
    f32,
    f64,
    boolean,
    ComplexDType,
    ci8,
    ci16,
    ci32,
    ci64,
    ci128,
    ci256,
    cint,
    cu8,
    cu16,
    cu32,
    cu64,
    cu128,
    cu256,
    cuint,
    cbf16,
    cf16,
    cf32,
    cf64,
    cboolean,
    cinvalid,
)

from numojo.core.complex import (
    ComplexSIMD,
    ComplexNDArray,
)

from numojo.core.memory import (
    DataContainer,
    HostStorage,
    DeviceStorage,
    AcceleratorDataContainer,
)
from numojo.core.accelerator import Device, cpu, cuda, mps, rocm

from numojo.core.indexing import Item, IndexMethods, TraverseMethods, Validator

import numojo.core.dtype
import numojo.core.layout
import numojo.core.memory
import numojo.core.matrix
import numojo.core.complex
import numojo.core.traits
import numojo.core.accelerator
