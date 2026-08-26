# ===----------------------------------------------------------------------=== #
# NuMojo: A numerical computation library for Mojo.
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Prelude (numojo.prelude).
=========================
Core types and common utilities for day-to-day NuMojo usage.

Exports
-------
- Container types: `NDArray`.
- Shape/index helpers: `Shape`, `NDArrayShape`, `Item`.
- Dtype aliases: `f32`, `f64`, `i32`, `boolean`.
- Complex helpers: `ComplexSIMD`, `ComplexScalar`, `CScalar`, `1j`.

Usage:
```mojo
from numojo.prelude import *
```

For more functions (math, linalg, statistics), import from `numojo.routines.*`.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
import numojo as nm
from numojo.core.accelerator_ndarray import AcceleratorNDArray
from numojo.core.complex.complex_ndarray import ComplexNDArray
from numojo.core.complex.complex_simd import ComplexSIMD
from numojo.core.dtype.complex_dtype import (
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
    cu128,
    cu16,
    cu256,
    cu32,
    cu64,
    cu8,
    cuint,
)
from numojo.core.dtype.default_dtype import (
    bf16,
    boolean,
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
from numojo.core.indexing.item import Item
from numojo.core.layout import NDArrayShape
from numojo.core.ndarray import NDArray
from numojo.core.type_aliases import (
    `1j`,
    ComplexScalar,
    CScalar,
    Shape,
    Strides,
)
