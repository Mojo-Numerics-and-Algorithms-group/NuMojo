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

import numojo as nm

from numojo.core.indexing.item import Item
from numojo.core.ndarray import NDArray
from numojo.core.accelerator_ndarray import AcceleratorNDArray
from numojo.core.layout import NDArrayShape
from numojo.core.complex.complex_simd import (
    ComplexSIMD,
)
from numojo.core.type_aliases import (
    Shape,
    Strides,
    ComplexScalar,
    CScalar,
    `1j`,
)
from numojo.core.complex.complex_ndarray import ComplexNDArray
from numojo.core.dtype.complex_dtype import (
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
from numojo.core.dtype.default_dtype import (
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
)
