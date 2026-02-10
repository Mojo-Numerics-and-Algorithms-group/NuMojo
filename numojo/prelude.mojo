"""
NuMojo Prelude (`numojo.prelude`)
================================

The prelude is the recommended “batteries-included” import for day-to-day use.

Why it exists:
- Importing everything from `numojo` is convenient but pollutes your namespace.
- Importing every symbol manually makes headers long and repetitive.

What it exports:
- Core container types like `Matrix` and `NDArray`
- Shape/index helpers like `Shape`, `NDArrayShape`, and `Item`
- Common dtype aliases (e.g. `f32`, `f64`, `i32`, `boolean`)
- Complex number helpers (`ComplexSIMD`, `ComplexScalar`, `CScalar`, `1j`)

Usage:
```mojo
from numojo.prelude import *
```

For more functions (math, linalg, statistics, etc.), import them from
`numojo.routines.*` (or directly from `numojo`) as needed.
"""

import numojo as nm

from numojo.core.indexing.item import Item
from numojo.core.matrix import Matrix
from numojo.core.ndarray import NDArray
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
