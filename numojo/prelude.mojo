"""
prelude
=======

NuMojo comes a wide range of functions, types, and constants.
If you manually import everything,
it will make the header of the file too long.
On the other hand, using `from numojo import *` would import a lot of functions
that you never use and would pollute the naming space.

This module tries to find out a balance by providing a list of things
that can be imported at one time.
The list contains the functions or types
that are the most essential for a user.

You can use the following code to import them:

```mojo
from numojo.prelude import *
```
"""

import numojo as nm

from numojo.core.item import Item, item
from numojo.core.matrix import Matrix
from numojo.core.ndarray import NDArray
from numojo.core.ndshape import Shape, NDArrayShape
from numojo.core.complex.complex_simd import ComplexSIMD, CScalar
from numojo.core.complex.complex_ndarray import ComplexNDArray
from numojo.core.complex.complex_dtype import (
    ci8,
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
from numojo.core.datatypes import (
    i8,
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
