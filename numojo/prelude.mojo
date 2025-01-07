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

from numojo.core.ndarray import NDArray
from numojo.core.item import Item, item
from numojo.core.ndshape import Shape, NDArrayShape

from numojo.core.complex_dtype import CDType
from numojo.core.complex_simd import ComplexSIMD, ComplexScalar
from numojo.core.complex_ndarray import ComplexNDArray

from numojo.core.datatypes import (
    i8,
    i16,
    i32,
    i64,
    isize,
    u8,
    u16,
    u32,
    u64,
    f16,
    f32,
    f64,
    ci8,
    ci16,
    ci32,
    ci64,
    cu8,
    cu16,
    cu32,
    cu64,
    cf16,
    cf32,
    cf64,
)
