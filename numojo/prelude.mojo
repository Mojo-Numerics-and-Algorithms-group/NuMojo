"""
prelude
=======

Numojo comes a wide range of functions, types, and constants. 
If you have to manually import every thing, 
it would makes the header of the file too long. 
On the other hand, using `from numojo import *` would import a lot of functions 
that you never use and would pollute the naming space.

This module tries to find out a balance by providing a list of things 
that can be imported at one time. 
The lists on contains those functions or types 
that are the most essential for a user. 

You can use the following code to import them:

```mojo
from numojo.prelude import *
```
"""

from .core.ndarray import NDArray
from .core.datatypes import i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64
