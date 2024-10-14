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

from .core.ndarray import NDArray
from .core.index import Idx
from .core.ndarrayshape import NDArrayShape
from .core.datatypes import i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64
