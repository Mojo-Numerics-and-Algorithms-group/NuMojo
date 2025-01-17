import numojo as nm
from numojo.prelude import *
from memory import UnsafePointer
from python import Python
from collections import Dict
from sys import simdwidthof


fn main() raises:
    var a = nm.arange[i8](6)
    print(a[List[Bool](True, False, True, True, False, True)])

    var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
    print(b[List[Bool](False, True)])
