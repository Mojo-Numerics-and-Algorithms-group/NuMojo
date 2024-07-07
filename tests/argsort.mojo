# Test file for
# numojo.core.sort.argsort()

import numojo as nm
from numojo.core.ndarray import NDArray
import time

fn main() raises:
    test[nm.f64](6)
    test[nm.i32](12)
    test[nm.f64](1000)

fn test[dtype: DType](length: Int) raises:
    # Initialize an ND arrays of type
    var t0 = time.now()
    var A = NDArray[dtype](length, random=True)
    var idx = nm.core.sort.argsort(A)
    print("Array:", A)
    print("Sorted indices:", idx)
    print("Sorted array", A[idx])
    print((time.now() - t0)/1e9, "s")
