# Test file for
# numojo.core.sort.argsort()

import numojo as nm
from numojo.core.ndarray import NDArray
import time

fn main() raises:
    test[nm.i8](4, 4)

fn test[dtype: DType](*shape: Int) raises:
    var A = NDArray[dtype](shape, random=True)
    print(A)

    print("A[0]")
    print(A[0])
    
    print("A[0, 1]")
    print(A[0, 1])
    
    print("A[Slice(1,3)]")
    print(A[Slice(1,3)])
    
    print("A[1, Slice(2,4)]")
    print(A[1, Slice(2,4)])
    
    print("A[Slice(1,3), Slice(1,3)]")
    print(A[Slice(1,3), Slice(1,3)])
    
    print("A at (0,1) as scalar")
    print(A.at(0, 1))