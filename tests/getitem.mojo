# Test file for
# numojo.core.sort.argsort()

import numojo as nm
from numojo.core.ndarray import NDArray
import time

fn main() raises:
    test_matrix[nm.i8](4, 4)
    print(str("=")*30)
    test_vector[nm.i8](4)
    print(str("=")*30)
    test_3darray[nm.i8](4,4,4)
    print(str("=")*30)

fn test_matrix[dtype: DType](*shape: Int) raises:
    var A = NDArray[dtype](shape, random=True)
    print("A is a matrix")
    print(A, end="\n\n")

    print("A[0]")
    print(A[0], end="\n\n")
    
    print("A[0, 1]")
    print(A[0, 1], end="\n\n")
    
    print("A[Slice(1,3)]")
    print(A[Slice(1,3)], end="\n\n")
    
    print("A[1, Slice(2,4)]")
    print(A[1, Slice(2,4)], end="\n\n")
    
    print("A[Slice(1,3), Slice(1,3)]")
    print(A[Slice(1,3), Slice(1,3)], end="\n\n")
    
    print("A.item(0,1) as Scalar")
    print(A.item(0, 1), end="\n\n")

fn test_vector[dtype: DType](*shape: Int) raises:
    var A = NDArray[dtype](shape, random=True)
    print("A is a vector")
    print(A, end="\n\n")

    print("A[0]")
    print(A[0], end="\n\n")
    
    print("A[Slice(1,3)]")
    print(A[Slice(1,3)], end="\n\n")
    
    print("A.item(0) as Scalar")
    print(A.item(0), end="\n\n")

fn test_3darray[dtype: DType](*shape: Int) raises:
    var A = NDArray[dtype](shape, random=True)
    print("A is a 3darray")
    print(A, end="\n\n")

    print("A[0]")
    print(A[0], end="\n\n")
    
    print("A[0, 1]")
    print(A[0, 1], end="\n\n")

    print("A[0, 1, 2]")
    print(A[0, 1, 2], end="\n\n")
    
    print("A[Slice(1,3)]")
    print(A[Slice(1,3)], end="\n\n")
    
    print("A[1, Slice(2,4)]")
    print(A[1, Slice(2,4)], end="\n\n")
    
    print("A[Slice(1,3), Slice(1,3), 2]")
    print(A[Slice(1,3), Slice(1,3), 2], end="\n\n")
    
    print("A.item(0,1,2) as Scalar")
    print(A.item(0, 1, 2), end="\n\n")