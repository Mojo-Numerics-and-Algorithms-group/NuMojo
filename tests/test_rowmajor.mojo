from tensor import Tensor
import benchmark
from benchmark.compiler import keep
from testing import assert_raises
from random import rand

from numojo import *

import time

struct Vector[dtype: DType]():
    var data: DTypePointer[dtype]
    var shape: List[Int]

    fn __init__(inout self, shape:List[Int]):
        self.data = DTypePointer[dtype].alloc(shape[0])
        self.shape = shape
        rand[dtype](self.data, shape[0])

    fn __init__(inout self, data: DTypePointer[dtype], shape: List[Int]):
        self.data = data
        self.shape = shape
        
fn main() raises:
    # var shape: List[Int] = List[Int](10)
    # var v1: Vector[DType.float32] = Vector[DType.float32](shape)
    # # var v2: Vector[DType.float32] = Vector[DType.float32] (data, shape)
    # var v2 = Vector[DType.float32](v1.data, shape)
    # print(v1.data[0], v2.data[0], v1.data[1], v2.data[1])
    # print(v1.data, v2.data)
    # v1.data[0] = 10.0
    # print(v1.data[0], v2.data[0], v1.data[1], v2.data[1])
    # print(v1.data, v2.data)
    # v1.data.free()
    # print(v1.data, v2.data)
    # print(y.info.ndim, y.info.offset, y.info.shape[0], y.info.shape[1],
    # y.info.strides[0], y.info.strides[1])
    # print(y[0,0])
    # ## ND arrays
    # var t0 = time.now()

    # var new = NDArray[DType.int8](List[Int8](1,2,3,4,5,6), shape=List[Int](2,3))
    # print(new)

    # var orig = NDArray[DType.float32](VariadicList[Int](100, 100), random=True)
    # var narr = NDArray[DType.float32](VariadicList[Int](100, 100), random=True)
    # # var index = List[Int](0, 0)
    # # numojo.ndarray._traverse_iterative[DType.float16](orig, narr, orig._arrayInfo.dims, orig._arrayInfo.weights, 0, index, 0)
    # var _temp = orig.mdot(narr)
    # print(_temp)
    # print((time.now()-t0)/10e9, "seconds")

    # # # * ROW MAJOR INDEXING
    var arr = NDArray[f16](VariadicList[Int](2, 3, 3), random=True)
    print("2x3x3 array row major")
    print(arr)
    print()
    # # # # slicing doesn't support single integer index for now, therefore [:,:,2] 
    # # # # should be written as [:,:,2:3] -> both select 2x3 array out of 2x3x3 array
    # # # # * Following are exmaples of slicing operations that can be compared. 

    var sliced = arr[:,:,2:3] # selects 2 x 3 x 1 array out of  2 x 3 x 3
    # var sliced = arr[Slice(0,2),Slice(0,3),2] # selects 2 x 3 x 1 array out of  2 x 3 x 3
    print("2x3 ARRAY - arr[:,:,2:3]")
    print(sliced)
    print(sliced.stride)
    print(sliced.ndshape)
    print()

    var sliced1 = arr[:, ::2, 0:1]
    print("2x2 ARRAY - arr[:, ::2, 0:1]")
    print(sliced1)
    arr[0] = 10.0
    print()

    var sliced2 = arr[:,:,::2]
    print("2x3x2 ARRAY - arr[:,:,::2]")
    print(sliced2)
    print()

    var sliced3 = arr[:, ::2, ::2]
    print("2x2x2 ARRAY - arr[:, ::2, ::2]")
    print(sliced3)
    print()
