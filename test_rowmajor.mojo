from tensor import Tensor
import benchmark
from benchmark.compiler import keep
from testing import assert_raises

from numojo.ndarray import NDArray

import time

fn main() raises:
    ## ND arrays
    var t0 = time.now()

    var orig = NDArray[DType.float32](VariadicList[Int](100, 100), random=True)
    var narr = NDArray[DType.float32](VariadicList[Int](100, 100), random=True)
    # var index = List[Int](0, 0)
    # numojo.ndarray._traverse_iterative[DType.float16](orig, narr, orig._arrayInfo.dims, orig._arrayInfo.weights, 0, index, 0)
    var _temp = orig.mdot(narr)
    print(_temp)
    print((time.now()-t0)/10e9, "seconds")

    # # * ROW MAJOR INDEXING
    var arr = NDArray[DType.int64](VariadicList[Int](2, 2, 2), random=True)
    print("2x2x2 array row major")
    print(arr)
    print()
    # # slicing doesn't support single integer index for now, therefore [:,:,2] 
    # # should be written as [:,:,2:3] -> both select 2x3 array out of 2x3x3 array
    # # * Following are exmaples of slicing operations that can be compared. 

    var sliced = arr[:,:,2:3] # selects 2 x 3 x 1 array out of  2 x 3 x 3
    print("2x3 ARRAY - arr[:,:,2:3]")
    print(sliced)
    print()

    var sliced1 = arr[:, ::2, 0:1]
    print("2x2 ARRAY - arr[:, ::2, 0:1]")
    print(sliced1)
    print()

    var sliced2 = arr[:,:,::2]
    print("2x3x2 ARRAY - arr[:,:,::2]")
    print(sliced2)
    print()

    var sliced3 = arr[:, ::2, ::2]
    print("2x2x2 ARRAY - arr[:, ::2, ::2]")
    print(sliced3)
    print()