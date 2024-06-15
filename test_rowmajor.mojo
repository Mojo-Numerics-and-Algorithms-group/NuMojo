from tensor import Tensor
import benchmark
from benchmark.compiler import keep
from testing import assert_raises

from numojo.ndarray import Array

fn main() raises:
    ## ND arrays
    var orig = Array[DType.float16](VariadicList[Int](2, 2), random=True)
    print(orig, "\n")
    var narr = Array[DType.float16](VariadicList[Int](2, 2), random=True)
    print(narr, "\n")
    # var index = List[Int](0, 0)
    # numojo.ndarray._traverse_iterative[DType.float16](orig, narr, orig._arrayInfo.dims, orig._arrayInfo.weights, 0, index, 0)
    # print(narr)
    print(orig.mdot(narr), "\n")
    print(orig.vdot(narr), "\n")

    # # * ROW MAJOR INDEXING
    var arr = Array[DType.float16](VariadicList[Int](2, 3, 3), random=True)
    print("2x3x3 array row major")
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