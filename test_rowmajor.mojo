from tensor import Tensor
import benchmark
from benchmark.compiler import keep
from testing import assert_raises

from numojo.array_routines import *
from numojo.cumulative_reduce import *
from numojo.ndarray import *

fn main() raises:
    ## ND arrays

    # traverse_iterative
    var orig = Array[DType.float16](VariadicList[Int](3, 2), random=True)
    var narr = Array[DType.float16](VariadicList[Int](3, 2), random=True)
    print(orig)
    print()
    print(narr)
    var index = List[Int](0, 0)
    numojo.ndarray._traverse_iterative[DType.float16](orig, narr, orig._arrayInfo.dims, orig._arrayInfo.weights, 0, index, 0)
    print(narr)
    print(orig.mdot(narr))

    # * ROW MAJOR INDEXING
    var arr = Array[DType.float16](VariadicList[Int](2, 3, 3), random=True)
    print("2x3x3 array row major")
    print(arr)
    print()
    # print(arr[0,0,0], arr[0,0,1], arr[0,0,2], "\n",
    #     arr[0,1,0], arr[0,1,1], arr[0,1,2], "\n",
    #     arr[0,2,0], arr[0,2,1], arr[0,2,2])
    # print()
    # print(arr[1,0,0], arr[1,0,1], arr[1,0,2], "\n",
    #     arr[1,1,0], arr[1,1,1], arr[1,1,2], "\n",
    #     arr[1,2,0], arr[1,2,1], arr[1,2,2])

    # slicing doesn't support single integer index for now, therefore [:,:,2] 
    # should be written as [:,:,2:3] -> both select 2x3 array out of 2x3x3 array
    # * Following are exmaples of slicing operations that can be compared. 

    var sliced = arr[:,:,2:3] # selects 2 x 3 x 1 array out of  2 x 3 x 3
    print("2x3 ARRAY - arr[:,:,2:3]")
    # print()
    # print(sliced[0,0], sliced[0,1], sliced[0,2], "\n",
    # sliced[1,0], sliced[1,1], sliced[1,2])
    # print()
    # print("*****************")
    print(sliced)
    print()

    var sliced1 = arr[:, ::2, 0:1]
    print("2x2 ARRAY - arr[:, ::2, 0:1]")
    # print()
    # print(sliced1[0,0], sliced1[0,1], "\n",
    #     sliced1[1,0], sliced1[1,1])
    # print()
    # print("*****************")
    print(sliced1)
    print()

    var sliced2 = arr[:,:,::2]
    print("2x3x2 ARRAY - arr[:,:,::2]")
    print(sliced2)
    print()
    # print()
    # print(sliced2[0,0,0], sliced2[0,1,0], sliced2[0,2,0], "\n",
    #     sliced2[1,0,0], sliced2[1,1,0], sliced2[1,2,0])
    # print()
    # print(sliced2[0,0,1], sliced2[0,1,1], sliced2[0,2,1], "\n",
    #     sliced2[1,0,1], sliced2[1,1,1], sliced2[1,2,1])
    # print()
    # print("*****************")


    var sliced3 = arr[:, ::2, ::2]
    print("2x2x2 ARRAY - arr[:, ::2, ::2]")
    print(sliced3)
    print()
    # print()
    # print(sliced3[0,0,0], sliced3[0,1,0], "\n",
    #     sliced3[1,0,0], sliced3[1,1,0])
    # print()
    # print(sliced3[0,0,1], sliced3[0,1,1], "\n",
    #     sliced3[1,0,1], sliced3[1,1,1])
    # print()
    # print("*****************")

    # # var arr:array[DType.float64] = array[DType.float64](VariadicList[Int](3, 3, 3), random=True)
    # # print("3x3x3 ARRAY")
    # # print(arr[0,0,0], arr[0,1,0], arr[0,2,0], "\n",
    # #     arr[1,0,0], arr[1,1,0], arr[1,2,0], "\n",
    # #     arr[2,0,0], arr[2,1,0], arr[2,2,0])
    # # print()
    # # print(arr[0,0,1], arr[0,1,1], arr[0,2,1], "\n",
    # #     arr[1,0,1], arr[1,1,1], arr[1,2,1], "\n",
    # #     arr[2,0,1], arr[2,1,1], arr[2,2,1])
    # # print()
    # # print(arr[0,0,2], arr[0,1,2], arr[0,2,2], "\n",
    # #     arr[1,0,2], arr[1,1,2], arr[1,2,2], "\n",
    # #     arr[2,0,2], arr[2,1,2], arr[2,2,2])
    # # print()
    # # var sliced = arr[:,:,1:3]
