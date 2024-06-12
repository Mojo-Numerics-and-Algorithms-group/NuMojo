from tensor import Tensor
import benchmark
from benchmark.compiler import keep
from testing import assert_raises

from numojo.array_routines import *
from numojo.cumulative_reduce import *
from numojo.ndarray import *

fn main() raises:
    ## Basic array routines (returns Tensor type for now)
    # var start: Float64 = 1.0
    # var stop: Float64 = 1000.0
    # var step: Float64 = 1.0
    # var num: Int = 1000
    # var arange_tensor = arange[DType.float64](start, stop, step)
    # var linspace_tensor = linspace[DType.float64](start, stop, num)
    # var linspace_parallel_tensor = linspace[DType.float64](start, stop, num, parallel=True)
    # var logspace_tensor = logspace[DType.float64](-3.0, 0.0, num)
    # var logspace_parallel_tensor = logspace[DType.float64](-3.0, 0.0, num, parallel=True)
    # print("arange: ", arange_tensor)
    # print("linspace: ", linspace_tensor)
    # print("linspace_parallel: ", linspace_parallel_tensor)
    # print("logspace: ", logspace_tensor)
    # print("logspace_parallel: ", logspace_parallel_tensor)

    ## ND arrays
    # * COLUMN MAJOR INDEXING
    var arr = array[DType.int8](VariadicList[Int](5, 10), random=True)
    print(arr)

    print("2x3x3 ARRAY")
    print(arr[0,0,0], arr[0,1,0], arr[0,2,0], "\n",
        arr[1,0,0], arr[1,1,0], arr[1,2,0])
    print()
    print(arr[0,0,1], arr[0,1,1], arr[0,2,1], "\n",
        arr[1,0,1], arr[1,1,1], arr[1,2,1])
    print()
    print(arr[0,0,2], arr[0,1,2], arr[0,2,2], "\n",
        arr[1,0,2], arr[1,1,2], arr[1,2,2])
    print()

    # slicing doesn't support single integer index for now, therefore [:,:,2] 
    # should be written as [:,:,2:3] -> both select 2x3 array out of 2x3x3 array
    # * Following are exmaples of slicing operations that can be compared. 

    var sliced = arr[:,:,2:3] # selects 2 x 3 x 1 array out of  2 x 3 x 3
    print("2x3 ARRAY - arr[:,:,2:3]")
    print()
    print(sliced[0,0], sliced[0,1], sliced[0,2], "\n",
        sliced[1,0], sliced[1,1], sliced[1,2])
    print()
    print("*****************")

    var sliced1 = arr[:, ::2, 0:1]
    print("2x2 ARRAY - arr[:, ::2, 0:1]")
    print()
    print(sliced1[0,0], sliced1[0,1], "\n",
        sliced1[1,0], sliced1[1,1])
    print()
    print("*****************")

    var sliced2 = arr[:,:,::2]
    print("2x3x2 ARRAY - arr[:,:,::2]")
    print()
    print(sliced2[0,0,0], sliced2[0,1,0], sliced2[0,2,0], "\n",
        sliced2[1,0,0], sliced2[1,1,0], sliced2[1,2,0])
    print()
    print(sliced2[0,0,1], sliced2[0,1,1], sliced2[0,2,1], "\n",
        sliced2[1,0,1], sliced2[1,1,1], sliced2[1,2,1])
    print()
    print("*****************")


    var sliced3 = arr[:, ::2, ::2]
    print("2x2x2 ARRAY - arr[:, ::2, ::2]")
    print()
    print(sliced3[0,0,0], sliced3[0,1,0], "\n",
        sliced3[1,0,0], sliced3[1,1,0])
    print()
    print(sliced3[0,0,1], sliced3[0,1,1], "\n",
        sliced3[1,0,1], sliced3[1,1,1])
    print()
    print("*****************")

    # var arr:array[DType.float64] = array[DType.float64](VariadicList[Int](3, 3, 3), random=True)
    # print("3x3x3 ARRAY")
    # print(arr[0,0,0], arr[0,1,0], arr[0,2,0], "\n",
    #     arr[1,0,0], arr[1,1,0], arr[1,2,0], "\n",
    #     arr[2,0,0], arr[2,1,0], arr[2,2,0])
    # print()
    # print(arr[0,0,1], arr[0,1,1], arr[0,2,1], "\n",
    #     arr[1,0,1], arr[1,1,1], arr[1,2,1], "\n",
    #     arr[2,0,1], arr[2,1,1], arr[2,2,1])
    # print()
    # print(arr[0,0,2], arr[0,1,2], arr[0,2,2], "\n",
    #     arr[1,0,2], arr[1,1,2], arr[1,2,2], "\n",
    #     arr[2,0,2], arr[2,1,2], arr[2,2,2])
    # print()
    # var sliced = arr[:,:,1:3]
