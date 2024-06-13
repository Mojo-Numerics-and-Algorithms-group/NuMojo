# import math
from tensor import Tensor
import benchmark
from benchmark.compiler import keep
from testing import assert_raises
# import base as tensor_math
from numojo.array_routines import *
from numojo.cumulative_reduce import *

fn main():
    var start: Float64 = 1.0
    var stop: Float64 = 1000.0
    var step: Float64 = 1.0
    var num: Int = 1000
    var arange_tensor = arange[DType.float64](start, stop, step)
    var linspace_tensor = linspace[DType.float64](start, stop, num, endpoint=True)
    var linspace_parallel_tensor = linspace[DType.float64](start, stop, num, parallel=True)
    var logspace_tensor = logspace[DType.float64](-3.0, 0.0, num, endpoint=True)
    var logspace_parallel_tensor = logspace[DType.float64](-3.0, 0.0, num, parallel=True)
    print("arange: ", arange_tensor) # prints 10000 values from 1.0 to 1000.0
    print("linspace: ", linspace_tensor) # prints 10000 values from 1.0 to 1000.0
    print("linspace_parallel: ", linspace_parallel_tensor) # prints 10000 values from 1.0 to 1000.0
    print("logspace: ", logspace_tensor) # prints 10000 values from 10**-3 to 10**0
    print("logspace_parallel: ", logspace_parallel_tensor) # prints 10000 values from 10**-3 to 10**0
