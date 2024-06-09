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
    var linspace_tensor = linspace[DType.float64](start, stop, num)
    var linspace_parallel_tensor = linspace[DType.float64](start, stop, num, parallel=True)
    var logspace_tensor = logspace[DType.float64](-3.0, 0.0, num)
    var logspace_parallel_tensor = logspace[DType.float64](-3.0, 0.0, num, parallel=True)
    print("arange: ", arange_tensor)
    print("linspace: ", linspace_tensor)
    print("linspace_parallel: ", linspace_parallel_tensor)
    print("logspace: ", logspace_tensor)
    print("logspace_parallel: ", logspace_parallel_tensor)
