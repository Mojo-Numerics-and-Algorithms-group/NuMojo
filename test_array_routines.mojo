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

    print()
    var arr = Tensor[DType.float16]().rand(TensorShape(2,2))
    print("arr: ", arr)
    var arr_sum = sum(arr)
    print("arr_sum: ", arr_sum)
    var arr_prod = prod(arr)
    print("arr_prod: ", arr_prod)
    var arr_mean = mean(arr)
    print("arr_mean: ", arr_mean)
    var arr_max = maxT(arr)
    print("arr_max: ", arr_max)
    var arr_min = minT(arr)
    print("arr_min: ", arr_min)
    var arr_variance = variance(arr)
    print("arr_variance: ", arr_variance)
    var arr_stdev = stdev(arr)
    print("arr_stdev: ", arr_stdev)
    var arr_pvariance = pvariance(arr)
    print("arr_pvariance: ", arr_pvariance)
    var arr_pstdev = pstdev(arr)
    print("arr_pstdev: ", arr_pstdev)
