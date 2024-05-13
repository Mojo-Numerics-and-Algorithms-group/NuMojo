import math
from tensor import Tensor
import benchmark
from benchmark.compiler import keep
from testing import assert_raises
import base as tensor_math
"""
Basis of statistics module to be built later
TODO
Build functions for more generic use with axis parameter
max
min
mean
mode
std
var
"""
fn sum[dtype:DType](tensor: Tensor[dtype])->SIMD[dtype,1]:
    """
    Cumulative Sum of a tensor.
    """
    var result: SIMD[dtype,1] = SIMD[dtype,1]()
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(0, opt_nelts*(tensor.num_elements()//opt_nelts), opt_nelts):
        var simd_data = tensor.load[width=opt_nelts](i)
        result+=simd_data.reduce_add()
        
        
       
    if tensor.num_elements()%opt_nelts != 0:
        for i in range(opt_nelts*(tensor.num_elements()//opt_nelts), tensor.num_elements()):
                var simd_data = tensor.load[width=1](i)
                result+=simd_data.reduce_add()
    return result

fn prod[dtype:DType](tensor: Tensor[dtype])->SIMD[dtype,1]:
    """
    Cumulative Product of a tensor.
    """
    var result: SIMD[dtype,1] = SIMD[dtype,1]()
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(0, opt_nelts*(tensor.num_elements()//opt_nelts), opt_nelts):
        var simd_data = tensor.load[width=opt_nelts](i)
        result*=simd_data.reduce_mul()
        
        
       
    if tensor.num_elements()%opt_nelts != 0:
        for i in range(opt_nelts*(tensor.num_elements()//opt_nelts), tensor.num_elements()):
                var simd_data = tensor.load[width=1](i)
                result*=simd_data.reduce_mul()
    return result

fn mean[dtype:DType](tensor: Tensor[dtype])->SIMD[dtype,1]:
    """
    Cumulative Arithmatic Mean of a tensor.
    """
    return sum[dtype](tensor)/tensor.num_elements()

fn std[dtype:DType](tensor: Tensor[dtype])->SIMD[dtype,1]:
    """
    Population standard deviation of a tensor.
    """
    var sumv = sum[dtype](tensor)
    var n = tensor.num_elements()
    return math.sqrt[dtype](sum[dtype]((tensor-(sumv/n))**2)/n)

fn variance[dtype:DType](tensor: Tensor[dtype])->SIMD[dtype,1]:
    """
    Population variance of a tensor.
    """
    var sumv = sum[dtype](tensor)
    var n = tensor.num_elements()
    return sum[dtype]((tensor-(sumv/n))**2)/n

