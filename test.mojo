import math
from tensor import Tensor
import benchmark
from benchmark.compiler import keep
from testing import assert_raises
import base as tensor_math
"""
TODO
Build functions for
max
min
mean
mode
std
var
"""
fn sum[dtype:DType](tensor: Tensor[dtype])->SIMD[dtype,1]:
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
    return sum[dtype](tensor)/tensor.num_elements()

fn std[dtype:DType](tensor: Tensor[dtype])->SIMD[dtype,1]:
    """
    Population standard deviation.
    """
    var sumv = sum[dtype](tensor)
    var n = tensor.num_elements()
    return math.sqrt[dtype](sum[dtype]((tensor-(sumv/n))**2)/n)

fn variance[dtype:DType](tensor: Tensor[dtype])->SIMD[dtype,1]:
    """
    Population variance.
    """
    var sumv = sum[dtype](tensor)
    var n = tensor.num_elements()
    return sum[dtype]((tensor-(sumv/n))**2)/n

def main():
    var tens:Tensor[DType.float64] = Tensor[DType.float64](100,100)
    for i in range(10_000):
        tens[i]= SIMD[DType.float64,1](3.141592/4)
    var res:SIMD[DType.float64,1]
    fn test_abs()capturing:
        res = sum[DType.float64](tens)
        keep(res)
    var report_abs = benchmark.run[test_abs]()
    print('sum f64 100x100')
    report_abs.print()
    fn test_floor()capturing:
        res = std[DType.float64](tens)
        keep(res)
    var report_floor = benchmark.run[test_floor]()
    print('std f64 100x100')
    report_floor.print()
    res = std[DType.float64](tens)
    print(res)