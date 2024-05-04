import math
from tensor import Tensor
import benchmark
from benchmark.compiler import keep
from testing import assert_raises

fn _math_func[dtype:DType, func: fn[type:DType, simd_w:Int](SIMD[type, simd_w],SIMD[type, simd_w],SIMD[type, simd_w]) -> SIMD[type, simd_w]](tensor1: Tensor[dtype], tensor2: Tensor[dtype], tensor3: Tensor[dtype])raises->Tensor[dtype]:
    if not ((tensor1.shape() == tensor2.shape()) and (tensor1.shape() == tensor3.shape())):
        with assert_raises():
            raise "Shape Mismatch error shapes must match for this function"
    var result_tensor: Tensor[dtype]=Tensor[dtype](tensor1.shape())
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(0, opt_nelts*(tensor1.num_elements()//opt_nelts), opt_nelts):
        var simd_data1 = tensor1.load[width=opt_nelts](i)
        var simd_data2 = tensor2.load[width=opt_nelts](i)
        var simd_data3 = tensor2.load[width=opt_nelts](i)
        result_tensor.store[width=opt_nelts](i,func[dtype,opt_nelts](simd_data1,simd_data2,simd_data3))
        
       
    if tensor1.num_elements()%opt_nelts != 0:
        for i in range(opt_nelts*(tensor1.num_elements()//opt_nelts), tensor1.num_elements()):
                var simd_data1 = tensor1.load[width=1](i)
                var simd_data2 = tensor2.load[width=1](i)
                var simd_data3 = tensor2.load[width=1](i)
                result_tensor.store[width=1](i, func[dtype, 1](simd_data1,simd_data2,simd_data3))
    return result_tensor

fn clamp[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype],tensor3:Tensor[dtype])raises->Tensor[dtype]:
    return _math_func[dtype, math.clamp](tensor1, tensor2, tensor3)