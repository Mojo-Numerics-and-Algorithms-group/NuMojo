import math
from tensor import Tensor
import benchmark
from benchmark.compiler import keep
from testing import assert_raises

fn _math_func_shift[shift:Int, dtype:DType, func: fn[shift:Int ,type:DType,  simd_w:Int](SIMD[type, simd_w]) -> SIMD[type, simd_w]](tensor: Tensor[dtype])->Tensor[dtype]:
    var result_tensor: Tensor[dtype]=Tensor[dtype](tensor.shape())
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(0, opt_nelts*(tensor.num_elements()//opt_nelts), opt_nelts):
        var simd_data1 = tensor.load[width=opt_nelts](i)
        
        result_tensor.store[width=opt_nelts](i,func[shift, dtype,opt_nelts](simd_data1))
        
       
    if tensor.num_elements()%opt_nelts != 0:
        for i in range(opt_nelts*(tensor.num_elements()//opt_nelts), tensor.num_elements()):
                var simd_data1 = tensor.load[width=1](i)
                result_tensor.store[width=1](i, func[shift, dtype, 1](simd_data1))
    return result_tensor

fn rotate_bits_left[shift:Int, dtype:DType](tensor1:Tensor[dtype])->Tensor[dtype]:
    return _math_func_shift[shift, dtype, math.rotate_bits_left](tensor1)

fn rotate_bits_right[shift:Int, dtype:DType](tensor1:Tensor[dtype])->Tensor[dtype]:
    return _math_func_shift[shift, dtype, math.rotate_bits_right](tensor1)

fn rotate_left[shift:Int, dtype:DType](tensor1:Tensor[dtype])->Tensor[dtype]:
    return _math_func_shift[shift, dtype,math.rotate_left](tensor1)

fn rotate_right[shift:Int, dtype:DType](tensor1:Tensor[dtype])->Tensor[dtype]:
    return _math_func_shift[shift, dtype,math.rotate_right](tensor1)


# fn main():