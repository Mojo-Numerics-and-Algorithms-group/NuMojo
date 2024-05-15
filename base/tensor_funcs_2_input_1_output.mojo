import math
from tensor import Tensor
import benchmark
from benchmark.compiler import keep
from testing import assert_raises

fn _math_func[dtype:DType, func: fn[type:DType, simd_w:Int](SIMD[type, simd_w],SIMD[type, simd_w]) -> SIMD[type, simd_w]](tensor1: Tensor[dtype], tensor2: Tensor[dtype])raises->Tensor[dtype]:
    if tensor1.shape() != tensor2.shape():
        with assert_raises():
            raise "Shape Mismatch error shapes must match for this function"
    var result_tensor: Tensor[dtype]=Tensor[dtype](tensor1.shape())
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(0, opt_nelts*(tensor1.num_elements()//opt_nelts), opt_nelts):
        var simd_data1 = tensor1.load[width=opt_nelts](i)
        var simd_data2 = tensor2.load[width=opt_nelts](i)
        result_tensor.store[width=opt_nelts](i,func[dtype,opt_nelts](simd_data1,simd_data2))
        
       
    if tensor1.num_elements()%opt_nelts != 0:
        for i in range(opt_nelts*(tensor1.num_elements()//opt_nelts), tensor1.num_elements()):
                var simd_data1 = tensor1.load[width=1](i)
                var simd_data2 = tensor2.load[width=1](i)
                result_tensor.store[width=1](i, func[dtype, 1](simd_data1,simd_data2))
    return result_tensor

fn copysign[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    return _math_func[dtype,math.copysign](tensor1, tensor2)

fn nextafter[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    return _math_func[dtype,math.nextafter](tensor1, tensor2)

fn scalb[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    return _math_func[dtype,math.scalb](tensor1, tensor2)

fn remainder[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    return _math_func[dtype,math.remainder](tensor1, tensor2)

fn _math_func_simd_int[dtype:DType, func: fn[type:DType, simd_w:Int](SIMD[type, simd_w],Int) -> SIMD[type, simd_w]](tensor1: Tensor[dtype], intval:Int)->Tensor[dtype]:
    var result_tensor: Tensor[dtype]=Tensor[dtype](tensor1.shape())
    alias opt_nelts = simdwidthof[dtype]()
    for i in range(0, opt_nelts*(tensor1.num_elements()//opt_nelts), opt_nelts):
        var simd_data1 = tensor1.load[width=opt_nelts](i)
        
        result_tensor.store[width=opt_nelts](i,func[dtype,opt_nelts](simd_data1,intval))
        
       
    if tensor1.num_elements()%opt_nelts != 0:
        for i in range(opt_nelts*(tensor1.num_elements()//opt_nelts), tensor1.num_elements()):
                var simd_data1 = tensor1.load[width=1](i)
                result_tensor.store[width=1](i, func[dtype, 1](simd_data1,intval))
    return result_tensor

fn pow[dtype:DType](tensor1:Tensor[dtype],intval:Int)->Tensor[dtype]:
    return _math_func_simd_int[dtype,math.pow](tensor1, intval)


# fn main():
#     var tens1:Tensor[DType.float32] = Tensor[DType.float32](100,100)
#     var tens2:Tensor[DType.float32] = Tensor[DType.float32](100,100)
#     for i in range(10_000):
#         tens1[i]= SIMD[DType.float32,1](3.141592/4)
#         tens2[i]= SIMD[DType.float32,1](3.141592)
#     var res:Tensor[DType.float32]
#     fn test_mod()capturing:
#         try:
#             res = mod[DType.float32](tens1, tens2)
#             keep(res.data())
#         except:
#             print('mod: Failed shape error')
#     var report_mod = benchmark.run[test_mod]()
#     print('mod f32 100x100')
#     report_mod.print()
#     fn test_mul()capturing:
#         try:
#             res = mul[DType.float32](tens1, tens2)
#             keep(res.data())
#         except:
#             print('mul: Failed shape error')
#     var report_mul = benchmark.run[test_mul]()
#     print('mul f32 100x100')
#     report_mul.print()
#     fn test_sub()capturing:
#         try:
#             res = sub[DType.float32](tens1, tens2)
#             keep(res.data())
#         except:
#             print('sub: Failed shape error')
#     var report_sub = benchmark.run[test_sub]()
#     print('sub f32 100x100')
#     report_sub.print()
#     fn test_add()capturing:
#         try:
#             res = add[DType.float32](tens1, tens2)
#             keep(res.data())
#         except:
#             print('add: Failed shape error')
#     var report_add = benchmark.run[test_add]()
#     print('add f32 100x100')
#     report_add.print()
#     fn test_div()capturing:
#         try:
#             res = div[DType.float32](tens1, tens2)
#             keep(res.data())
#         except:
#             print('div: Failed shape error')
#     var report_div = benchmark.run[test_div]()
#     print('div f32 100x100')
#     report_div.print()
#     fn test_copysign()capturing:
#         try:
#             res = copysign[DType.float32](tens1, tens2)
#             keep(res.data())
#         except:
#             print('copysign: Failed shape error')
#     var report_copysign = benchmark.run[test_copysign]()
#     print('copysign f32 100x100')
#     report_copysign.print()
#     fn test_atan2()capturing:
#         try:
#             res = atan2[DType.float32](tens1, tens2)
#             keep(res.data())
#         except:
#             print('atan2: Failed shape error')
#     var report_atan2 = benchmark.run[test_atan2]()
#     print('atan2 f32 100x100')
#     report_atan2.print()
#     fn test_hypot()capturing:
#         try:
#             res = hypot[DType.float32](tens1, tens2)
#             keep(res.data())
#         except:
#             print('hypot: Failed shape error')
#     var report_hypot = benchmark.run[test_hypot]()
#     print('hypot f32 100x100')
#     report_hypot.print()
#     fn test_nextafter()capturing:
#         try:
#             res = nextafter[DType.float32](tens1, tens2)
#             keep(res.data())
#         except:
#             print('nextafter: Failed shape error')
#     var report_nextafter = benchmark.run[test_nextafter]()
#     print('nextafter f32 100x100')
#     report_nextafter.print()
#     fn test_scalb()capturing:
#         try:
#             res = scalb[DType.float32](tens1, tens2)
#             keep(res.data())
#         except:
#             print('scalb: Failed shape error')
#     var report_scalb = benchmark.run[test_scalb]()
#     print('scalb f32 100x100')
#     report_scalb.print()
#     fn test_remainder()capturing:
#         try:
#             res = remainder[DType.float32](tens1, tens2)
#             keep(res.data())
#         except:
#             print('remainder: Failed shape error')
#     var report_remainder = benchmark.run[test_remainder]()
#     print('remainder f32 100x100')
#     report_remainder.print()
