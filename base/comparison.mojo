import math
import ._math_funcs as _mf
from tensor import Tensor

fn greater[dtype:DType](tensor1:Tensor[dtype], tensor2:Tensor[dtype])raises->Tensor[DType.bool]:
    return _mf._math_func_compare_2_tensors[dtype,math.greater](tensor1, tensor2)

fn greater_equal[dtype:DType](tensor1:Tensor[dtype], tensor2:Tensor[dtype])raises->Tensor[DType.bool]:
    return _mf._math_func_compare_2_tensors[dtype,math.greater_equal](tensor1, tensor2)

fn less[dtype:DType](tensor1:Tensor[dtype], tensor2:Tensor[dtype])raises->Tensor[DType.bool]:
    return _mf._math_func_compare_2_tensors[dtype,math.less](tensor1, tensor2)

fn less_equal[dtype:DType](tensor1:Tensor[dtype], tensor2:Tensor[dtype])raises->Tensor[DType.bool]:
    return _mf._math_func_compare_2_tensors[dtype,math.less_equal](tensor1, tensor2)

fn equal[dtype:DType](tensor1:Tensor[dtype], tensor2:Tensor[dtype])raises->Tensor[DType.bool]:
    return _mf._math_func_compare_2_tensors[dtype,math.equal](tensor1, tensor2)

fn not_equal[dtype:DType](tensor1:Tensor[dtype], tensor2:Tensor[dtype])raises->Tensor[DType.bool]:
    return _mf._math_func_compare_2_tensors[dtype,math.not_equal](tensor1, tensor2)
