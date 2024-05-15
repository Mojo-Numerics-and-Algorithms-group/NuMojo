import math
import ._math_funcs as _mf

# ===------------------------------------------------------------------------===#
# Addition/Subtraction
# ===------------------------------------------------------------------------===#

fn sub[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.sub](tensor1, tensor2)

fn add[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.add](tensor1, tensor2)

# ===------------------------------------------------------------------------===#
# Multiplication/Division
# ===------------------------------------------------------------------------===#

fn mod[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.mod](tensor1, tensor2)

fn mul[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.mul](tensor1, tensor2)

fn div[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.div](tensor1, tensor2)


fn reciprocal[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.reciprocal](tensor)

# ===------------------------------------------------------------------------===#
# Exponents/Roots
# ===------------------------------------------------------------------------===#
fn cbrt[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.cbrt](tensor)

fn rsqrt[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.rsqrt](tensor)

fn sqrt[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.sqrt](tensor)

fn exp2[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.exp2](tensor)

fn exp[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.exp](tensor)

fn expm1[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.expm1](tensor)
# ===------------------------------------------------------------------------===#
# Logarithms
# ===------------------------------------------------------------------------===#

fn log[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.log](tensor)

fn log2[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.log2](tensor)

fn log10[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.log10](tensor)

fn log1p[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.log1p](tensor)

# ===------------------------------------------------------------------------===#
# Rounding and Similiar concepts
# ===------------------------------------------------------------------------===#

fn abs[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.abs](tensor)

fn floor[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.floor](tensor)

fn ceil[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.ceil](tensor)

fn trunc[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.trunc](tensor)

fn round[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.round](tensor)

fn roundeven[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.roundeven](tensor)

fn round_half_down[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.round_half_down](tensor)

fn round_half_up[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.round_half_up](tensor)