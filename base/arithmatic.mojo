import math
import ._math_funcs as _mf
from tensor import Tensor
"""
implements arithmetic functions
"""
# ===------------------------------------------------------------------------===#
# Addition/Subtraction
# ===------------------------------------------------------------------------===#

fn add[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    """
    Perform addition on two tensors.
        
    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
    
    Args:
        tensor1: A tensor.
        tensor2: A tensor.
    
    Returns:
        The elementwise sum of `tensor1` and`tensor2`.
    """
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.add](tensor1, tensor2)

fn sub[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    """
    Perform subtraction on two tensors.
        
    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
    
    Args:
        tensor1: A tensor.
        tensor2: A tensor.
    
    Returns:
        The elementwise difference of `tensor1` and`tensor2`.
    """
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.sub](tensor1, tensor2)
# ===------------------------------------------------------------------------===#
# Multiplication/Division
# ===------------------------------------------------------------------------===#

fn copysign[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    """
    Copy the sign of the first tensor and apply it to the second tensor.
        
    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
    
    Args:
        tensor1: A tensor.
        tensor2: A tensor.
    
    Returns:
        The second tensor multipied by the sign of the first tensor.
    """
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.copysign](tensor1, tensor2)

fn mod[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    """
    Elementwise modulo of tensor1 and tensor2.
        
    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
    
    Args:
        tensor1: A tensor.
        tensor2: A tensor.
    
    Returns:
        A tensor equal to tensor1%tensor2.
    """
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.mod](tensor1, tensor2)

fn mul[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    """
    Elementwise product of tensor1 and tensor2.
        
    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
    
    Args:
        tensor1: A tensor.
        tensor2: A tensor.
    
    Returns:
        A tensor equal to tensor1*tensor2.
    """
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.mul](tensor1, tensor2)

fn div[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    """
    Elementwise quotent of tensor1 and tensor2.
        
    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
    
    Args:
        tensor1: A tensor.
        tensor2: A tensor.
    
    Returns:
        A tensor equal to tensor1/tensor2.
    """
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.div](tensor1, tensor2)

fn remainder[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    """
    Elementwise remainders of tensor.
        
    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
    
    Args:
        tensor1: A tensor.
        tensor2: A tensor.
    
    Returns:
        A tensor equal to tensor1//tensor2.
    """
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.remainder](tensor1, tensor2)

fn reciprocal[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    """
    Elementwise reciprocals of tensor1 and tensor2.
        
    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
    
    Args:
        tensor: A tensor.
    
    Returns:
        A tensor equal to 1/tensor.
    """
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.reciprocal](tensor)

# ===------------------------------------------------------------------------===#
# Exponents/Roots
# ===------------------------------------------------------------------------===#
fn cbrt[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    """
    Elementwise cuberoot of tensor.
        
    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
    
    Args:
        tensor: A tensor.
    
    Returns:
        A tensor equal to tensor**(1/3).
    """
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.cbrt](tensor)

fn pow[dtype:DType](tensor1:Tensor[dtype],intval:Int)->Tensor[dtype]:
    """
    Elementwise tensor to the power of intval.
        
    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
    
    Args:
        tensor1: A tensor.
        intval: An integer.
    
    Returns:
        A tensor equal to tensor**intval.
    """
    return _mf._math_func_simd_int[dtype,math.pow](tensor1, intval)

fn rsqrt[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    """
    Elementwise reciprocal squareroot of tensor.
        
    Parameters:
        dtype: The element type.
    
    Args:
        tensor: A tensor.
    
    Returns:
        A tensor equal to 1/tensor**(1/2).
    """
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.rsqrt](tensor)

fn sqrt[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    """
    Elementwise squareroot of tensor.
        
    Parameters:
        dtype: The element type.
    
    Args:
        tensor: A tensor.
    
    Returns:
        A tensor equal to tensor**(1/2).
    """
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.sqrt](tensor)

fn exp2[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    """
    Calculate elementwise two to the power of tensor[i].
        
    Parameters:
        dtype: The element type.
    
    Args:
        tensor: A tensor.
    
    Returns:
        A tensor with the shape of `tensor` with values equal to the
        2 to the power of the value in the original tensor at each position.
    """
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.exp2](tensor)

fn exp[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    """
    Calculate elementwise euler's constant(e) to the power of tensor[i].

    Parameters:
        dtype: The element type.
    
    Args:
        tensor: A tensor.
    
    Returns:
        A tensor with the shape of `tensor` with values equal to the
        e to the power of the value in the original tensor at each position.
    """
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.exp](tensor)

fn expm1[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    """
    Calculate elementwise euler's constant(e) to the power of tensor[i] minus1.

    Parameters:
        dtype: The element type.
    
    Args:
        tensor: A tensor.
    
    Returns:
        A tensor with the shape of `tensor` with values equal to the negative one plus
        e to the power of the value in the original tensor at each position.
    """
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.expm1](tensor)

fn scalb[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.scalb](tensor1, tensor2)
# ===------------------------------------------------------------------------===#
# Logarithms
# ===------------------------------------------------------------------------===#

fn log[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    """
    Elementwise natural logarithm of tensor.
        
    Parameters:
        dtype: The element type.
    
    Args:
        tensor: A tensor.
    
    Returns:
        A tensor equal to ln(tensor).
    """
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.log](tensor)

alias ln = log

fn log2[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    """
    Elementwise logarithm base two of tensor.
        
    Parameters:
        dtype: The element type.
    
    Args:
        tensor: A tensor.
    
    Returns:
        A tensor equal to log_2(tensor).
    """
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.log2](tensor)

fn log10[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    """
    Elementwise logarithm base ten of tensor.
        
    Parameters:
        dtype: The element type.
    
    Args:
        tensor: A tensor.
    
    Returns:
        A tensor equal to log_10(tensor).
    """
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.log10](tensor)

fn log1p[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    """
    Elementwise natural logarithm of 1 plus tensor.
        
    Parameters:
        dtype: The element type.
    
    Args:
        tensor: A tensor.
    
    Returns:
        A tensor equal to ln(tensor+1).
    """
    return _mf._math_func_1_tensor_in_one_tensor_out[dtype,math.log1p](tensor)

# ===------------------------------------------------------------------------===#
# Rounding and Similiar concepts
# ===------------------------------------------------------------------------===#

fn abs[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    """
    Elementwise absolute value of plus tensor.
        
    Parameters:
        dtype: The element type.
    
    Args:
        tensor: A tensor.
    
    Returns:
        A tensor equal to abs(tensor).
    """
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

fn nextafter[dtype:DType](tensor1:Tensor[dtype],tensor2:Tensor[dtype])raises->Tensor[dtype]:
    return _mf._math_func_2_tensor_in_one_tensor_out[dtype,math.nextafter](tensor1, tensor2)