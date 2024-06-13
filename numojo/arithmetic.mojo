import math
import . _math_funcs as _mf
from tensor import Tensor

"""
implements arithmetic functions
"""
# ===------------------------------------------------------------------------===#
# Addition/Subtraction
# ===------------------------------------------------------------------------===#


fn add[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Perform addition on two tensors.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor1: A tensor.
        tensor2: A tensor.

    Returns:
        The elementwise sum of `tensor1` and`tensor2`.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[dtype, SIMD.__add__](
        tensor1, tensor2
    )


fn sub[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Perform subtraction on two tensors.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor1: A tensor.
        tensor2: A tensor.

    Returns:
        The elementwise difference of `tensor1` and`tensor2`.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[dtype, SIMD.__sub__](
        tensor1, tensor2
    )


# ===------------------------------------------------------------------------===#
# Multiplication/Division
# ===------------------------------------------------------------------------===#


fn copysign[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Copy the sign of the first tensor and apply it to the second tensor.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor1: A tensor.
        tensor2: A tensor.

    Returns:
        The second tensor multipied by the sign of the first tensor.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[
        dtype, math.copysign
    ](tensor1, tensor2)


fn mod[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Elementwise modulo of tensor1 and tensor2.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor1: A tensor.
        tensor2: A tensor.

    Returns:
        A tensor equal to tensor1%tensor2.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[dtype, SIMD.__mod__](
        tensor1, tensor2
    )


fn mul[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Elementwise product of tensor1 and tensor2.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor1: A tensor.
        tensor2: A tensor.

    Returns:
        A tensor equal to tensor1*tensor2.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[dtype, SIMD.__mul__](
        tensor1, tensor2
    )


fn div[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Elementwise quotent of tensor1 and tensor2.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor1: A tensor.
        tensor2: A tensor.

    Returns:
        A tensor equal to tensor1/tensor2.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[
        dtype, SIMD.__truediv__
    ](tensor1, tensor2)


fn fma[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](
    tensor1: Tensor[dtype], tensor2: Tensor[dtype], tensor3: Tensor[dtype]
) raises -> Tensor[dtype]:
    """
    Apply a SIMD level fuse multipy add function of three variables and one return to a tensor.

    Constraints:
        Both tensors must have the same shape.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor1: A tensor.
        tensor2: A tensor.
        tensor3: A tensor.

    Returns:
        A a new tensor that is tensor with the function func applied.
    """
    return backend()._math_func_fma(tensor1, tensor2, tensor3)


fn fma[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](
    tensor1: Tensor[dtype], tensor2: Tensor[dtype], simd: SIMD[dtype, 1]
) raises -> Tensor[dtype]:
    """
    Apply a SIMD level fuse multipy add function of three variables and one return to a tensor.

    Constraints:
        Both tensors must have the same shape

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor1: A tensor.
        tensor2: A tensor.
        simd: A SIMD[dtype,1] value to be added.

    Returns:
        A a new tensor that is tensor with the function func applied.
    """
    return backend()._math_func_fma(tensor1, tensor2, simd)


fn remainder[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Elementwise remainders of tensor.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor1: A tensor.
        tensor2: A tensor.

    Returns:
        A tensor equal to tensor1//tensor2.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[
        dtype, math.remainder
    ](tensor1, tensor2)


# fn reciprocal[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](tensor: Tensor[dtype]) -> Tensor[dtype]:
#     """
#     Elementwise reciprocals of tensor1 and tensor2.

#     Constraints:
#         Both tensors must have the same shapes.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defualts to `Vectorized`.

#     Args:
#         tensor: A tensor.

#     Returns:
#         A tensor equal to 1/tensor.
#     """
#     return backend()._math_func_1_tensor_in_one_tensor_out[
#         dtype, math.reciprocal
#     ](tensor)


# ===------------------------------------------------------------------------===#
# Exponents/Roots
# ===------------------------------------------------------------------------===#
fn cbrt[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Elementwise cuberoot of tensor.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor equal to tensor**(1/3).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.cbrt](
        tensor
    )


# fn pow[dtype: DType,
#     backend: _mf.Backend = _mf.Vectorized](tensor1: Tensor[dtype], intval: Int) -> Tensor[dtype]:
#     """
#     Elementwise tensor to the power of intval.

#     Constraints:
#         Both tensors must have the same shapes.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defualts to `Vectorized`.

#     Args:
#         tensor1: A tensor.
#         intval: An integer.

#     Returns:
#         A tensor equal to tensor**intval.
#     """
#     return backend()._math_func_simd_int[dtype, math.pow](tensor1, intval)


fn rsqrt[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Elementwise reciprocal squareroot of tensor.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor equal to 1/tensor**(1/2).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.rsqrt](
        tensor
    )


fn sqrt[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Elementwise squareroot of tensor.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor equal to tensor**(1/2).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.sqrt](
        tensor
    )


fn exp2[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Calculate elementwise two to the power of tensor[i].

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor with the shape of `tensor` with values equal to the
        2 to the power of the value in the original tensor at each position.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.exp2](
        tensor
    )


fn exp[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Calculate elementwise euler's constant(e) to the power of tensor[i].

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor with the shape of `tensor` with values equal to the
        e to the power of the value in the original tensor at each position.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.exp](
        tensor
    )


fn expm1[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Calculate elementwise euler's constant(e) to the power of tensor[i] minus1.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor with the shape of `tensor` with values equal to the negative one plus
        e to the power of the value in the original tensor at each position.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.expm1](
        tensor
    )


fn scalb[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
    return backend()._math_func_2_tensor_in_one_tensor_out[dtype, math.scalb](
        tensor1, tensor2
    )


# ===------------------------------------------------------------------------===#
# Logarithms
# ===------------------------------------------------------------------------===#


fn log[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Elementwise natural logarithm of tensor.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor equal to ln(tensor).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.log](
        tensor
    )


alias ln = log


fn log2[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Elementwise logarithm base two of tensor.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor equal to log_2(tensor).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.log2](
        tensor
    )


fn log10[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Elementwise logarithm base ten of tensor.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor equal to log_10(tensor).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.log10](
        tensor
    )


fn log1p[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Elementwise natural logarithm of 1 plus tensor.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor equal to ln(tensor+1).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.log1p](
        tensor
    )


# ===------------------------------------------------------------------------===#
# Rounding and Similiar concepts
# ===------------------------------------------------------------------------===#


fn tabs[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Elementwise absolute value of tensor.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor equal to abs(tensor).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, SIMD.__abs__](
        tensor
    )


fn tfloor[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Elementwise round down to nearest whole number of tensor.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor equal to floor(tensor).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[
        dtype, SIMD.__floor__
    ](tensor)


fn tceil[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Elementwise round up to nearest whole number of tensor.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor equal to ceil(tensor).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[
        dtype, SIMD.__ceil__
    ](tensor)


fn ttrunc[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Elementwise remove decimal value from float whole number of tensor.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor equal to trunc(tensor).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[
        dtype, SIMD.__trunc__
    ](tensor)


fn tround[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Elementwise  tensor.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: A tensor.

    Returns:
        A tensor equal to trunc(tensor).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[
        dtype, SIMD.__round__
    ](tensor)


fn roundeven[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """
    Performs elementwise banker's rounding on the elements of a tensor.

    Parameters:
        dtype: The dtype of the input and output Tensor.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor: Tensor to perform rounding on.

    Returns:
    The elementwise banker's rounding of tensor.

    This rounding goes to the nearest integer with ties toward the nearest even integer.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[
        dtype, SIMD.roundeven
    ](tensor)


# fn round_half_down[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](tensor: Tensor[dtype]) -> Tensor[dtype]:
#     """
#     Rounds ties towards the smaller integer.

#     Parameters:
#         dtype: The dtype of the input and output Tensor.
#         backend: Sets utility function origin, defualts to `Vectorized`.

#     Args:
#         tensor: Tensor to perform rounding on.

#     Returns:
#     The elementwise rounding of x evaluating ties towards the smaller integer.
#     """
#     return backend()._math_func_1_tensor_in_one_tensor_out[
#         dtype, SIMD.__round_half_down
#     ](tensor)


# fn round_half_up[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](tensor: Tensor[dtype]) -> Tensor[dtype]:
#     """
#     Rounds ties towards the larger integer.

#     Parameters:
#         dtype: The dtype of the input and output Tensor.
#         backend: Sets utility function origin, defualts to `Vectorized`.

#     Args:
#         tensor: Tensor to perform rounding on.

#     Returns:
#     The elementwise rounding of x evaluating ties towards the larger integer.
#     """
#     return backend()._math_func_1_tensor_in_one_tensor_out[
#         dtype, math.round_half_up
#     ](tensor)


fn nextafter[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Computes the nextafter of the inputs.

    Parameters:
        dtype: The dtype of the input and output Tensor. Constraints: must be a floating-point type.
        backend: Sets utility function origin, defualts to `Vectorized`.


    Args:
        tensor1: The first input argument.
        tensor2: The second input argument.

    Returns:
    The nextafter of the inputs.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[
        dtype, math.nextafter
    ](tensor1, tensor2)
