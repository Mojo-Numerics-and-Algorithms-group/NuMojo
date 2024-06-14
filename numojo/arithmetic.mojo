import math
import . _math_funcs as _mf
from tensor import Tensor
from algorithm import parallelize

from .utility_functions import *

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
    """
    Calculate elementwise scalb of tensor1 and tensor2.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        tensor1: A tensor.
        tensor2: A tensor.

    Returns:
        A tensor with the shape of `tensor1` and `tensor2` with values equal to the scalb of the value in the original tensor at each position.
    """
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


# ===------------------------------------------------------------------------===#
# Calculus
# ===------------------------------------------------------------------------===#


# naive loop implementation, optimize later
fn trapz[
    Idtype: DType, Fdtype: DType = DType.float32
](y: Tensor[Idtype], x: Tensor[Idtype]) raises -> SIMD[Fdtype, 1]:
    """
    Compute the integral of y over x using the trapezoidal rule.

    Parameters:
        Idtype: Input data type.
        Fdtype: Output data type, defaults to float32.

    Args:
        y: A tensor.
        x: A tensor.

    Constraints:
        `x` and `y` must have the same shape.
        `fdtype` must be a floating-point type if `idtype` is not a floating-point type.

    Returns:
        The integral of y over x using the trapezoidal rule.
    """
    if x.shape() != y.shape():
        raise Error("x and y must have the same shape")

    # move this check to compile time using constrained?
    if is_inttype[Idtype]() and not is_floattype[Fdtype]():
        raise Error(
            "output dtype `Fdtype` must be a floating-point type if input dtype"
            " `Idtype` is not a floating-point type"
        )

    var integral: Scalar[Fdtype] = 0.0
    for i in range(x.num_elements() - 1):
        var temp = (x[i + 1] - x[i]).cast[Fdtype]() * (y[i] + y[i + 1]).cast[
            Fdtype
        ]() / 2.0
        integral += temp
    return integral


fn diff[
    Idtype: DType, Fdtype: DType = Idtype
](tensor: Tensor[Idtype], n: Int) raises -> Tensor[Fdtype]:
    """
    Compute the n-th order difference of the input tensor.

    Parameters:
        Idtype: Input data type.
        Fdtype: Output data type, defaults to float32.

    Args:
        tensor: A tensor.
        n: The order of the difference.

    Returns:
        The n-th order difference of the input tensor.
    """

    var t1: Tensor[Fdtype] = Tensor[Fdtype](TensorShape(tensor.num_elements()))
    for i in range(tensor.num_elements()):
        t1[i] = tensor[i].cast[Fdtype]()

    for num in range(n):
        var result: Tensor[Fdtype] = Tensor[Fdtype](
            TensorShape(tensor.num_elements() - (num + 1))
        )
        for i in range(t1.num_elements() - 1):
            result[i] = (t1.load[1](i + 1) - t1.load[1](i)).cast[Fdtype]()
        t1 = result
    return t1


# Implement it for (2,) tensor and add axis parameters later
fn cross[
    Idtype: DType, Fdtype: DType = DType.float32
](tensor1: Tensor[Idtype], tensor2: Tensor[Idtype]) raises -> Tensor[Fdtype]:
    """
    Compute the cross product of two tensors.

    Parameters:
        Idtype: Input data type.
        Fdtype: Output data type, defaults to float32.

    Args:
        tensor1: A tensor.
        tensor2: A tensor.

    Constraints:
        `tensor1` and `tensor2` must be of shape (3,).

    Returns:
        The cross product of two tensors.
    """

    if tensor1.shape() == tensor2.shape() == 3:
        var tensor3: Tensor[Fdtype] = Tensor[Fdtype](TensorShape(3))
        tensor3[0] = (tensor1[1] * tensor2[2] - tensor1[2] * tensor2[1]).cast[Fdtype]()
        tensor3[1] = (tensor1[2] * tensor2[0] - tensor1[0] * tensor2[2]).cast[Fdtype]()
        tensor3[2] = (tensor1[0] * tensor2[1] - tensor1[1] * tensor2[0]).cast[Fdtype]()
        return tensor3
    else:
        raise Error(
            "Cross product is not supported for tensors of shape "
            + tensor1.shape().__str__()
            + " and "
            + tensor2.shape().__str__()
        )
