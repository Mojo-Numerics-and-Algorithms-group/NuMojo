"""
# ===----------------------------------------------------------------------=== #
# implements arithmetic functions
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #
"""

import math
import . _math_funcs as _mf

# ===------------------------------------------------------------------------===#
# Addition/Subtraction
# ===------------------------------------------------------------------------===#


fn add[
    dtype: DType,
    backend: _mf.Backend = _mf.Vectorized,
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on two tensors.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        The elementwise sum of `array1` and`array2`.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[dtype, SIMD.__add__](
        array1, array2
    )


fn sub[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on two tensors.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        The elementwise difference of `array1` and`array2`.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[dtype, SIMD.__sub__](
        array1, array2
    )


# ===------------------------------------------------------------------------===#
# Multiplication/Division
# ===------------------------------------------------------------------------===#


fn copysign[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Copy the sign of the first NDArray and apply it to the second NDArray.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        The second NDArray multipied by the sign of the first NDArray.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[
        dtype, math.copysign
    ](array1, array2)


fn mod[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise modulo of array1 and array2.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1%array2.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[dtype, SIMD.__mod__](
        array1, array2
    )


fn mul[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise product of array1 and array2.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1*array2.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[dtype, SIMD.__mul__](
        array1, array2
    )


fn div[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise quotent of array1 and array2.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1/array2.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[
        dtype, SIMD.__truediv__
    ](array1, array2)


fn fma[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](
    array1: NDArray[dtype], array2: NDArray[dtype], array3: NDArray[dtype]
) raises -> NDArray[dtype]:
    """
    Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray.

    Constraints:
        Both tensors must have the same shape.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.
        array3: A NDArray.

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """
    return backend()._math_func_fma(array1, array2, array3)


fn fma[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](
    array1: NDArray[dtype], array2: NDArray[dtype], simd: SIMD[dtype, 1]
) raises -> NDArray[dtype]:
    """
    Apply a SIMD level fuse multipy add function of three variables and one return to a NDArray.

    Constraints:
        Both tensors must have the same shape

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.
        simd: A SIMD[dtype,1] value to be added.

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """
    return backend()._math_func_fma(array1, array2, simd)


fn remainder[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise remainders of NDArray.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1//array2.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[
        dtype, math.remainder
    ](array1, array2)


# fn reciprocal[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](NDArray: NDArray[dtype]) -> NDArray[dtype]:
#     """
#     Elementwise reciprocals of array1 and array2.

#     Constraints:
#         Both tensors must have the same shapes.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defualts to `Vectorized`.

#     Args:
#         NDArray: A NDArray.

#     Returns:
#         A NDArray equal to 1/NDArray.
#     """
#     return backend()._math_func_1_tensor_in_one_tensor_out[
#         dtype, math.reciprocal
#     ](NDArray)


# ===------------------------------------------------------------------------===#
# Exponents/Roots
# ===------------------------------------------------------------------------===#
fn cbrt[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Elementwise cuberoot of NDArray.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to NDArray**(1/3).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.cbrt](
        array
    )


# fn pow[dtype: DType,
#     backend: _mf.Backend = _mf.Vectorized](array1: NDArray[dtype], intval: Int) -> NDArray[dtype]:
#     """
#     Elementwise NDArray to the power of intval.

#     Constraints:
#         Both tensors must have the same shapes.

#     Parameters:
#         dtype: The element type.
#         backend: Sets utility function origin, defualts to `Vectorized`.

#     Args:
#         array1: A NDArray.
#         intval: An integer.

#     Returns:
#         A NDArray equal to NDArray**intval.
#     """
#     return backend()._math_func_simd_int[dtype, math.pow](array1, intval)


fn rsqrt[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Elementwise reciprocal squareroot of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to 1/NDArray**(1/2).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.rsqrt](
        array
    )


fn sqrt[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Elementwise squareroot of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to NDArray**(1/2).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.sqrt](
        array
    )


fn exp2[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Calculate elementwise two to the power of NDArray[i].

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray with the shape of `NDArray` with values equal to the
        2 to the power of the value in the original NDArray at each position.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.exp2](
        array
    )


fn exp[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Calculate elementwise euler's constant(e) to the power of NDArray[i].

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray with the shape of `NDArray` with values equal to the
        e to the power of the value in the original NDArray at each position.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.exp](
        array
    )


fn expm1[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Calculate elementwise euler's constant(e) to the power of NDArray[i] minus1.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray with the shape of `NDArray` with values equal to the negative one plus
        e to the power of the value in the original NDArray at each position.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.expm1](
        array
    )


# this is a temporary doc, write a more explanatory one
fn scalb[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Calculate the scalb of array1 and array2.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray with the shape of `NDArray` with values equal to the negative one plus
        e to the power of the value in the original NDArray at each position.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[dtype, math.scalb](
        array1, array2
    )


# ===------------------------------------------------------------------------===#
# Logarithms
# ===------------------------------------------------------------------------===#


fn log[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Elementwise natural logarithm of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to ln(NDArray).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.log](
        array
    )


alias ln = log


fn log2[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Elementwise logarithm base two of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to log_2(NDArray).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.log2](
        array
    )


fn log10[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Elementwise logarithm base ten of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to log_10(NDArray).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.log10](
        array
    )


fn log1p[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Elementwise natural logarithm of 1 plus NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to ln(NDArray+1).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.log1p](
        array
    )


# ===------------------------------------------------------------------------===#
# Rounding and Similiar concepts
# ===------------------------------------------------------------------------===#


fn tabs[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Elementwise absolute value of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to abs(NDArray).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, SIMD.__abs__](
        array
    )


fn tfloor[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Elementwise round down to nearest whole number of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to floor(NDArray).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[
        dtype, SIMD.__floor__
    ](array)


fn tceil[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Elementwise round up to nearest whole number of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to ceil(NDArray).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[
        dtype, SIMD.__ceil__
    ](array)


fn ttrunc[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Elementwise remove decimal value from float whole number of NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to trunc(NDArray).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[
        dtype, SIMD.__trunc__
    ](array)


fn tround[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Elementwise  NDArray.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: A NDArray.

    Returns:
        A NDArray equal to trunc(NDArray).
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[
        dtype, SIMD.__round__
    ](array)


fn roundeven[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) -> NDArray[dtype]:
    """
    Performs elementwise banker's rounding on the elements of a NDArray.

    Parameters:
        dtype: The dtype of the input and output Tensor.
        backend: Sets utility function origin, defualts to `Vectorized`.

    Args:
        array: Tensor to perform rounding on.

    Returns:
    The elementwise banker's rounding of NDArray.

    This rounding goes to the nearest integer with ties toward the nearest even integer.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[
        dtype, SIMD.roundeven
    ](array)


# fn round_half_down[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](NDArray: NDArray[dtype]) -> NDArray[dtype]:
#     """
#     Rounds ties towards the smaller integer.

#     Parameters:
#         dtype: The dtype of the input and output Tensor.
#         backend: Sets utility function origin, defualts to `Vectorized`.

#     Args:
#         NDArray: Tensor to perform rounding on.

#     Returns:
#     The elementwise rounding of x evaluating ties towards the smaller integer.
#     """
#     return backend()._math_func_1_tensor_in_one_tensor_out[
#         dtype, SIMD.__round_half_down
#     ](NDArray)


# fn round_half_up[
#     dtype: DType, backend: _mf.Backend = _mf.Vectorized
# ](NDArray: NDArray[dtype]) -> NDArray[dtype]:
#     """
#     Rounds ties towards the larger integer.

#     Parameters:
#         dtype: The dtype of the input and output Tensor.
#         backend: Sets utility function origin, defualts to `Vectorized`.

#     Args:
#         NDArray: Tensor to perform rounding on.

#     Returns:
#     The elementwise rounding of x evaluating ties towards the larger integer.
#     """
#     return backend()._math_func_1_tensor_in_one_tensor_out[
#         dtype, math.round_half_up
#     ](NDArray)


fn nextafter[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Computes the nextafter of the inputs.

    Parameters:
        dtype: The dtype of the input and output Tensor. Constraints: must be a floating-point type.
        backend: Sets utility function origin, defualts to `Vectorized`.


    Args:
        array1: The first input argument.
        array2: The second input argument.

    Returns:
    The nextafter of the inputs.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[
        dtype, math.nextafter
    ](array1, array2)


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
