"""
# ===----------------------------------------------------------------------=== #
# implements arithmetic functions
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #
"""

import math
import . _math_funcs as _mf
from ..core.ndarray import NDArray, NDArrayShape
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc

"""
TODO:
1) change dtype -> in_dtype and out_dtype
"""

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


fn diff[
    in_dtype: DType, out_dtype: DType = in_dtype
](array: NDArray[in_dtype], n: Int) raises -> NDArray[out_dtype]:
    """
    Compute the n-th order difference of the input array.

    Parameters:
        in_dtype: Input data type.
        out_dtype: Output data type, defaults to float32.

    Args:
        array: A array.
        n: The order of the difference.

    Returns:
        The n-th order difference of the input array.
    """

    var array1: NDArray[out_dtype] = NDArray[out_dtype](
        NDArrayShape(array.num_elements())
    )
    for i in range(array.num_elements()):
        array1[i] = array[i].cast[out_dtype]()

    for num in range(n):
        var result: NDArray[out_dtype] = NDArray[out_dtype](
            NDArrayShape(array.num_elements() - (num + 1))
        )
        for i in range(array1.num_elements() - 1):
            result[i] = (array1.load[1](i + 1) - array1.load[1](i)).cast[
                out_dtype
            ]()
        array1 = result
    return array1


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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Elementwise round NDArray to whole number.

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
](array: NDArray[dtype]) raises -> NDArray[dtype]:
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
