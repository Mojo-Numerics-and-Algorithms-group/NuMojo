"""
# ===----------------------------------------------------------------------=== #
# implements arithmetic functions
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #
"""

import math
import . _math_funcs as _mf
from .ndarray import NDArray, NDArrayShape, _get_index

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
    Indtype: DType, Outdtype: DType = DType.float32
](y: NDArray[Indtype], x: NDArray[Indtype]) raises -> SIMD[Outdtype, 1]:
    """
    Compute the integral of y over x using the trapezoidal rule.

    Parameters:
        Indtype: Input data type.
        Outdtype: Output data type, defaults to float32.

    Args:
        y: An array.
        x: An array.

    Constraints:
        `x` and `y` must have the same shape.
        `fdtype` must be a floating-point type if `idtype` is not a floating-point type.

    Returns:
        The integral of y over x using the trapezoidal rule.
    """
    if x.shape() != y.shape():
        raise Error("x and y must have the same shape")

    # move this check to compile time using constrained?
    if is_inttype[Indtype]() and not is_floattype[Outdtype]():
        raise Error(
            "output dtype `Fdtype` must be a floating-point type if input dtype"
            " `Idtype` is not a floating-point type"
        )

    var integral: SIMD[Outdtype] = 0.0
    for i in range(x.num_elements() - 1):
        var temp = (x[i + 1] - x[i]).cast[Outdtype]() * (y[i] + y[i + 1]).cast[
            Outdtype
        ]() / 2.0
        integral += temp
    return integral


fn diff[
    Indtype: DType, Outdtype: DType = Indtype
](array: NDArray[Indtype], n: Int) raises -> NDArray[Outdtype]:
    """
    Compute the n-th order difference of the input array.

    Parameters:
        Indtype: Input data type.
        Outdtype: Output data type, defaults to float32.

    Args:
        array: A array.
        n: The order of the difference.

    Returns:
        The n-th order difference of the input array.
    """

    var array1: NDArray[Outdtype] = NDArray[Outdtype](
        NDArrayShape(array.num_elements())
    )
    for i in range(array.num_elements()):
        array1[i] = array[i].cast[Outdtype]()

    for num in range(n):
        var result: NDArray[Outdtype] = NDArray[Outdtype](
            NDArrayShape(array.num_elements() - (num + 1))
        )
        for i in range(array1.num_elements() - 1):
            result[i] = (array1.load[1](i + 1) - array1.load[1](i)).cast[
                Outdtype
            ]()
        array1 = result
    return array1


# Implement it for (2,) array and add axis parameters later
fn cross[
    Indtype: DType, Outdtype: DType = DType.float32
](array1: NDArray[Indtype], array2: NDArray[Indtype]) raises -> NDArray[
    Outdtype
]:
    """
    Compute the cross product of two tensors.

    Parameters
        Indtype: Input data type.
        Outdtype: Output data type, defaults to float32.

    Args:
        array1: A array.
        array2: A array.

    Constraints:
        `array1` and `array2` must be of shape (3,).

    Returns:
        The cross product of two tensors.
    """

    if array1.shape() == array2.shape() == 3:
        var array3: NDArray[Outdtype] = NDArray[Outdtype](NDArrayShape(3))
        array3[0] = (array1[1] * array2[2] - array1[2] * array2[1]).cast[
            Outdtype
        ]()
        array3[1] = (array1[2] * array2[0] - array1[0] * array2[2]).cast[
            Outdtype
        ]()
        array3[2] = (array1[0] * array2[1] - array1[1] * array2[0]).cast[
            Outdtype
        ]()
        return array3
    else:
        raise Error(
            "Cross product is not supported for tensors of shape "
            + array1.shape().__str__()
            + " and "
            + array2.shape().__str__()
        )



# fn matmul[
#     dtype: DType
# ](array1: NDArray[dtype], array2: NDArray[dtype]) -> NDArray[dtype]:

#     var result = NDArray[dtype](array1.shape(), 0.0)
#     alias opt_nelts = simdwidthof[dtype]()

#     @parameter
#     fn calc_row(m: Int):
#         for k in range(self.info.shape[1]):
#             @parameter
#             fn dot[nelts : Int](n : Int):
#                 result.store[nelts](m, n, val=result.load[nelts](m,n) 
#                     + self.load[nelts](m,k) * other.load[nelts](k,n))
#             vectorize[dot, opt_nelts](other.info.shape[1])
#     parallelize[calc_row](self.info.shape[0], self.info.shape[0])
#     return result


from algorithm import Static2DTileUnitFunc as Tile2DFunc

# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int) raises:
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)
            
# fn matmul[dtype: DType](C: NDArray[dtype], A: NDArray[dtype], B: NDArray[dtype]) raises:
#     alias nelts = simdwidthof[dtype]()
#     @parameter
#     fn calc_row(m: Int):
#         @parameter
#         fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
#             for k in range(y, y + tile_y):
#                 @parameter
#                 fn dot[nelts: Int](n: Int):
#                     try:
#                         C.store(m, n + x, val=C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x))
#                     except:
#                         print("lol")
#                 # Vectorize by nelts and unroll by tile_x/nelts
#                 # Here unroll factor is 4
#                 alias unroll_factor = tile_x // nelts
#                 vectorize[dot, nelts, size=tile_x, unroll_factor=unroll_factor]()

#         alias tile_size = 4
#         tile[calc_tile, nelts * tile_size, tile_size](A.info.shape[1], C.info.shape[1])

#     parallelize[calc_row](C.info.shape[1], C.info.shape[0])

fn matmul[
    dtype: DType
](A: NDArray[dtype], B: NDArray[dtype]) raises -> NDArray[dtype]:
    alias nelts = simdwidthof[dtype]()

    var C: NDArray[dtype] = NDArray[dtype](A.info.shape[0], B.info.shape[1])
    print(C.info.shape[0], "x", C.info.shape[1])
    # @parameter
    # fn calc_row(m: Int):
    #     for k in range(2048):
    #         @parameter
    #         fn dot[nelts : Int](n : Int):
    #             C.store[nelts](m,n, val= C.load[nelts](m,n) + A.load(m,k) * B.load[nelts](k,n))
    #         vectorize[dot, nelts](512)
    # parallelize[calc_row](1024, 1024)

    # var C: NDArray[dtype] = NDArray[dtype](A.info.shape[0], B.info.shape[1])
    for m in range(5):
        for k in range(5):
            @parameter
            fn dot[nelts: Int](n: Int):
                C.store(m,n,
                    val=C.load[nelts](m,n)
                    + A.load(m,k) 
                    * B.load[nelts](k,n))
            vectorize[dot, nelts](5)

    return C

fn matmul_naive[dtype: DType](inout A: NDArray[dtype], inout B: NDArray[dtype]) raises -> NDArray[dtype]:

    var C: NDArray[dtype] = NDArray[dtype](5,5,random=False)

    for m in range(10):
        print(m)
        for k in range(10):
            for n in range(10):
                C.store(_get_index(m,n,weights=C.info.strides), 
                val=C.load[1](m,n) 
                + A.load[1](m,k)  
                * B.load[1](k,n))

    print("lol")
    # for m in range(C.info.shape[0]):
    #     for k in range(A.info.shape[1]):
    #         print(k)
    #         for n in range(C.info.shape[1]):
    #             C.store(_get_index(m,n,weights=C.info.strides), 
    #             val=C.load[1](_get_index(m,n,weights=C.info.strides)) 
    #             + A.load[1](_get_index(m,k,weights=A.info.strides))  
    #             * B.load[1](_get_index(k,n,weights=B.info.strides)))

    # for m in range(C.info.shape[0]):
    # for k in range(A.info.shape[1]):
    #     for n in range(C.info.shape[1]):
    #         C.__setitem__(List[Int](m,n), val=C.__getitem__(m,n) + A.__getitem__(m,k) * B.__getitem__(k,n))

    return C