"""
# ===----------------------------------------------------------------------=== #
# Implements Trigonometry functions
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #
"""

import math
import . _math_funcs as _mf
from .arithmetic import sqrt, fma

from ..core.ndarray import NDArray

# TODO: add in_dtype, out_dtype in backends and pass it here.
# ===------------------------------------------------------------------------===#
# Inverse Trig
# ===------------------------------------------------------------------------===#


fn acos[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply acos also known as inverse cosine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array.

    Returns:
        The elementwise acos of `array` in radians.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.acos](
        array
    )


fn asin[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply asin also known as inverse sine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array.

    Returns:
        The elementwise asin of `array` in radians.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.asin](
        array
    )


fn atan[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply atan also known as inverse tangent .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array.

    Returns:
        The elementwise atan of `array` in radians.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.atan](
        array
    )


fn atan2[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: NDArray[dtype], tensor2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply atan2 also known as inverse tangent.
    [atan2 wikipedia](https://en.wikipedia.org/wiki/Atan2).

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        tensor1: An Array.
        tensor2: An Array.

    Returns:
        The elementwise atan2 of `tensor1` and`tensor2` in radians.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[dtype, math.atan2](
        tensor1, tensor2
    )


# ===------------------------------------------------------------------------===#
# Trig
# ===------------------------------------------------------------------------===#


fn cos[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply cos also known as cosine.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array assumed to be in radian.

    Returns:
        The elementwise cos of `array`.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.cos](
        array
    )


fn sin[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply sin also known as sine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array assumed to be in radian.

    Returns:
        The elementwise sin of `array`.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.sin](
        array
    )


fn tan[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply tan also known as tangent .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array assumed to be in radian.

    Returns:
        The elementwise tan of `array`.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.tan](
        array
    )


fn hypot[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: NDArray[dtype], tensor2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply hypot also known as hypotenuse which finds the longest section of a right triangle
    given the other two sides.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        tensor1: An Array.
        tensor2: An Array.

    Returns:
        The elementwise hypotenuse of `tensor1` and`tensor2`.
    """
    return backend()._math_func_2_tensor_in_one_tensor_out[dtype, math.hypot](
        tensor1, tensor2
    )


fn hypot_fma[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](tensor1: NDArray[dtype], tensor2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply hypot also known as hypotenuse which finds the longest section of a right triangle
    given the other two sides.

    Constraints:
        Both tensors must have the same shapes.

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        tensor1: An Array.
        tensor2: An Array.

    Returns:
        The elementwise hypotenuse of `tensor1` and`tensor2`.
    """

    var tensor2_squared = fma[dtype, backend=backend](
        tensor2, tensor2, SIMD[dtype, 1](0)
    )
    return sqrt[dtype, backend=backend](
        fma[dtype, backend=backend](tensor1, tensor1, tensor2_squared)
    )


# ===------------------------------------------------------------------------===#
# Inverse Hyperbolic Trig
# ===------------------------------------------------------------------------===#


fn acosh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply acosh also known as inverse hyperbolic cosine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array.

    Returns:
        The elementwise acosh of `array` in radians.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.acosh](
        array
    )


fn asinh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply asinh also known as inverse hyperbolic sine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array.

    Returns:
        The elementwise asinh of `array` in radians.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.asinh](
        array
    )


fn atanh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply atanh also known as inverse hyperbolic tangent .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array.

    Returns:
        The elementwise atanh of `array` in radians.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.atanh](
        array
    )


# ===------------------------------------------------------------------------===#
# Hyperbolic Trig
# ===------------------------------------------------------------------------===#


fn cosh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply cosh also known as hyperbolic cosine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array assumed to be in radian.

    Returns:
        The elementwise cosh of `array`.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.cosh](
        array
    )


fn sinh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply sin also known as hyperbolic sine .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array assumed to be in radian.

    Returns:
        The elementwise sinh of `array`.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.sinh](
        array
    )


fn tanh[
    dtype: DType, backend: _mf.Backend = _mf.Vectorized
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply tan also known as hyperbolic tangent .

    Parameters:
        dtype: The element type.
        backend: Sets utility function origin, defualts to `Vectorized.

    Args:
        array: An Array assumed to be in radian.

    Returns:
        The elementwise tanh of `array`.
    """
    return backend()._math_func_1_tensor_in_one_tensor_out[dtype, math.tanh](
        array
    )
