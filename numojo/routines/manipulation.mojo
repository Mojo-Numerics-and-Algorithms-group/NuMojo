"""
Array manipulation routines.
"""
# ===----------------------------------------------------------------------=== #
# ARRAY MANIPULATION ROUTINES
# Last updated: 2024-08-03
# ===----------------------------------------------------------------------=== #

from sys import simdwidthof
from algorithm import vectorize
from numojo.core.ndshape import NDArrayShape, Shape
from numojo.core.ndstrides import NDArrayStrides


fn copyto():
    pass


fn ndim[dtype: DType](array: NDArray[dtype]) -> Int:
    """
    Returns the number of dimensions of the NDArray.

    Args:
        array: A NDArray.

    Returns:
        The number of dimensions of the NDArray.
    """
    return array.ndim


fn shape[dtype: DType](array: NDArray[dtype]) -> NDArrayShape:
    """
    Returns the shape of the NDArray.

    Args:
        array: A NDArray.

    Returns:
        The shape of the NDArray.
    """
    return array.shape


fn size[dtype: DType](array: NDArray[dtype], axis: Int) raises -> Int:
    """
    Returns the size of the NDArray.

    Args:
        array: A NDArray.
        axis: The axis to get the size of.

    Returns:
        The size of the NDArray.
    """
    return array.shape[axis]


fn reshape[
    dtype: DType
](
    inout array: NDArray[dtype], shape: VariadicList[Int], order: String = "C"
) raises:
    """
        Reshapes the NDArray to given Shape.

    Raises:
        Error: If the number of elements do not match.

    Args:
        array: A NDArray.
        shape: Variadic integers of shape.
        order: Order of the array - Row major `C` or Column major `F`.

    """
    var num_elements_new: Int = 1
    var ndim_new: Int = 0
    for i in shape:
        num_elements_new *= i
        ndim_new += 1

    if array.size != num_elements_new:
        raise Error("Cannot reshape: Number of elements do not match.")

    var shape_new: List[Int] = List[Int]()
    for i in range(ndim_new):
        shape_new.append(shape[i])
        var temp: Int = 1
        for j in range(i + 1, ndim_new):  # temp
            temp *= shape[j]

    array.ndim = ndim_new
    array.shape = NDArrayShape(shape=shape_new)
    array.strides = NDArrayStrides(shape=shape_new, order=order)
    array.order = order


fn ravel[dtype: DType](inout array: NDArray[dtype], order: String = "C") raises:
    """
    Returns the raveled version of the NDArray.
    """
    if array.ndim == 1:
        print("Array is already 1D")
        return
    else:
        if order == "C":
            reshape[dtype](array, array.size, order="C")
        else:
            reshape[dtype](array, array.size, order="F")


fn where[
    dtype: DType
](
    inout x: NDArray[dtype], scalar: SIMD[dtype, 1], mask: NDArray[DType.bool]
) raises:
    """
    Replaces elements in `x` with `scalar` where `mask` is True.

    Parameters:
        dtype: DType.

    Args:
        x: A NDArray.
        scalar: A SIMD value.
        mask: A NDArray.

    """
    for i in range(x.size):
        if mask._buf[i] == True:
            x._buf.store(i, scalar)


# TODO: do it with vectorization
fn where[
    dtype: DType
](inout x: NDArray[dtype], y: NDArray[dtype], mask: NDArray[DType.bool]) raises:
    """
    Replaces elements in `x` with elements from `y` where `mask` is True.

    Raises:
        ShapeMismatchError: If the shapes of `x` and `y` do not match.

    Parameters:
        dtype: DType.

    Args:
        x: NDArray[dtype].
        y: NDArray[dtype].
        mask: NDArray[DType.bool].

    """
    if x.shape != y.shape:
        raise Error("Shape mismatch error: x and y must have the same shape")
    for i in range(x.size):
        if mask._buf[i] == True:
            x._buf.store(i, y._buf[i])


fn flip[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Flips the NDArray along the given axis.

    Parameters:
        dtype: DType.

    Args:
        array: A NDArray.

    Returns:
        The flipped NDArray.
    """
    if array.ndim != 1:
        raise Error("Flip is only supported for 1D arrays")

    var result: NDArray[dtype] = NDArray[dtype](
        shape=array.shape, order=array.order
    )
    for i in range(array.size):
        result._buf.store(i, array._buf[array.size - i - 1])
    return result


fn flatten[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Flattens the NDArray.

    Parameters:
        dtype: Dataype of the NDArray elements.

    Args:
        array: A NDArray.

    Returns:
        The 1 dimensional flattened NDArray.
    """

    var res: NDArray[dtype] = NDArray[dtype](Shape(array.size))
    alias width: Int = simdwidthof[dtype]()

    @parameter
    fn vectorized_flatten[simd_width: Int](index: Int) -> None:
        res._buf.store(index, array._buf.load[width=simd_width](index))

    vectorize[vectorized_flatten, width](array.size)
    return res
