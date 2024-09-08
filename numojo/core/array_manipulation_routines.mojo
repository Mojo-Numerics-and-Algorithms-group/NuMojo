"""
Array manipulation routines.
"""
# ===----------------------------------------------------------------------=== #
# ARRAY MANIPULATION ROUTINES
# Last updated: 2024-08-03
# ===----------------------------------------------------------------------=== #


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
    return array.ndshape


fn size[dtype: DType](array: NDArray[dtype], axis: Int) raises -> Int:
    """
    Returns the size of the NDArray.

    Args:
        array: A NDArray.
        axis: The axis to get the size of.

    Returns:
        The size of the NDArray.
    """
    return array.ndshape[axis]


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

    if array.ndshape.ndsize != num_elements_new:
        raise Error("Cannot reshape: Number of elements do not match.")

    var shape_new: List[Int] = List[Int]()
    for i in range(ndim_new):
        shape_new.append(shape[i])
        var temp: Int = 1
        for j in range(i + 1, ndim_new):  # temp
            temp *= shape[j]

    array.ndim = ndim_new
    array.ndshape = NDArrayShape(shape=shape_new)
    array.stride = NDArrayStride(shape=shape_new, order=order)
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
            reshape[dtype](array, array.ndshape.ndsize, order="C")
        else:
            reshape[dtype](array, array.ndshape.ndsize, order="F")


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
    for i in range(x.ndshape.ndsize):
        if mask.data[i] == True:
            x.data.store(i, scalar)


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
    if x.ndshape != y.ndshape:
        raise Error("Shape mismatch error: x and y must have the same shape")
    for i in range(x.ndshape.ndsize):
        if mask.data[i] == True:
            x.data.store(i, y.data[i])


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
        shape=array.ndshape, order=array.order
    )
    for i in range(array.ndshape.ndsize):
        result.data.store(i, array.data[array.ndshape.ndsize - i - 1])
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

    var res: NDArray[dtype] = NDArray[dtype](array.ndshape.ndsize)
    alias width: Int = simdwidthof[dtype]()

    @parameter
    fn vectorized_flatten[simd_width: Int](index: Int) -> None:
        res.data.store[width=simd_width](
            index, array.data.load[width=simd_width](index)
        )

    vectorize[vectorized_flatten, width](array.ndshape.ndsize)
    return res


# TODO: implement for arbitrary axis
fn trace[
    dtype: DType
](
    array: NDArray[dtype], offset: Int = 0, axis1: Int = 0, axis2: Int = 1
) raises -> NDArray[dtype]:
    """
    Computes the trace of a ndarray.

    Parameters:
        dtype: Data type of the array.

    Args:
        array: A NDArray.
        offset: Offset of the diagonal from the main diagonal.
        axis1: First axis.
        axis2: Second axis.

    Returns:
        The trace of the NDArray.
    """
    if array.ndim != 2:
        raise Error("Trace is currently only supported for 2D arrays")
    if axis1 > array.ndim - 1 or axis2 > array.ndim - 1:
        raise Error("axis cannot be greater than the rank of the array")
    var result: NDArray[dtype] = NDArray[dtype](1)
    var rows = array.ndshape[0]
    var cols = array.ndshape[1]
    var diag_length = min(rows, cols - offset) if offset >= 0 else min(
        rows + offset, cols
    )

    for i in range(diag_length):
        var row = i if offset >= 0 else i - offset
        var col = i + offset if offset >= 0 else i
        result.data.store(0, result.data.load(0) + array.data[row * cols + col])

    return result
