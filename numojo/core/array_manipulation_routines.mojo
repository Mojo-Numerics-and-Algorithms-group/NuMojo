"""
# ===----------------------------------------------------------------------=== #
# ARRAY MANIPULATION ROUTINES
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #
"""


fn copyto():
    pass

fn ndim[dtype](array: NDArray[dtype]) -> Int:
    """
    Returns the number of dimensions of the NDArray.

    Args:
        array: A NDArray.

    Returns:
        The number of dimensions of the NDArray.
    """
    return array.ndim


fn shape[dtype](array: NDArray[dtype]) -> NDArrayShape:
    """
    Returns the shape of the NDArray.

    Args:
        array: A NDArray.

    Returns:
        The shape of the NDArray.
    """
    return array.ndshape

fn size[dtype](array: NDArray[dtype], axis: Int) -> Int:
    """
    Returns the size of the NDArray.

    Args:
        array: A NDArray.

    Returns:
        The size of the NDArray.
    """
    else:
        return array.ndshape[axis]

fn moveaxis[dtype: DType](inout array: NDArray[dtype], owned source: Int, owned destination: Int) raises:
    """
    Moves the axis from source to destination.

    Raises:
        Error: If the axis is out of bounds.

    Args:
        array: A NDArray.
        source: The source axis.
        destination: The destination axis.

    Returns:
        A NDArray with the axis moved from source to destination.
    """
    if source < 0:
        source += array.ndim
    if destination < 0:
        destination += array.ndim
    if source >= array.ndim or destination >= array.ndim:
        raise Error("Axis out of bounds")

    var new_shape = List[Int]()
    for i in range(array.ndim):
        if i == source:
            new_shape.append(array.ndshape[destination])
        elif i == destination:
            new_shape.append(array.ndshape[source])
        else:
            new_shape.append(array.ndshape[i])
        
    array.ndshape = NDArrayShape(shape=new_shape)
    array.stride = NDArrayStride(shape=new_shape, order=array.order)

fn reshape[dtype: DType](inout array: NDArray[dtype], *Shape: Int, order: String = "C") raises:
    """
        Reshapes the NDArray to given Shape.

        Raises:
            Error: If the number of elements do not match.

        Args:
            array: A NDArray.
            Shape: Variadic integers of shape.
            order: Order of the array - Row major `C` or Column major `F`.
        
        Returns:
            A reshaped NDArray.
    """
    var num_elements_new: Int = 1
    var ndim_new: Int = 0
    for i in Shape:
        num_elements_new *= i
        ndim_new += 1

    if self.ndshape._size != num_elements_new:
        raise Error("Cannot reshape: Number of elements do not match.")

    var shape_new: List[Int] = List[Int]()

    for i in range(ndim_new):
        shape_new.append(Shape[i])
        var temp: Int = 1
        for j in range(i + 1, ndim_new):  # temp
            temp *= Shape[j]

    self.ndim = ndim_new
    self.ndshape = NDArrayShape(shape=shape_new)
    self.stride = NDArrayStride(shape=shape_new, order=order)
    self.order = order


fn ravel():
    pass


fn where[dtype: DType](inout x: NDArray[dtype], scalar: SIMD[dtype, 1], mask:NDArray[DType.bool]) raises:
    """
    Replaces elements in `x` with `scalar` where `mask` is True.

    Parameters:
        dtype: DType.
    
    Args:
        array: A NDArray.
        scalar: A SIMD value.
        mask: A NDArray.

    Returns:
        The modified array `x` after applying the mask
    """
    for i in range(array.ndshape._size):
        if mask.data[i] == True:
            array.data.store(i, scalar)

# TODO: do it with vectorization
fn where[dtype: DType](inout x: NDArray[dtype], y: NDArray[dtype], mask:NDArray[DType.bool]) raises:
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

    Returns:
        The modified array `x` after applying the mask
    """
    if x.ndshape != y.ndshape:
        raise Error("Shape mismatch error: x and y must have the same shape")
    for i in range(x.ndshape._size):
        if mask.data[i] == True:
            x.data.store(i, y.data[i])




