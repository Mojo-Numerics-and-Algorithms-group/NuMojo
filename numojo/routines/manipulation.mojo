"""
Array manipulation routines.

"""

from memory import memcpy
from sys import simdwidthof
from algorithm import vectorize

from numojo.core.ndarray import NDArray
from numojo.core.ndshape import NDArrayShape, Shape
from numojo.core.ndstrides import NDArrayStrides
from numojo.core.utility import _list_of_flipped_range

# ===----------------------------------------------------------------------=== #
# Basic operations
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


# ===----------------------------------------------------------------------=== #
# Changing array shape
# ===----------------------------------------------------------------------=== #


fn reshape[
    dtype: DType
](
    owned A: NDArray[dtype], shape: NDArrayShape, order: String = "C"
) raises -> NDArray[dtype]:
    """
        Returns an array of the same data with a new shape.

    Raises:
        Error: If the number of elements do not match.

    Args:
        A: A NDArray.
        shape: New shape.
        order: "C" or "F". Read in this order from the original array and
            write in this order into the new array.

    Returns:
        Array of the same data with a new shape.
    """

    if A.size != shape.size:
        raise Error("Cannot reshape: Number of elements do not match.")

    if A.order != order:
        # Read in this order from the original array
        A = ravel(A, order=order)

    # Write in this order into the new array
    var B = NDArray[dtype](shape=shape, order=order)
    memcpy(dest=B._buf, src=A._buf, count=A.size)

    return B^


fn ravel[
    dtype: DType
](owned A: NDArray[dtype], order: String = "C") raises -> NDArray[dtype]:
    """
    Returns the raveled version of the NDArray.

    Args:
        A: NDArray.
        order: The order to flatten the array.

    Return:
        A contiguous flattened array.
    """
    if A.ndim == 1:
        return A
    else:
        if A.order != order:
            A = transpose(A, axes=_list_of_flipped_range(A.ndim))
        var B = NDArray[dtype](Shape(A.size))
        memcpy(B._buf, A._buf, A.size)
        return B


# ===----------------------------------------------------------------------=== #
# Transpose-like operations
# ===----------------------------------------------------------------------=== #


fn _set_values_according_to_new_shape_and_strides(
    mut I: NDArray[DType.index],
    mut index: Int,
    current_dim: Int,
    previous_sum: Int,
    new_shape: NDArrayShape,
    new_strides: NDArrayStrides,
) raises:
    """
    Auxiliary function for `transpose` that set values according to new shape
    and strides for variadic number of dimensions.
    """
    for index_of_axis in range(new_shape[current_dim]):
        var current_sum = previous_sum + index_of_axis * new_strides[
            current_dim
        ]
        if current_dim >= new_shape.ndim - 1:
            I._buf[index] = current_sum
            index = index + 1
        else:
            _set_values_according_to_new_shape_and_strides(
                I,
                index,
                current_dim + 1,
                current_sum,
                new_shape,
                new_strides,
            )


fn transpose[
    dtype: DType
](A: NDArray[dtype], axes: List[Int]) raises -> NDArray[dtype]:
    """
    Transpose array of any number of dimensions according to
    arbitrary permutation of the axes.

    If `axes` is not given, it is equal to flipping the axes.
    ```mojo
    import numojo as nm
    var A = nm.random.rand(2,3,4,5)
    print(nm.transpose(A))  # A is a 4darray.
    print(nm.transpose(A, axes=List(3,2,1,0)))
    ```

    Examples.
    ```mojo
    import numojo as nm
    # A is a 2darray
    print(nm.transpose(A, axes=List(0, 1)))  # equal to transpose of matrix
    # A is a 3darray
    print(nm.transpose(A, axes=List(2, 1, 0)))  # transpose 0-th and 2-th dimensions
    ```
    """
    if len(axes) != A.ndim:
        raise Error(
            String(
                "Length of `axes` ({}) does not match `ndim` of array ({})"
            ).format(len(axes), A.ndim)
        )

    for i in range(A.ndim):
        if i not in axes:
            raise Error(
                String(
                    "`axes` is not a valid permutation of axes of the array. "
                    "It does not contain index {}"
                ).format(i)
            )

    var _shape = List[Int]()
    var _strides = List[Int]()

    for i in range(A.ndim):
        _shape.append(A.shape[axes[i]])
    var new_shape = NDArrayShape(shape=_shape)

    for i in range(A.ndim):
        _strides.append(A.strides[axes[i]])
    var new_strides = NDArrayStrides(strides=_strides)

    var _index = 0
    var I = NDArray[DType.index](shape=new_shape, order=A.order)

    _set_values_according_to_new_shape_and_strides(
        I, _index, 0, 0, new_shape, new_strides
    )

    var B = NDArray[dtype](I.shape, order=A.order)
    for i in range(B.size):
        B._buf[i] = A._buf[I._buf[i]]
    return B^


fn transpose[dtype: DType](A: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    (overload) Transpose the array when `axes` is not given.
    If `axes` is not given, it is equal to flipping the axes.
    See docstring of `transpose`.
    """

    if A.ndim == 1:
        return A
    if A.ndim == 2:
        var B = NDArray[dtype](Shape(A.shape[1], A.shape[0]), order=A.order)
        if A.shape[0] == 1 or A.shape[1] == 1:
            memcpy(B._buf, A._buf, A.size)
        else:
            for i in range(B.shape[0]):
                for j in range(B.shape[1]):
                    B._setitem(i, j, val=A._getitem(j, i))
        return B^
    else:
        flipped_axes = List[Int]()
        for i in range(A.ndim - 1, -1, -1):
            flipped_axes.append(i)

        return transpose(A, axes=flipped_axes)


# ===----------------------------------------------------------------------=== #
# Rearranging elements
# ===----------------------------------------------------------------------=== #


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
