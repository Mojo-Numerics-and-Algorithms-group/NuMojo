"""
Implements N-DIMENSIONAL ARRAY UTILITY FUNCTIONS
"""
# ===----------------------------------------------------------------------=== #
# Implements N-DIMENSIONAL ARRAY UTILITY FUNCTIONS
# Last updated: 2024-10-14
# ===----------------------------------------------------------------------=== #

from algorithm.functional import vectorize
from python import Python, PythonObject
from memory import UnsafePointer, memcpy
from sys import simdwidthof

from .ndshape import NDArrayShape
from .ndstrides import NDArrayStrides
from .ndarray import NDArray


# FIXME: No long useful from 24.6:
# `width` is now inferred from the SIMD's width.
fn fill_pointer[
    dtype: DType
](
    mut array: UnsafePointer[Scalar[dtype]], size: Int, value: Scalar[dtype]
) raises:
    """
    Fill a NDArray with a specific value.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        array: The pointer to the NDArray.
        size: The size of the NDArray.
        value: The value to fill the NDArray with.
    """
    alias width = simdwidthof[dtype]()

    @parameter
    fn vectorized_fill[simd_width: Int](idx: Int):
        array.store(idx, value)

    vectorize[vectorized_fill, width](size)


# ===----------------------------------------------------------------------=== #
# GET INDEX FUNCTIONS FOR NDARRAY
# ===----------------------------------------------------------------------=== #
# define a ndarray internal trait and remove multiple overloads of these _get_index
fn _get_index(indices: List[Int], weights: NDArrayShape) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.ndim):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: VariadicList[Int], weights: NDArrayShape) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.ndim):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: List[Int], weights: NDArrayStrides) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.ndim):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: Idx, weights: NDArrayStrides) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var index: Int = 0
    for i in range(weights.ndim):
        index += indices[i] * weights[i]
    return index


fn _get_index(
    indices: VariadicList[Int], weights: NDArrayStrides
) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.ndim):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: List[Int], weights: List[Int]) -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.__len__()):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: VariadicList[Int], weights: VariadicList[Int]) -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.__len__()):
        idx += indices[i] * weights[i]
    return idx


# ===----------------------------------------------------------------------=== #
# Funcitons to traverse a multi-dimensional array
# ===----------------------------------------------------------------------=== #


fn _traverse_buffer_according_to_shape_and_strides(
    mut ptr: UnsafePointer[Scalar[DType.index]],
    shape: NDArrayShape,
    strides: NDArrayStrides,
    current_dim: Int = 0,
    previous_sum: Int = 0,
) raises:
    """
    Traverses buffer according to new shape and strides.

    UNSAFE: Raw pointer is used!

    It is auxiliary functions that set values according to new shape
    and strides for variadic number of dimensions.

    Args:
        ptr: Pointer to buffer of 1-d index array, uninitialized.
        shape: NDArrayShape.
        strides: NDArrayStrides.
        current_dim: Temporarily save the current dimension.
        previous_sum: Temporarily save the previous summed index.

    Example:
    ```console
    var A = nm.random.randn(2, 3, 4)
    var I = nm.NDArray[DType.index](nm.Shape(A.size))
    var ptr = I._buf
    _traverse_buffer_according_to_shape_and_strides(
        ptr, A.shape._flip(), A.strides._flip()
    )
    print(I)
    # This prints:
    # [       0       12      4       ...     19      11      23      ]
    # 1-D array  Shape: [24]  DType: index  order: C
    ```
    """
    for index_of_axis in range(shape[current_dim]):
        var current_sum = previous_sum + index_of_axis * strides[current_dim]
        if current_dim >= shape.ndim - 1:
            ptr.init_pointee_copy(current_sum)
            ptr += 1
        else:
            _traverse_buffer_according_to_shape_and_strides(
                ptr,
                shape,
                strides,
                current_dim + 1,
                current_sum,
            )


fn _traverse_iterative[
    dtype: DType
](
    orig: NDArray[dtype],
    mut narr: NDArray[dtype],
    ndim: List[Int],
    coefficients: List[Int],
    strides: List[Int],
    offset: Int,
    mut index: List[Int],
    depth: Int,
) raises:
    """
    Traverse a multi-dimensional array in a iterative manner.

    Raises:
        Error: If the index is out of bound.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        orig: The original array.
        narr: The array to store the result.
        ndim: The number of dimensions of the array.
        coefficients: The coefficients to traverse the sliced part of the original array.
        strides: The strides to traverse the new NDArray `narr`.
        offset: The offset to the first element of the original NDArray.
        index: The list of indices.
        depth: The depth of the indices.
    """
    var total_elements = narr.size

    # # parallelized version was slower xD
    for _ in range(total_elements):
        var orig_idx = offset + _get_index(index, coefficients)
        var narr_idx = _get_index(index, strides)
        try:
            if narr_idx >= total_elements:
                raise Error("Invalid index: index out of bound")
        except:
            return

        narr._buf.store(narr_idx, orig._buf.load[width=1](orig_idx))

        for d in range(ndim.__len__() - 1, -1, -1):
            index[d] += 1
            if index[d] < ndim[d]:
                break
            index[d] = 0


fn _traverse_iterative_setter[
    dtype: DType
](
    orig: NDArray[dtype],
    mut narr: NDArray[dtype],
    ndim: List[Int],
    coefficients: List[Int],
    strides: List[Int],
    offset: Int,
    mut index: List[Int],
) raises:
    """
    Traverse a multi-dimensional array in a iterative manner.

    Raises:
        Error: If the index is out of bound.

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        orig: The original array.
        narr: The array to store the result.
        ndim: The number of dimensions of the array.
        coefficients: The coefficients to traverse the sliced part of the original array.
        strides: The strides to traverse the new NDArray `narr`.
        offset: The offset to the first element of the original NDArray.
        index: The list of indices.
    """
    # # parallelized version was slower xD
    var total_elements = narr.size
    for _ in range(total_elements):
        var orig_idx = offset + _get_index(index, coefficients)
        var narr_idx = _get_index(index, strides)
        try:
            if narr_idx >= total_elements:
                raise Error("Invalid index: index out of bound")
        except:
            return

        narr._buf.store(orig_idx, orig._buf.load[width=1](narr_idx))

        for d in range(ndim.__len__() - 1, -1, -1):
            index[d] += 1
            if index[d] < ndim[d]:
                break
            index[d] = 0


# ===----------------------------------------------------------------------=== #
# NDArray conversions
# ===----------------------------------------------------------------------=== #
fn bool_to_numeric[
    dtype: DType
](array: NDArray[DType.bool]) raises -> NDArray[dtype]:
    """
    Convert a boolean NDArray to a numeric NDArray.

    Parameters:
        dtype: The data type of the output NDArray elements.

    Args:
        array: The boolean NDArray to convert.

    Returns:
        The converted NDArray of type `dtype` with 1s (True) and 0s (False).
    """
    # Can't use simd becuase of bit packing error
    var res: NDArray[dtype] = NDArray[dtype](array.shape)
    for i in range(array.size):
        var t: Bool = array.item(i)
        if t:
            res._buf[i] = 1
        else:
            res._buf[i] = 0
    return res


fn to_numpy[dtype: DType](array: NDArray[dtype]) raises -> PythonObject:
    """
    Convert a NDArray to a numpy array.

    Example:
    ```console
    var arr = NDArray[DType.float32](3, 3, 3)
    var np_arr = to_numpy(arr)
    var np_arr1 = arr.to_numpy()
    ```

    Parameters:
        dtype: The data type of the NDArray elements.

    Args:
        array: The NDArray to convert.

    Returns:
        The converted numpy array.
    """
    try:
        var np = Python.import_module("numpy")

        np.set_printoptions(4)

        var dimension = array.ndim
        var np_arr_dim = PythonObject([])

        for i in range(dimension):
            np_arr_dim.append(array.shape[i])

        # Implement a dictionary for this later
        var numpyarray: PythonObject
        var np_dtype = np.float64
        if dtype == DType.float16:
            np_dtype = np.float16
        elif dtype == DType.float32:
            np_dtype = np.float32
        elif dtype == DType.int64:
            np_dtype = np.int64
        elif dtype == DType.int32:
            np_dtype = np.int32
        elif dtype == DType.int16:
            np_dtype = np.int16
        elif dtype == DType.int8:
            np_dtype = np.int8
        elif dtype == DType.index:
            np_dtype = np.intp
        elif dtype == DType.uint64:
            np_dtype = np.uint64
        elif dtype == DType.uint32:
            np_dtype = np.uint32
        elif dtype == DType.uint16:
            np_dtype = np.uint16
        elif dtype == DType.uint8:
            np_dtype = np.uint8
        elif dtype == DType.bool:
            np_dtype = np.bool_

        numpyarray = np.empty(np_arr_dim, dtype=np_dtype, order=array.order)
        var pointer_d = numpyarray.__array_interface__["data"][
            0
        ].unsafe_get_as_pointer[dtype]()
        memcpy(pointer_d, array.unsafe_ptr(), array.num_elements())
        _ = array

        return numpyarray^

    except e:
        print("Error in converting to numpy", e)
        return PythonObject()


# ===----------------------------------------------------------------------=== #
# Type checking functions
# ===----------------------------------------------------------------------=== #
@parameter
fn is_inttype[dtype: DType]() -> Bool:
    """
    Check if the given dtype is an integer type at compile time.

    Parameters:
        dtype: DType.

    Returns:
        Bool: True if the given dtype is an integer type, False otherwise.
    """

    @parameter
    if (
        dtype == DType.int8
        or dtype == DType.int16
        or dtype == DType.int32
        or dtype == DType.int64
    ):
        return True
    return False


fn is_inttype(dtype: DType) -> Bool:
    """
    Check if the given dtype is an integer type at run time.

    Args:
        dtype: DType.

    Returns:
        Bool: True if the given dtype is an integer type, False otherwise.
    """
    if (
        dtype == DType.int8
        or dtype == DType.int16
        or dtype == DType.int32
        or dtype == DType.int64
    ):
        return True
    return False


@parameter
fn is_floattype[dtype: DType]() -> Bool:
    """
    Check if the given dtype is a floating point type at compile time.

    Parameters:
        dtype: DType.

    Returns:
        Bool: True if the given dtype is a floating point type, False otherwise.
    """

    @parameter
    if (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
    ):
        return True
    return False


fn is_floattype(dtype: DType) -> Bool:
    """
    Check if the given dtype is a floating point type at run time.

    Args:
        dtype: DType.

    Returns:
        Bool: True if the given dtype is a floating point type, False otherwise.
    """
    if (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
    ):
        return True
    return False


@parameter
fn is_booltype[dtype: DType]() -> Bool:
    """
    Check if the given dtype is a boolean type at compile time.

    Parameters:
        dtype: DType.

    Returns:
        Bool: True if the given dtype is a boolean type, False otherwise.
    """

    @parameter
    if dtype == DType.bool:
        return True
    return False


fn is_booltype(dtype: DType) -> Bool:
    """
    Check if the given dtype is a boolean type at run time.

    Args:
        dtype: DType.

    Returns:
        Bool: True if the given dtype is a boolean type, False otherwise.
    """
    if dtype == DType.bool:
        return True
    return False


fn _list_of_range(n: Int) -> List[Int]:
    """
    Generate a list of integers starting from 0 and of size n.
    """

    var l = List[Int]()
    for i in range(n):
        l.append(i)
    return l


fn _list_of_flipped_range(n: Int) -> List[Int]:
    """
    Generate a list of integers starting from n-1 to 0 and of size n.
    """

    var l = List[Int]()
    for i in range(n - 1, -1, -1):
        l.append(i)
    return l
