"""
Implements N-DIMENSIONAL ARRAY UTILITY FUNCTIONS
"""
# ===----------------------------------------------------------------------=== #
# Implements N-DIMENSIONAL ARRAY UTILITY FUNCTIONS
# Last updated: 2024-09-08
# ===----------------------------------------------------------------------=== #

from algorithm.functional import vectorize

from python import Python
from .ndarray import NDArray, NDArrayShape, NDArrayStride


fn fill_pointer[
    dtype: DType
](inout array: DTypePointer[dtype], size: Int, value: Scalar[dtype]) raises:
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
        array.store[width=simd_width](idx, value)

    vectorize[vectorized_fill, width](size)


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
    for i in range(weights.ndlen):
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
    for i in range(weights.ndlen):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: List[Int], weights: NDArrayStride) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.ndlen):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: VariadicList[Int], weights: NDArrayStride) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and weights.

    Args:
        indices: The list of indices.
        weights: The weights of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(weights.ndlen):
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


fn _traverse_iterative[
    dtype: DType
](
    orig: NDArray[dtype],
    inout narr: NDArray[dtype],
    ndim: List[Int],
    coefficients: List[Int],
    strides: List[Int],
    offset: Int,
    inout index: List[Int],
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
    if depth == ndim.__len__():
        var idx = offset + _get_index(index, coefficients)
        var nidx = _get_index(index, strides)
        var temp = orig.data.load[width=1](idx)
        if nidx >= narr.ndshape.ndsize:
            raise Error("Invalid index: index out of bound")
        else:
            narr.data.store[width=1](nidx, temp)
        return

    for i in range(ndim[depth]):
        index[depth] = i
        var newdepth = depth + 1
        _traverse_iterative(
            orig, narr, ndim, coefficients, strides, offset, index, newdepth
        )

fn _traverse_iterative_setter[
    dtype: DType
](
    orig: NDArray[dtype],
    inout narr: NDArray[dtype],
    ndim: List[Int],
    coefficients: List[Int],
    strides: List[Int],
    offset: Int,
    inout index: List[Int],
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
    if depth == ndim.__len__():
        var idx = offset + _get_index(index, coefficients)
        var nidx = _get_index(index, strides)
        var temp = orig.data.load[width=1](idx)
        if nidx >= narr.ndshape.ndsize:
            raise Error("Invalid index: index out of bound")
        else:
            narr.data.store[width=1](nidx, temp)
        return

    fn parallelized(idx: Int) -> None:
    # for i in range(ndim[depth]):
        index[depth] = i
        var newdepth = depth + 1
        _traverse_iterative(
            orig, narr, ndim, coefficients, strides, offset, index, newdepth
        )
    parallelized[parallelized](ndim[depth], ndim[depth])

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
    var res: NDArray[dtype] = NDArray[dtype](array.shape())
    for i in range(array.size()):
        var t: Bool = array.item(i)
        if t:
            res.data[i] = 1
        else:
            res.data[i] = 0
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
            np_arr_dim.append(array.ndshape[i])

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

        numpyarray = np.empty(np_arr_dim, dtype=np_dtype)
        var pointer = numpyarray.__array_interface__["data"][0]
        var pointer_d = DTypePointer[array.dtype](address=pointer)
        memcpy(pointer_d, array.data, array.num_elements())
        _ = array

        return numpyarray^

    except e:
        print("Error in converting to numpy", e)
        return PythonObject()
