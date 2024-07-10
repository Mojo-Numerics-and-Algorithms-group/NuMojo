"""
# ===----------------------------------------------------------------------=== #
# Implements N-DIMENSIONAL ARRAY UTILITY FUNCTIONS
# Last updated: 2024-06-20
# ===----------------------------------------------------------------------=== #
"""

from python import Python
from .ndarray import NDArray, NDArrayShape, NDArrayStride

# TODO: there's some problem with using narr[idx] in traverse function, Make sure to correct this before v0.1


fn _get_index(indices: List[Int], weights: NDArrayShape) raises -> Int:
    var idx: Int = 0
    for i in range(weights._len):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: VariadicList[Int], weights: NDArrayShape) raises -> Int:
    var idx: Int = 0
    for i in range(weights._len):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: List[Int], weights: NDArrayStride) raises -> Int:
    var idx: Int = 0
    for i in range(weights._len):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: VariadicList[Int], weights: NDArrayStride) raises -> Int:
    var idx: Int = 0
    for i in range(weights._len):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: List[Int], weights: List[Int]) -> Int:
    var idx: Int = 0
    for i in range(weights.__len__()):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: VariadicList[Int], weights: VariadicList[Int]) -> Int:
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
    if depth == ndim.__len__():
        var idx = offset + _get_index(index, coefficients)
        var nidx = _get_index(index, strides)
        var temp = orig.data.load[width=1](idx)
        if nidx >= narr.ndshape._size:
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


fn bool_to_numeric[
    dtype: DType
](array: NDArray[DType.bool]) raises -> NDArray[dtype]:
    # Can't use simd becuase of bit packing error
    var res: NDArray[dtype] = NDArray[dtype](array.shape())
    for i in range(array.size()):
        var t = array.at(i)
        if t:
            res.__setitem__(i, 1)
        else:
            res.__setitem__(i, 0)
    return res


fn to_numpy[dtype: DType](array: NDArray[dtype]) raises -> PythonObject:
    """Transform the Numojo NDArray to a Numpy NDArray.

    Example:
    ```console
    > var A = NDArray(3, 3, random=True)
    > print(A)
    [[      0.1315377950668335      0.458650141954422       0.21895918250083923     ]
     [      0.67886471748352051     0.93469291925430298     0.51941639184951782     ]
     [      0.034572109580039978    0.52970021963119507     0.007698186207562685    ]]
    2-D array  Shape: [3, 3]  DType: float32
    > var B = A.to_numpy()
    > print(B)
    [[0.1315 0.4587 0.219 ]
     [0.6789 0.9347 0.5194]
     [0.0346 0.5297 0.0077]]
    > print(B.dtype)
    float32
    ```

    Parameters:
        dtype: The dtype of the original array.
    
    Args:
        array: The original Numojo NDArray.

    Returns:
        A Numpy NDArray with the same data type.
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
