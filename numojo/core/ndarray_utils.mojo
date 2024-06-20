"""
# ===----------------------------------------------------------------------=== #
# Implements N-DIMENSIONAL ARRAY UTILITY FUNCTIONS
# Last updated: 2024-06-20
# ===----------------------------------------------------------------------=== #
"""

from python import Python
from .ndarray import NDArray


fn _get_index(indices: VariadicList[Int], weights: StaticIntTuple) -> Int:
    var idx: Int = 0
    for i in range(weights.__len__()):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(*indices: Int, weights: List[Int]) -> Int:
    var idx: Int = 0
    for i in range(weights.__len__()):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: List[Int], weights: List[Int]) -> Int:
    var idx: Int = 0
    for i in range(weights.__len__()):
        idx += indices[i] * weights[i]
    return idx


fn _get_index[
    MAX: Int
](indices: List[Int], weights: StaticIntTuple[MAX]) -> Int:
    var idx: Int = 0
    for i in range(weights.__len__()):
        idx += indices[i] * weights[i]
    return idx


fn indexing[
    MAX: Int
](indices: StaticIntTuple[MAX], weights: StaticIntTuple[MAX]) -> Int:
    var idx: Int = 0
    for i in range(ALLOWED):
        idx += indices[i] * weights[i]
    return idx


fn _traverse_iterative[
    dtype: DType
](
    orig: NDArray[dtype],
    inout narr: NDArray[dtype],
    shape_length: Int,
    nshape: StaticIntTuple[ALLOWED],
    coefficients: StaticIntTuple[ALLOWED],
    strides: StaticIntTuple[ALLOWED],
    offset: Int,
    inout index: StaticIntTuple[ALLOWED],
    depth: Int,
) raises:
    if depth == shape_length:
        var idx = offset + indexing[ALLOWED](
            indices=index, weights=coefficients
        )
        var nidx = indexing[ALLOWED](indices=index, weights=strides)
        # var temp = orig.data.load[width=1](idx)
        narr[nidx] = orig[
            idx
        ]  # TODO: replace with load_unsafe later for reduced checks overhead
        return

    for i in range(nshape[depth]):
        index[depth] = i
        var newdepth = depth + 1
        _traverse_iterative(
            orig,
            narr,
            shape_length,
            nshape,
            coefficients,
            strides,
            offset,
            index,
            newdepth,
        )


fn to_numpy(array: NDArray) -> PythonObject:
    try:
        var np = Python.import_module("numpy")

        np.set_printoptions(4)

        var dimension = array.ndim
        var np_arr_dim = PythonObject([])

        for i in range(dimension):
            np_arr_dim.append(array.ndshape[i])

        # Implement a dictionary for this later
        var numpyarray: PythonObject
        if array.datatype == DType.float16:
            numpyarray = np.empty(np_arr_dim, dtype=np.float16)
            var pointer = int(numpyarray.__array_interface__["data"][0].to_float16())
            var pointer_d = DTypePointer[array.dtype](address=pointer)
            memcpy(pointer_d, array.data(), array.num_elements())
        # elif array.datatype == DType.float32:
        #     numpyarray = np.empty(np_arr_dim, dtype=np.float32)
        #     var pointer = int(numpyarray.__array_interface__["data"][0].to_float32())
        # elif array.datatype == DType.float64:
        #     numpyarray = np.empty(np_arr_dim, dtype=np.float64)
        #     var pointer = int(numpyarray.__array_interface__["data"][0].to_float64())
        # elif array.datatype == DType.int8:
        #     numpyarray = np.empty(np_arr_dim, dtype=np.int8)
        #     var pointer = int(numpyarray.__array_interface__["data"][0].to_int8())
        # elif array.datatype == DType.int16:
        #     numpyarray = np.empty(np_arr_dim, dtype=np.int16)
        #     var pointer = int(numpyarray.__array_interface__["data"][0].to_int16())
        # elif array.datatype == DType.int32:
        #     numpyarray = np.empty(np_arr_dim, dtype=np.int32)
        #     var pointer = int(numpyarray.__array_interface__["data"][0].to_int32())
        # elif array.datatype == DType.int64:
        #     numpyarray = np.empty(np_arr_dim, dtype=np.int64)
        #     var pointer = int(numpyarray.__array_interface__["data"][0].to_int64())

        _ = array

        return numpyarray^

    except e:
        print("Error in converting to numpy", e)
        return PythonObject()