# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Implements N-DIMENSIONAL ARRAY UTILITY FUNCTIONS
"""
# ===----------------------------------------------------------------------=== #
# SECTIONS OF THE FILE:
#
# 1. Offset and traverse functions.
# 2. Functions to traverse a multi-dimensional array.
# 3. Apply a function to NDArray by axis.
# 4. NDArray dtype conversions.
# 5. Numojo.NDArray to other collections.
# 6. Type checking functions.
# 7. Miscellaneous utility functions.
# ===----------------------------------------------------------------------=== #

from algorithm.functional import vectorize, parallelize
from collections import Dict
from memory import memcpy
from memory import UnsafePointer
from python import Python, PythonObject
from sys import simd_width_of

# from tensor import Tensor, TensorShape

from numojo.core.flags import Flags
from numojo.core.ndarray import NDArray
from numojo.core.ndshape import NDArrayShape
from numojo.core.ndstrides import NDArrayStrides

# ===----------------------------------------------------------------------=== #
# Offset and traverse functions
# ===----------------------------------------------------------------------=== #


fn _get_offset(indices: List[Int], strides: NDArrayStrides) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and strides.

    Args:
        indices: The list of indices.
        strides: The strides of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(strides.ndim):
        idx += indices[i] * strides[i]
    return idx


fn _get_offset(indices: Item, strides: NDArrayStrides) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and strides.

    Args:
        indices: The list of indices.
        strides: The strides of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var index: Int = 0
    for i in range(strides.ndim):
        index += indices[i] * strides[i]
    return index


fn _get_offset(
    indices: VariadicList[Int], strides: NDArrayStrides
) raises -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and strides.

    Args:
        indices: The list of indices.
        strides: The strides of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(strides.ndim):
        idx += indices[i] * strides[i]
    return idx


fn _get_offset(indices: List[Int], strides: List[Int]) -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and strides.

    Args:
        indices: The list of indices.
        strides: The strides of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(strides.__len__()):
        idx += indices[i] * strides[i]
    return idx


fn _get_offset(indices: VariadicList[Int], strides: VariadicList[Int]) -> Int:
    """
    Get the index of a multi-dimensional array from a list of indices and strides.

    Args:
        indices: The list of indices.
        strides: The strides of the indices.

    Returns:
        The scalar index of the multi-dimensional array.
    """
    var idx: Int = 0
    for i in range(strides.__len__()):
        idx += indices[i] * strides[i]
    return idx


fn _get_offset(indices: Tuple[Int, Int], strides: Tuple[Int, Int]) -> Int:
    """
    Get the index of matrix from a list of indices and strides.

    Args:
        indices: The list of indices.
        strides: The strides of the indices.

    Returns:
        Offset of contiguous memory layout.
    """
    return indices[0] * strides[0] + indices[1] * strides[1]


fn _transfer_offset(offset: Int, strides: NDArrayStrides) raises -> Int:
    """
    Transfers the offset by flipping the strides information.
    It can be used to transfer between C-contiguous and F-continuous memory
    layout. For example, in a 4x4 C-contiguous array, the item with offset 4
    has the indices (1, 0). The item with the same indices (1, 0) in a
    F-continuous array has an offset of 1.

    Args:
        offset: The offset in memory of an element of array.
        strides: The strides of the array.

    Returns:
        The offset of the array of a flipped memory layout.
    """

    var remainder: Int = offset
    var indices: Item = Item(ndim=len(strides))
    for i in range(len(strides)):
        indices[i] = remainder // strides[i]
        remainder %= strides[i]

    return _get_offset(indices, strides._flip())


# ===----------------------------------------------------------------------=== #
# Functions to traverse a multi-dimensional array
# ===----------------------------------------------------------------------=== #


fn _traverse_buffer_according_to_shape_and_strides[
    origin: MutOrigin
](
    mut ptr: UnsafePointer[Scalar[DType.int], origin=origin],
    shape: NDArrayShape,
    strides: NDArrayStrides,
    current_dim: Int = 0,
    previous_sum: Int = 0,
) raises:
    """
    Store sequence of indices according to shape and strides into the pointer
    given in the arguments.

    It is auxiliary functions that get or set values according to new shape
    and strides for variadic number of dimensions.

    UNSAFE: Raw pointer is used!

    Args:
        ptr: Pointer to buffer of uninitialized 1-d index array.
        shape: NDArrayShape.
        strides: NDArrayStrides.
        current_dim: Temporarily save the current dimension.
        previous_sum: Temporarily save the previous summed index.

    Example:
    ```console
    # A is a 2x3x4 array
    var I = nm.NDArray[DType.int](nm.Shape(A.size))
    var ptr = I._buf
    _traverse_buffer_according_to_shape_and_strides(
        ptr, A.shape._flip(), A.strides._flip()
    )
    # I = [       0       12      4       ...     19      11      23      ]
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
        var orig_idx = offset + _get_offset(index, coefficients)
        var narr_idx = _get_offset(index, strides)
        try:
            if narr_idx >= total_elements:
                raise Error("Invalid index: index out of bound")
        except:
            return

        narr._buf.ptr.store(narr_idx, orig._buf.ptr.load[width=1](orig_idx))

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
        var orig_idx = offset + _get_offset(index, coefficients)
        var narr_idx = _get_offset(index, strides)
        try:
            if narr_idx >= total_elements:
                raise Error("Invalid index: index out of bound")
        except:
            return

        narr._buf.ptr.store(orig_idx, orig._buf.ptr.load[width=1](narr_idx))

        for d in range(ndim.__len__() - 1, -1, -1):
            index[d] += 1
            if index[d] < ndim[d]:
                break
            index[d] = 0


# ===----------------------------------------------------------------------=== #
# NDArray dtype conversions
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
    var result: NDArray[dtype] = NDArray[dtype](array.shape)
    for i in range(array.size):
        var t: Bool = array.item(i)
        if t:
            result._buf.ptr[i] = 1
        else:
            result._buf.ptr[i] = 0
    return result^


# ===----------------------------------------------------------------------=== #
# Numojo.NDArray to other collections
# ===----------------------------------------------------------------------=== #
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
        var np_arr_dim = Python.list()

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
        elif dtype == DType.int:
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

        var order = "C" if array.flags.C_CONTIGUOUS else "F"
        numpyarray = np.empty(
            np_arr_dim, dtype=np_dtype, order=PythonObject(order)
        )
        var pointer_d = numpyarray.__array_interface__[PythonObject("data")][
            0
        ].unsafe_get_as_pointer[dtype]()
        memcpy(dest=pointer_d, src=array.unsafe_ptr(), count=array.size)
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


# ===----------------------------------------------------------------------=== #
# Miscellaneous utility functions
# ===----------------------------------------------------------------------=== #


fn _list_of_range(n: Int) -> List[Int]:
    """
    Generate a list of integers starting from 0 and of size n.
    """

    var list_of_range: List[Int] = List[Int]()
    for i in range(n):
        list_of_range.append(i)
    return list_of_range^


fn _list_of_flipped_range(n: Int) -> List[Int]:
    """
    Generate a list of integers starting from n-1 to 0 and of size n.
    """

    var list_of_range: List[Int] = List[Int]()
    for i in range(n - 1, -1, -1):
        list_of_range.append(i)
    return list_of_range^
