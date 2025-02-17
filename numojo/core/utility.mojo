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
from memory import UnsafePointer, memcpy
from python import Python, PythonObject
from sys import simdwidthof
from tensor import Tensor, TensorShape

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

    var remainder = offset
    var indices = Item(ndim=len(strides), initialized=False)
    for i in range(len(strides)):
        indices[i], remainder = divmod(remainder, strides[i])

    return _get_offset(indices, strides._flip())


# ===----------------------------------------------------------------------=== #
# Functions to traverse a multi-dimensional array
# ===----------------------------------------------------------------------=== #


fn _traverse_buffer_according_to_shape_and_strides(
    mut ptr: UnsafePointer[Scalar[DType.index]],
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
    var I = nm.NDArray[DType.index](nm.Shape(A.size))
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
# Apply a function to NDArray by axis
# ===----------------------------------------------------------------------=== #


fn apply_func_on_array_with_dim_reduction[
    dtype: DType,
    func: fn[dtype_func: DType] (NDArray[dtype_func]) raises -> Scalar[
        dtype_func
    ],
](a: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Applies a function to a NDArray by axis and reduce that dimension.

    Raises:
        Error when the array is 1-d.

    Parameters:
        dtype: The data type of the input NDArray elements.
        func: The function to apply to the NDArray.

    Args:
        a: The NDArray to apply the function to.
        axis: The axis to apply the function to.

    Returns:
        The NDArray with the function applied to the input NDArray by axis.
    """

    if a.ndim == 1:
        raise Error("\n`axis` argument is not allowed for 1-d array.")

    var res = NDArray[dtype](a.shape._pop(axis=axis))
    var offset = 0
    for i in a.iter_by_axis(axis=axis):
        (res._buf.ptr + offset).init_pointee_copy(func[dtype](i))
        offset += 1
    return res^


fn apply_func_on_array_with_dim_reduction[
    dtype: DType, //,
    returned_dtype: DType,
    func: fn[dtype_func: DType, //, returned_dtype_func: DType] (
        NDArray[dtype_func]
    ) raises -> Scalar[returned_dtype_func],
](a: NDArray[dtype], axis: Int) raises -> NDArray[returned_dtype]:
    """
    Applies a function to a NDArray by axis and reduce that dimension.
    The target data type of the returned NDArray is different from the input
    NDArray.
    This is a function overload.

    Raises:
        Error when the array is 1-d.

    Parameters:
        dtype: The data type of the input NDArray elements.
        returned_dtype: The data type of the output NDArray elements.
        func: The function to apply to the NDArray.

    Args:
        a: The NDArray to apply the function to.
        axis: The axis to apply the function to.

    Returns:
        The NDArray with the function applied to the input NDArray by axis.
    """
    if a.ndim == 1:
        raise Error("\n`axis` argument is not allowed for 1-d array.")

    var res = NDArray[returned_dtype](a.shape._pop(axis=axis))
    # The iterator along the axis
    var iterator = a.iter_by_axis(axis=axis)

    @parameter
    fn parallelized_func(i: Int):
        try:
            (res._buf.ptr + i).init_pointee_copy(
                func[returned_dtype](iterator.ith(i))
            )
        except e:
            print("Error in parallelized_func", e)

    parallelize[parallelized_func](a.size // a.shape[axis])

    return res^


fn apply_func_on_array_without_dim_reduction[
    dtype: DType,
    func: fn[dtype_func: DType] (NDArray[dtype_func]) raises -> NDArray[
        dtype_func
    ],
](a: NDArray[dtype], axis: Int) raises -> NDArray[dtype]:
    """
    Applies a function to a NDArray by axis without reducing that dimension.
    The resulting array will have the same shape as the input array.

    Parameters:
        dtype: The data type of the input NDArray elements.
        func: The function to apply to the NDArray.

    Args:
        a: The NDArray to apply the function to.
        axis: The axis to apply the function to.

    Returns:
        The NDArray with the function applied to the input NDArray by axis.
    """

    # The iterator along the axis
    var iterator = a.iter_by_axis(axis=axis)
    # The final output array will have the same shape as the input array
    var res = NDArray[dtype](a.shape)

    if a.flags.C_CONTIGUOUS and (axis == a.ndim - 1):
        # The memory layout is C-contiguous
        var iterator = a.iter_by_axis(axis=axis)

        @parameter
        fn parallelized_func_c(i: Int):
            try:
                var elements: NDArray[dtype] = func[dtype](iterator.ith(i))
                memcpy(
                    res._buf.ptr + i * elements.size,
                    elements._buf.ptr,
                    elements.size,
                )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func_c](a.size // a.shape[axis])

    else:
        # The memory layout is not contiguous
        @parameter
        fn parallelized_func(i: Int):
            try:
                # The indices of the input array in each iteration
                var indices: NDArray[DType.index]
                # The elements of the input array in each iteration
                var elements: NDArray[dtype]
                # The array after applied the function
                indices, elements = iterator.ith_with_offsets(i)

                var res_along_axis: NDArray[dtype] = func[dtype](elements)

                for j in range(a.shape[axis]):
                    (res._buf.ptr + Int(indices[j])).init_pointee_copy(
                        (res_along_axis._buf.ptr + j)[]
                    )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func](a.size // a.shape[axis])

    return res^


fn apply_func_on_array_without_dim_reduction[
    dtype: DType,
    func: fn[dtype_func: DType] (NDArray[dtype_func]) raises -> NDArray[
        DType.index
    ],
](a: NDArray[dtype], axis: Int) raises -> NDArray[DType.index]:
    """
    Applies a function to a NDArray by axis without reducing that dimension.
    The resulting array will have the same shape as the input array.
    The resulting array is an index array.
    It can be used for, e.g., argsort.

    Parameters:
        dtype: The data type of the input NDArray elements.
        func: The function to apply to the NDArray.

    Args:
        a: The NDArray to apply the function to.
        axis: The axis to apply the function to.

    Returns:
        The index array with the function applied to the input array by axis.
    """

    # The iterator along the axis
    var iterator = a.iter_by_axis(axis=axis)
    # The final output array will have the same shape as the input array
    var res = NDArray[DType.index](a.shape)

    if a.flags.C_CONTIGUOUS and (axis == a.ndim - 1):
        # The memory layout is C-contiguous
        var iterator = a.iter_by_axis(axis=axis)

        @parameter
        fn parallelized_func_c(i: Int):
            try:
                var elements: NDArray[DType.index] = func[dtype](
                    iterator.ith(i)
                )
                memcpy(
                    res._buf.ptr + i * elements.size,
                    elements._buf.ptr,
                    elements.size,
                )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func_c](a.size // a.shape[axis])

    else:
        # The memory layout is not contiguous
        @parameter
        fn parallelized_func(i: Int):
            try:
                # The indices of the input array in each iteration
                var indices: NDArray[DType.index]
                # The elements of the input array in each iteration
                var elements: NDArray[dtype]
                # The array after applied the function
                indices, elements = iterator.ith_with_offsets(i)

                var res_along_axis: NDArray[DType.index] = func[dtype](elements)

                for j in range(a.shape[axis]):
                    (res._buf.ptr + Int(indices[j])).init_pointee_copy(
                        (res_along_axis._buf.ptr + j)[]
                    )
            except e:
                print("Error in parallelized_func", e)

        parallelize[parallelized_func](a.size // a.shape[axis])

    return res^


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
    var res: NDArray[dtype] = NDArray[dtype](array.shape)
    for i in range(array.size):
        var t: Bool = array.item(i)
        if t:
            res._buf.ptr[i] = 1
        else:
            res._buf.ptr[i] = 0
    return res


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

        var order = "C" if array.flags.C_CONTIGUOUS else "F"
        numpyarray = np.empty(np_arr_dim, dtype=np_dtype, order=order)
        var pointer_d = numpyarray.__array_interface__["data"][
            0
        ].unsafe_get_as_pointer[dtype]()
        memcpy(pointer_d, array.unsafe_ptr(), array.num_elements())
        _ = array

        return numpyarray^

    except e:
        print("Error in converting to numpy", e)
        return PythonObject()


fn to_tensor[dtype: DType](a: NDArray[dtype]) raises -> Tensor[dtype]:
    """
    Convert to a tensor.
    """
    pass

    var shape = List[Int]()
    for i in range(a.ndim):
        shape.append(a.shape[i])
    var t = Tensor[dtype](TensorShape(shape))
    memcpy(t._ptr, a._buf.ptr, a.size)

    return t


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
