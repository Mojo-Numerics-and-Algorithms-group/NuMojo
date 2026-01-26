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
# 1. NDArray dtype conversions.
# 2. Numojo.NDArray to other collections.
# 3. Miscellaneous utility functions.
# ===----------------------------------------------------------------------=== #

from algorithm.functional import vectorize, parallelize
from collections import Dict
from memory import memcpy
from memory import UnsafePointer
from python import Python, PythonObject
from sys import simd_width_of


from numojo.core.layout import Flags, NDArrayShape, NDArrayStrides
from numojo.core.ndarray import NDArray
from numojo.core.error import NumojoError


comptime newaxis: NewAxis = NewAxis()


# TODO: add an initializer with int field to specify number of new axes to add!
struct NewAxis(Stringable):
    fn __init__(out self):
        """
        Initializes a NewAxis instance.
        """
        pass

    fn __repr__(self) -> String:
        """
        Returns a string representation of the NewAxis instance.

        Returns:
            Str: The string "NewAxis()".
        """
        return "numojo.newaxis()"

    fn __str__(self) -> String:
        """
        Returns a string representation of the NewAxis instance.

        Returns:
            Str: The string "NewAxis()".
        """
        return "numojo.newaxis()"

    fn __eq__(self, other: Self) -> Bool:
        """
        Checks equality between two NewAxis instances.
        """
        return True

    fn __ne__(self, other: Self) -> Bool:
        """
        Checks inequality between two NewAxis instances.
        """
        return False


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
