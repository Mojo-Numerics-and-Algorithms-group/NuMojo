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


from numojo.core.layout import Flags, NDArrayShape, NDArrayStrides
from numojo.core.ndarray import NDArray
from numojo.core.error import IndexError

# ===----------------------------------------------------------------------=== #
# Internal Data Structures
# ===----------------------------------------------------------------------=== #


struct InternalSlice(ImplicitlyCopyable):
    var start: Int
    var end: Int
    var step: Int

    fn __init__(out self, start: Int, end: Int, step: Int):
        self.start = start
        self.end = end
        self.step = step

    fn __repr__(self) -> String:
        return "InternalSlice(start={}, end={}, step={})".format(
            self.start, self.end, self.step
        )

    fn __str__(self) -> String:
        return "InternalSlice(start={}, end={}, step={})".format(
            self.start, self.end, self.step
        )

    fn __eq__(self, other: Self) -> Bool:
        return (
            self.start == other.start
            and self.end == other.end
            and self.step == other.step
        )

    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    fn to_tuple(self) -> Tuple[Int, Int, Int]:
        return (self.start, self.end, self.step)

    fn to_slice(self) -> Slice:
        return Slice(self.start, self.end, self.step)

    fn normalize(self, dim: Int) -> InternalSlice:
        var start_norm = self.start
        var end_norm = self.end

        if self.start < 0:
            start_norm = dim + self.start
        if self.end < 0:
            end_norm = dim + self.end

        return InternalSlice(start_norm, end_norm, self.step)

    fn check_bounds(self, dim: Int) raises CustomError:
        if self.start < 0 or self.start >= dim:
            raise IndexError(
                message=(
                    "Slice start index {} out of bounds for dimension of"
                    " size {}".format(self.start, dim)
                ),
                location="InternalSlice.check_bounds()",
            )
        if self.end < 0 or self.end > dim:
            raise IndexError(
                message=(
                    "Slice end index {} out of bounds for dimension of size {}"
                    .format(self.end, dim)
                ),
                location="InternalSlice.check_bounds()",
            )
        if self.step == 0:
            raise IndexError(
                message="Slice step cannot be zero",
                location="InternalSlice.check_bounds()",
            )


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
# Offset and traverse functions
# ===----------------------------------------------------------------------=== #


struct Validator:
    @staticmethod
    fn check_row_bounds(x: Int, dim: Int) raises:
        """
        Check if row index is within bounds.

        Args:
            x: The row index to check.
            dim: The size of the dimension.

        Raises:
            Error: If the row index is out of bounds.
        """
        if x >= dim or x < -dim:
            raise Error(
                String(
                    "Row index {} out of bounds for matrix with {} rows"
                ).format(x, dim)
            )

    @staticmethod
    fn check_col_bounds(y: Int, dim: Int) raises:
        """
        Check if column index is within bounds.

        Args:
            y: The column index to check.
            dim: The size of the dimension.

        Raises:
            Error: If the column index is out of bounds.
        """
        if y >= dim or y < -dim:
            raise Error(
                String(
                    "Column index {} out of bounds for matrix with {} columns"
                ).format(y, dim)
            )

    @staticmethod
    fn check_bounds(x: Int, y: Int, dim_x: Int, dim_y: Int) raises:
        """
        Check if both row and column indices are within bounds.

        Args:
            x: The row index to check.
            y: The column index to check.
            dim_x: The size of the row dimension.
            dim_y: The size of the column dimension.

        Raises:
            Error: If either index is out of bounds.
        """
        if x >= dim_x or x < -dim_x or y >= dim_y or y < -dim_y:
            raise Error(
                String(
                    "Index ({}, {}) out of bounds for matrix shape ({}, {})"
                ).format(x, y, dim_x, dim_y)
            )


struct IndexMethods:
    @staticmethod
    fn normalize(idx: Int, dim: Int) -> Int:
        """
        Normalize a potentially negative index to its positive equivalent
        within the bounds of the given dimension.

        Args:
            idx: The index to normalize. Can be negative to indicate indexing
                 from the end (e.g., -1 refers to the last element).
            dim: The size of the dimension to normalize against.

        Returns:
            The normalized index as a non-negative integer.
            ```
        """
        var idx_norm = idx
        if idx_norm < 0:
            idx_norm = dim + idx_norm
        return idx_norm

    @staticmethod
    fn is_valid_index(idx: Int, dim: Int) -> Bool:
        """
        Check if the given index is valid for the specified dimension.

        Args:
            idx: The index to check.
            dim: The size of the dimension.

        Returns:
            True if the index is valid (0 <= idx < dim), False otherwise.
        """
        return idx >= 0 and idx < dim

    @staticmethod
    fn get_1d_index(indices: List[Int], strides: NDArrayStrides) raises -> Int:
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

    @staticmethod
    fn get_1d_index(indices: Item, strides: NDArrayStrides) raises -> Int:
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

    @staticmethod
    fn get_1d_index(
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

    @staticmethod
    fn get_1d_index(indices: List[Int], strides: List[Int]) -> Int:
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

    @staticmethod
    fn get_1d_index(
        indices: VariadicList[Int], strides: VariadicList[Int]
    ) -> Int:
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

    @staticmethod
    fn get_1d_index(indices: Tuple[Int, Int], strides: Tuple[Int, Int]) -> Int:
        """
        Get the index of matrix from a list of indices and strides.

        Args:
            indices: The list of indices.
            strides: The strides of the indices.

        Returns:
            Offset of contiguous memory layout.
        """
        return indices[0] * strides[0] + indices[1] * strides[1]

    @staticmethod
    fn transfer_offset(offset: Int, strides: NDArrayStrides) raises -> Int:
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

        return Self.get_1d_index(indices, strides._flip())


# ===----------------------------------------------------------------------=== #
# Functions to traverse a multi-dimensional array
# ===----------------------------------------------------------------------=== #
#
struct TraverseMethods:
    @staticmethod
    fn traverse_buffer_according_to_shape_and_strides[
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

        Parameters:
            origin: The mutability origin of the pointer.

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
            var current_sum = (
                previous_sum + index_of_axis * strides[current_dim]
            )
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

    @staticmethod
    fn traverse_iterative[
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

        # `strides` here is a logical multi-index -> linear offset mapping.
        # Using it directly as the destination offset breaks when `narr.strides`
        # is not a contiguous layout mapping (e.g. slices that create F-order views).
        # The destination buffer is always laid out contiguously for `narr`, so we
        # write using a simple linear counter.
        for lin in range(total_elements):
            var orig_idx = offset + _get_offset(index, coefficients)
            narr._buf.ptr.store(lin, orig._buf.ptr.load[width=1](orig_idx))

            for d in range(ndim.__len__() - 1, -1, -1):
                index[d] += 1
                if index[d] < ndim[d]:
                    break
                index[d] = 0

    @staticmethod
    fn traverse_iterative_setter[
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
        # The source `orig` being assigned from is contiguous in its own buffer.
        # When iterating logical indices, write/read using a contiguous linear
        # counter for `orig`, not a potentially non-contiguous stride mapping.
        var total_elements = narr.size
        for lin in range(total_elements):
            var orig_idx = offset + _get_offset(index, coefficients)
            orig._buf.ptr.store(orig_idx, narr._buf.ptr.load[width=1](lin))

            for d in range(ndim.__len__() - 1, -1, -1):
                index[d] += 1
                if index[d] < ndim[d]:
                    break
                index[d] = 0


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

    # `strides` here is a logical multi-index -> linear offset mapping.
    # Using it directly as the destination offset breaks when `narr.strides`
    # is not a contiguous layout mapping (e.g. slices that create F-order views).
    # The destination buffer is always laid out contiguously for `narr`, so we
    # write using a simple linear counter.
    for lin in range(total_elements):
        var orig_idx = offset + _get_offset(index, coefficients)
        narr._buf.ptr.store(lin, orig._buf.ptr.load[width=1](orig_idx))

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
    # The source `orig` being assigned from is contiguous in its own buffer.
    # When iterating logical indices, write/read using a contiguous linear
    # counter for `orig`, not a potentially non-contiguous stride mapping.
    var total_elements = narr.size
    for lin in range(total_elements):
        var orig_idx = offset + _get_offset(index, coefficients)
        orig._buf.ptr.store(orig_idx, narr._buf.ptr.load[width=1](lin))

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
