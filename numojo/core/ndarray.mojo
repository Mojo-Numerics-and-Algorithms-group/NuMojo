# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Implements basic object methods for working with N-Dimensional Array.
"""

from algorithm import parallelize, vectorize
import builtin.math as builtin_math
import builtin.bool as builtin_bool
from builtin.type_aliases import Origin
from collections import Dict
from collections.optional import Optional
from memory import UnsafePointer, memset_zero, memcpy
from python import Python, PythonObject
from sys import simdwidthof
from tensor import Tensor
from utils import Variant
from utils.numerics import isnan, isinf

import numojo.core._array_funcs as _af
from numojo.core.ndshape import NDArrayShape
from numojo.core.ndstrides import NDArrayStrides
from numojo.core.item import Item
from numojo.core.own_data import OwnData
from numojo.core._math_funcs import Vectorized
from numojo.core.utility import (
    _get_offset,
    _update_flags,
    _traverse_iterative,
    _traverse_iterative_setter,
    to_numpy,
    to_tensor,
    bool_to_numeric,
    is_floattype,
)

from numojo.routines.io.formatting import (
    format_floating_precision,
    format_floating_scientific,
    PrintOptions,
    printoptions,
    GLOBAL_PRINT_OPTIONS,
)
import numojo.routines.creation as creation
import numojo.routines.sorting as sorting
import numojo.routines.math.arithmetic as arithmetic
import numojo.routines.logic.comparison as comparison
import numojo.routines.math.rounding as rounding
import numojo.routines.bitwise as bitwise
import numojo.routines.linalg as linalg
from numojo.core.datatypes import TypeCoercion, _concise_dtype_str
from numojo.routines.statistics.averages import mean, cummean
from numojo.routines.math.products import prod, cumprod
from numojo.routines.math.sums import sum, cumsum
from numojo.routines.logic.truth import any
from numojo.routines.linalg.products import matmul
from numojo.routines.manipulation import reshape, ravel

# ===----------------------------------------------------------------------===#
# NDArray
#
# TODO: Generalize mdot, rdot to take any IxJx...xKxL and LxMx...xNxP matrix and
#       matmul it into IxJx..xKxMx...xNxP array.
# TODO: Add vectorization for _get_offset.
# TODO: Create NDArrayView that points to the buffer of the raw array.
#       This requires enhancement of functionalities of traits from Mojo's side.
#       The data buffer can implement an ArrayData trait (RawData or RefData)
#       RawData type is just a wrapper of `UnsafePointer`.
#       RefData type has an extra property `indices`: getitem(i) -> A[I[i]].
# TODO: Rename some variables or methods that should not be exposed to users.
# ===----------------------------------------------------------------------===#


struct NDArray[dtype: DType = DType.float64](
    Stringable, Representable, CollectionElement, Sized, Writable
):
    # TODO: NDArray[dtype: DType = DType.float64,
    #               Buffer: Bufferable[dtype] = OwnData[dtype]]
    """The N-dimensional array (NDArray).

    Parameters:
        dtype: Type of item in NDArray. Default type is DType.float64.

    The array can be uniquely defined by the following:
        1. The data buffer of all items.
        2. The shape of the array.
        3. The strides (Length of item to travel to next dimension).
        4. The datatype of the elements.

    The following attributes are also helpful:
        - The number of dimensions
        - Size of the array (number of items)
        - The order of the array: Row vs Columns major
    """

    alias width: Int = simdwidthof[dtype]()
    """Vector size of the data type."""

    var _buf: OwnData[dtype]
    """Data buffer of the items in the NDArray."""
    var ndim: Int
    """Number of Dimensions."""
    var shape: NDArrayShape
    """Size and shape of NDArray."""
    var size: Int
    """Size of NDArray."""
    var strides: NDArrayStrides
    """Contains offset, strides."""
    var flags: Dict[String, Bool]
    "Information about the memory layout of the array."

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    # default constructor

    @always_inline("nodebug")
    fn __init__(
        mut self,
        shape: NDArrayShape,
        order: String = "C",
    ) raises:
        """
        Initialize an NDArray with given shape.

        The memory is not filled with values.

        Args:
            shape: Variadic shape.
            order: Memory order C or F.
        """

        self.ndim = shape.ndim
        self.shape = NDArrayShape(shape)
        self.size = self.shape.size_of_array()
        self.strides = NDArrayStrides(shape, order=order)
        self._buf = OwnData[dtype](self.size)
        # Initialize information on memory layout
        self.flags = Dict[String, Bool]()
        _update_flags(
            self.flags, shape=self.shape, strides=self.strides, ndim=self.ndim
        )
        self.flags["OWNDATA"] = True

    @always_inline("nodebug")
    fn __init__(
        mut self,
        shape: List[Int],
        order: String = "C",
    ) raises:
        """
        (Overload) Initialize an NDArray with given shape (list of integers).

        Args:
            shape: List of shape.
            order: Memory order C or F.
        """

        self = Self(Shape(shape), order)

    @always_inline("nodebug")
    fn __init__(
        mut self,
        shape: VariadicList[Int],
        order: String = "C",
    ) raises:
        """
        (Overload) Initialize an NDArray with given shape (variadic list of integers).

        Args:
            shape: Variadic List of shape.
            order: Memory order C or F.
        """

        self = Self(Shape(shape), order)

    # constructor when offset is known
    fn __init__(
        mut self,
        shape: List[Int],
        offset: Int,
        strides: List[Int],
    ) raises:
        """
        Extremely specific NDArray initializer.
        """
        self.shape = NDArrayShape(shape)
        self.ndim = self.shape.ndim
        self.size = self.shape.size_of_array()
        self.strides = NDArrayStrides(strides=strides)
        self._buf = OwnData[dtype](self.size)
        memset_zero(self._buf.ptr, self.size)
        # Initialize information on memory layout
        self.flags = Dict[String, Bool]()
        _update_flags(
            self.flags, shape=self.shape, strides=self.strides, ndim=self.ndim
        )
        self.flags["OWNDATA"] = True

    # for creating views (unsafe!)
    fn __init__(
        mut self,
        shape: NDArrayShape,
        ref buffer: UnsafePointer[Scalar[dtype]],
        offset: Int,
        strides: NDArrayStrides,
    ) raises:
        self.shape = shape
        self.strides = strides
        self.ndim = self.shape.ndim
        self.size = self.shape.size_of_array()
        self._buf = OwnData(ptr=buffer.offset(offset))
        # Initialize information on memory layout
        self.flags = Dict[String, Bool]()
        _update_flags(
            self.flags, shape=self.shape, strides=self.strides, ndim=self.ndim
        )
        self.flags["OWNDATA"] = False

    @always_inline("nodebug")
    fn __copyinit__(mut self, other: Self):
        """
        Copy other into self.

        It is a deep copy. So the new array owns the data.
        """
        self.ndim = other.ndim
        self.shape = other.shape
        self.size = other.size
        self.strides = other.strides
        self.flags = other.flags
        self._buf = OwnData[dtype](self.size)
        memcpy(self._buf.ptr, other._buf.ptr, other.size)
        self.flags["OWNDATA"] = True

    @always_inline("nodebug")
    fn __moveinit__(mut self, owned existing: Self):
        """
        Move other into self.
        """
        self.ndim = existing.ndim
        self.shape = existing.shape
        self.size = existing.size
        self.strides = existing.strides
        self.flags = existing.flags^
        self._buf = existing._buf^

    @always_inline("nodebug")
    fn __del__(owned self):
        var owndata = True
        try:
            owndata = self.flags["OWNDATA"]
        except:
            print("Invalid `OWNDATA` flag. Treat as `True`.")
        if owndata:
            self._buf.ptr.free()

    # ===-------------------------------------------------------------------===#
    # Indexing and slicing
    # Getter and setter dunders and other methods
    # ===-------------------------------------------------------------------===#

    fn _setitem(self, *indices: Int, val: Scalar[dtype]):
        """
        (UNSAFE! for internal use only.)
        Get item at indices and bypass all boundary checks.

        ```mojo
        import numojo
        var A = numojo.ones(numojo.Shape(2,3,4))
        A._setitem(1,2,3, val=10)
        ```
        """
        var index_of_buffer: Int = 0
        for i in range(self.ndim):
            index_of_buffer += indices[i] * self.strides._buf[i]
        self._buf.ptr[index_of_buffer] = val

    fn __setitem__(mut self, idx: Int, val: Self) raises:
        """
        Set a slice of array with given array.

        Example:
        ```mojo
        import numojo as nm
        var A = nm.random.rand[nm.i16](3, 2)
        var B = nm.random.rand[nm.i16](3)
        A[1:4] = B
        ```
        """
        if self.ndim == 0 and val.ndim == 0:
            self._buf.ptr.store(0, val._buf.ptr.load(0))

        var slice_list = List[Slice]()
        if idx >= self.shape[0]:
            var message = String(
                "Error: Slice value exceeds the array shape!\n"
                "The {}-th dimension is of size {}.\n"
                "The slice goes from {} to {}"
            ).format(
                0,
                self.shape[0],
                idx,
                idx + 1,
            )
            raise Error(message)
        slice_list.append(Slice(idx, idx + 1, 1))
        if self.ndim > 1:
            for i in range(1, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim, 1))

        var n_slices: Int = len(slice_list)
        var ndims: Int = 0
        var count: Int = 0
        var spec: List[Int] = List[Int]()
        for i in range(n_slices):
            if slice_list[i].step is None:
                raise Error(String("Step of slice is None."))
            var slice_len: Int = (
                (slice_list[i].end.value() - slice_list[i].start.value())
                / slice_list[i].step.or_else(1)
            ).__int__()
            spec.append(slice_len)
            if slice_len != 1:
                ndims += 1
            else:
                count += 1
        if count == slice_list.__len__():
            ndims = 1

        var nshape: List[Int] = List[Int]()
        var ncoefficients: List[Int] = List[Int]()
        var nstrides: List[Int] = List[Int]()
        var nnum_elements: Int = 1

        var j: Int = 0
        count = 0
        for _ in range(ndims):
            while spec[j] == 1:
                count += 1
                j += 1
            if j >= self.ndim:
                break
            var slice_len: Int = (
                (slice_list[j].end.value() - slice_list[j].start.value())
                / slice_list[j].step.or_else(1)
            ).__int__()
            nshape.append(slice_len)
            nnum_elements *= slice_len
            ncoefficients.append(
                self.strides[j] * slice_list[j].step.or_else(1)
            )
            j += 1

        # TODO: We can remove this check after we have support for broadcasting
        for i in range(ndims):
            if nshape[i] != val.shape[i]:
                var message = String(
                    "Error: Shape mismatch!\n"
                    "Cannot set the array values with given array.\n"
                    "The {}-th dimension of the array is of shape {}.\n"
                    "The {}-th dimension of the value is of shape {}."
                ).format(nshape[i], val.shape[i])
                raise Error(message)

        var noffset: Int = 0
        if self.flags["C_CONTIGUOUS"]:
            noffset = 0
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]
        elif self.flags["F_CONTIGUOUS"]:
            noffset = 0
            nstrides.append(1)
            for i in range(0, ndims - 1):
                nstrides.append(nstrides[i] * nshape[i])
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]

        var index = List[Int]()
        for _ in range(ndims):
            index.append(0)

        _traverse_iterative_setter[dtype](
            val, self, nshape, ncoefficients, nstrides, noffset, index
        )

    fn __setitem__(mut self, index: Item, val: Scalar[dtype]) raises:
        """
        Set the value at the index list.
        """
        if index.__len__() != self.ndim:
            var message = String(
                "Error: Length of `index` does not match the number of"
                " dimensions!\n"
                "Length of indices is {}.\n"
                "The array dimension is {}."
            ).format(index.__len__(), self.ndim)
            raise Error(message)
        for i in range(index.__len__()):
            if index[i] >= self.shape[i]:
                var message = String(
                    "Error: `index` exceeds the size!\n"
                    "For {}-th dimension:\n"
                    "The index value is {}.\n"
                    "The size of the corresponding dimension is {}"
                ).format(i, index[i], self.shape[i])
                raise Error(message)
        var idx: Int = _get_offset(index, self.strides)
        self._buf.ptr.store(idx, val)

    # only works if array is called as array.__setitem__(), mojo compiler doesn't parse it implicitly
    fn __setitem__(
        mut self, mask: NDArray[DType.bool], value: Scalar[dtype]
    ) raises:
        """
        Set the value of the array at the indices where the mask is true.
        """
        if (
            mask.shape != self.shape
        ):  # this behavious could be removed potentially
            raise Error("Mask and array must have the same shape")

        for i in range(mask.size):
            if mask._buf.ptr.load[width=1](i):
                self._buf.ptr.store(i, value)

    fn __setitem__(mut self, *slices: Slice, val: Self) raises:
        """
        Retreive slices of an array from variadic slices.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).
        """
        var slice_list: List[Slice] = List[Slice]()
        for i in range(slices.__len__()):
            slice_list.append(slices[i])
        self.__setitem__(slices=slice_list, val=val)

    fn __setitem__(mut self, slices: List[Slice], val: Self) raises:
        """
        Sets the slices of an array from list of slices and array.

        Example:
        ```console
        >>> var a = nm.arange[i8](16).reshape(Shape(4, 4))
        print(a)
        [[      0       1       2       3       ]
         [      4       5       6       7       ]
         [      8       9       10      11      ]
         [      12      13      14      15      ]]
        2-D array  Shape: [4, 4]  DType: int8  C-cont: True  F-cont: False  own data: True
        >>> a[2:4, 2:4] = a[0:2, 0:2]
        print(a)
        [[      0       1       2       3       ]
         [      4       5       6       7       ]
         [      8       9       0       1       ]
         [      12      13      4       5       ]]
        2-D array  Shape: [4, 4]  DType: int8  C-cont: True  F-cont: False  own data: True
        ```
        """
        var n_slices: Int = len(slices)
        var ndims: Int = 0
        var count: Int = 0
        var spec: List[Int] = List[Int]()
        var slice_list: List[Slice] = self._adjust_slice(slices)
        for i in range(n_slices):
            if (
                slice_list[i].start.value() >= self.shape[i]
                or slice_list[i].end.value() > self.shape[i]
            ):
                var message = String(
                    "Error: Slice value exceeds the array shape!\n"
                    "The {}-th dimension is of size {}.\n"
                    "The slice goes from {} to {}"
                ).format(
                    i,
                    self.shape[i],
                    slice_list[i].start.value(),
                    slice_list[i].end.value(),
                )
                raise Error(message)
            # if slice_list[i].step is None:
            #     raise Error(String("Step of slice is None."))
            var slice_len: Int = (
                (slice_list[i].end.value() - slice_list[i].start.value())
                / slice_list[i].step.or_else(1)
            ).__int__()
            spec.append(slice_len)
            if slice_len != 1:
                ndims += 1
            else:
                count += 1
        if count == slice_list.__len__():
            ndims = 1

        var nshape: List[Int] = List[Int]()
        var ncoefficients: List[Int] = List[Int]()
        var nstrides: List[Int] = List[Int]()
        var nnum_elements: Int = 1

        var j: Int = 0
        count = 0
        for _ in range(ndims):
            while spec[j] == 1:
                count += 1
                j += 1
            if j >= self.ndim:
                break
            var slice_len: Int = (
                (slice_list[j].end.value() - slice_list[j].start.value())
                / slice_list[j].step.or_else(1)
            ).__int__()
            nshape.append(slice_len)
            nnum_elements *= slice_len
            ncoefficients.append(
                self.strides[j] * slice_list[j].step.or_else(1)
            )
            j += 1

        # TODO: We can remove this check after we have support for broadcasting
        for i in range(ndims):
            if nshape[i] != val.shape[i]:
                var message = String(
                    "Error: Shape mismatch!\n"
                    "For {}-th dimension: \n"
                    "The size of the array is {}.\n"
                    "The size of the input value is {}."
                ).format(i, nshape[i], val.shape[i])
                raise Error(message)

        var noffset: Int = 0
        if self.flags["C_CONTIGUOUS"]:
            noffset = 0
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):  # temp
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]
        elif self.flags["F_CONTIGUOUS"]:
            noffset = 0
            nstrides.append(1)
            for i in range(0, ndims - 1):
                nstrides.append(nstrides[i] * nshape[i])
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]

        var index = List[Int]()
        for _ in range(ndims):
            index.append(0)

        _traverse_iterative_setter[dtype](
            val, self, nshape, ncoefficients, nstrides, noffset, index
        )

    fn __setitem__(mut self, *slices: Variant[Slice, Int], val: Self) raises:
        """
        Get items by a series of either slices or integers.

        Example:
        ```console
        >>> var a = nm.arange[i8](16).reshape(Shape(4, 4))
        print(a)
        [[      0       1       2       3       ]
         [      4       5       6       7       ]
         [      8       9       10      11      ]
         [      12      13      14      15      ]]
        2-D array  Shape: [4, 4]  DType: int8  C-cont: True  F-cont: False  own data: True
        >>> a[0, Slice(2, 4)] = a[3, Slice(0, 2)]
        print(a)
        [[      0       1       12      13      ]
         [      4       5       6       7       ]
         [      8       9       10      11      ]
         [      12      13      14      15      ]]
        2-D array  Shape: [4, 4]  DType: int8  C-cont: True  F-cont: False  own data: True
        ```
        """
        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim:
            raise Error("Error: No of slices greater than rank of array")
        var slice_list: List[Slice] = List[Slice]()

        var count_int = 0
        for i in range(len(slices)):
            if slices[i].isa[Slice]():
                slice_list.append(slices[i]._get_ptr[Slice]()[0])
            elif slices[i].isa[Int]():
                count_int += 1
                var int: Int = slices[i]._get_ptr[Int]()[0]
                slice_list.append(Slice(int, int + 1, 1))

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim, 1))

        self.__setitem__(slices=slice_list, val=val)

    # TODO: fix this setter, add bound checks. Not sure about it's use case.
    fn __setitem__(self, index: NDArray[DType.index], val: NDArray) raises:
        """
        Returns the items of the array from an array of indices.

        Refer to `__getitem__(self, index: List[Int])`.

        Example:
        ```console
        > var X = nm.NDArray[nm.i8](3,random=True)
        > print(X)
        [       32      21      53      ]
        1-D array  Shape: [3]  DType: int8
        > print(X.argsort())
        [       1       0       2       ]
        1-D array  Shape: [3]  DType: index
        > print(X[X.argsort()])
        [       21      32      53      ]
        1-D array  Shape: [3]  DType: int8
        ```
        """

        for i in range(len(index)):
            self.store(int(index.load(i)), rebind[Scalar[dtype]](val.load(i)))

    fn __setitem__(
        mut self, mask: NDArray[DType.bool], val: NDArray[dtype]
    ) raises:
        """
        Set the value of the array at the indices where the mask is true.

        Example:
        ```
        var A = numojo.core.NDArray[numojo.i16](6, random=True)
        var mask = A > 0
        print(A)
        print(mask)
        A[mask] = 0
        print(A)
        ```
        """
        if (
            mask.shape != self.shape
        ):  # this behavious could be removed potentially
            var message = String(
                "Shape of mask does not match the shape of array."
            )
            raise Error(message)

        for i in range(mask.size):
            if mask._buf.ptr.load(i):
                self._buf.ptr.store(i, val._buf.ptr.load(i))

    # ===-------------------------------------------------------------------===#
    # Getter dunders and other getter methods
    #
    # INDEXING: to get a scalar from array.
    # fn _getitem(self, *indices: Int) -> Scalar[dtype]
    # fn __getitem__(self, index: Item) raises -> SIMD[dtype, 1]
    #
    # SLICING: to get a slice of array.
    # fn __getitem__(self, idx: Int) raises -> Self
    # fn __getitem__(self, *slices: Slice) raises -> Self
    # fn __getitem__(self, slice_list: List[Slice]) raises -> Self
    # fn __getitem__(self, *slices: Variant[Slice, Int]) raises -> Self
    #
    # SLICING: to get a slice of array from index or mask.
    # fn __getitem__(self, index: List[Int]) raises -> Self
    # fn __getitem__(self, index: NDArray[index]) raises -> Self
    # fn __getitem__(self, mask: List[Bool]) raises -> Self
    # fn __getitem__(self, mask: NDArray[bool]) raises -> Self
    # ===-------------------------------------------------------------------===#

    fn _getitem(self, *indices: Int) -> Scalar[dtype]:
        """
        (UNSAFE! for internal use only.)
        Get item at indices and bypass all boundary checks.

        ```mojo
        import numojo
        var A = numojo.ones(numojo.Shape(2,3,4))
        print(A._getitem(1,2,3))
        ```
        """
        var index_of_buffer: Int = 0
        for i in range(self.ndim):
            index_of_buffer += indices[i] * self.strides._buf[i]
        return self._buf.ptr[index_of_buffer]

    fn __getitem__(self, index: Item) raises -> SIMD[dtype, 1]:
        """
        Set the value at the index list.
        """
        if index.__len__() != self.ndim:
            var message = String(
                "Error: Length of `index` do not match the number of"
                " dimensions!\n"
                "Length of indices is {}.\n"
                "The number of dimensions is {}."
            ).format(index.__len__(), self.ndim)
            raise Error(message)
        for i in range(index.__len__()):
            if index[i] >= self.shape[i]:
                var message = String(
                    "Error: `index` exceeds the size!\n"
                    "For {}-the mension:\n"
                    "The index is {}.\n"
                    "The size of the dimensions is {}"
                ).format(i, index[i], self.shape[i])
                raise Error(message)
        var idx: Int = _get_offset(index, self.strides)
        return self._buf.ptr.load[width=1](idx)

    fn __getitem__(self, idx: Int) raises -> Self:
        """
        Retreive a slice of the array corresponding to the index at the first dimension.

        Example:
            `arr[1]` returns the second row of the array.
        """

        var slice_list = List[Slice]()
        slice_list.append(Slice(idx, idx + 1, 1))

        # 0-d array always return itself
        if self.ndim == 0:
            return self

        if self.ndim > 1:
            for i in range(1, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim, 1))

        var narr: Self = self.__getitem__(slice_list)

        if self.ndim == 1:
            narr.ndim = 0
            narr.shape._buf[0] = 0

        return narr

    fn __getitem__(self, owned *slices: Slice) raises -> Self:
        """
        Retreive slices of an array from variadic slices.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).
        """

        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim:
            raise Error("Error: No of slices exceed the array dimensions.")
        var slice_list: List[Slice] = List[Slice]()
        for i in range(len(slices)):
            slice_list.append(slices[i])

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                slice_list.append(Slice(0, self.shape[i], 1))

        var narr: Self = self[slice_list]
        return narr

    fn __getitem__(self, owned slice_list: List[Slice]) raises -> Self:
        """
        Retreive slices of an array from list of slices.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).
        """

        var n_slices: Int = slice_list.__len__()
        if n_slices > self.ndim or n_slices < self.ndim:
            raise Error("Error: No of slices do not match shape")

        var ndims: Int = 0
        var spec: List[Int] = List[Int]()
        var count: Int = 0

        var slices: List[Slice] = self._adjust_slice(slice_list)
        for i in range(slices.__len__()):
            if (
                slices[i].start.value() >= self.shape[i]
                or slices[i].end.value() > self.shape[i]
            ):
                raise Error("Error: Slice value exceeds the array shape")
            var slice_len: Int = len(
                range(
                    slices[i].start.value(),
                    slices[i].end.value(),
                    slices[i].step.or_else(1),
                )
            )
            spec.append(slice_len)
            if slice_len != 1:
                ndims += 1
            else:
                count += 1
        if count == slices.__len__():
            ndims = 1

        var nshape: List[Int] = List[Int]()
        var ncoefficients: List[Int] = List[Int]()
        var nstrides: List[Int] = List[Int]()
        var nnum_elements: Int = 1

        var j: Int = 0
        count = 0
        for _ in range(ndims):
            while spec[j] == 1:
                count += 1
                j += 1
            if j >= self.ndim:
                break
            var slice_len: Int = len(
                range(
                    slices[j].start.value(),
                    slices[j].end.value(),
                    slices[j].step.or_else(1),
                )
            )
            nshape.append(slice_len)
            nnum_elements *= slice_len
            ncoefficients.append(self.strides[j] * slices[j].step.value())
            j += 1

        if count == slices.__len__():
            nshape.append(1)
            nnum_elements = 1
            ncoefficients.append(1)

        var noffset: Int = 0
        if self.flags["C_CONTIGUOUS"]:
            noffset = 0
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):  # temp
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slices.__len__()):
                noffset += slices[i].start.value() * self.strides[i]

        elif self.flags["F_CONTIGUOUS"]:
            noffset = 0
            nstrides.append(1)
            for i in range(0, ndims - 1):
                nstrides.append(nstrides[i] * nshape[i])
            for i in range(slices.__len__()):
                noffset += slices[i].start.value() * self.strides[i]

        var narr = Self(
            offset=noffset,
            shape=nshape,
            strides=nstrides,
        )

        var index = List[Int]()
        for _ in range(ndims):
            index.append(0)

        _traverse_iterative[dtype](
            self, narr, nshape, ncoefficients, nstrides, noffset, index, 0
        )

        return narr

    fn __getitem__(self, owned *slices: Variant[Slice, Int]) raises -> Self:
        """
        Get items by a series of either slices or integers.

        A decrease of dimensions may or may not happen when `__getitem__` is
        called on an ndarray. An ndarray of X-D array can become Y-D array after
        `__getitem__` where `Y <= X`.

        Whether the dimension decerases or not depends on:
        1. What types of arguments are passed into `__getitem__`.
        2. The number of arguments that are passed in `__getitem__`.

        PRINCIPAL: The number of dimensions to be decreased is determined by
        the number of `Int` passed in `__getitem__`.

        For example, `A` is a 10x10x10 ndarray (3-D). Then,

        - `A[1, 2, 3]` leads to a 0-D array (scalar), since there are 3 integers.
        - `A[1, 2]` leads to a 1-D array (vector), since there are 2 integers,
        so the dimension decreases by 2.
        - `A[1]` leads to a 2-D array (matrix), since there is 1 integer, so the
        dimension decreases by 1.

        The number of dimensions will not decrease when Slice is passed in
        `__getitem__` or no argument is passed in for a certain dimension
        (it is an implicit slide and a slide of all items will be used).

        Take the same example `A` with 10x10x10 in shape. Then,

        - `A[1:4, 2:5, 3:6]`, leads to a 3-D array (no decrease in dimension),
        since there are 3 slices.
        - `A[2:8]`, leads to a 3-D array (no decrease in dimension), since there
        are 1 explicit slice and 2 implicit slices.

        When there is a mixture of int and slices passed into `__getitem__`,
        the number of integers will be the number of dimensions to be decreased.
        Example,

        - `A[1:4, 2, 2]`, leads to a 1-D array (vector), since there are 2
        integers, so the dimension decreases by 2.

        Note that, even though a slice contains one row, it does not reduce the
        dimensions. Example,

        - `A[1:2, 2:3, 3:4]`, leads to a 3-D array (no decrease in dimension),
        since there are 3 slices.

        Note that, when the number of integers equals to the number of
        dimensions, the final outcome is an 0-D array instead of a number.
        The user has to upack the 0-D array with the method`A.item(0)` to get the
        corresponding number.
        This behavior is different from numpy where the latter returns a number.

        More examples for 1-D, 2-D, and 3-D arrays.

        ```console
        A is a matrix
        [[      -128    -95     65      -11     ]
        [      8       -72     -116    45      ]
        [      45      111     -30     4       ]
        [      84      -120    -115    7       ]]
        2-D array  Shape: [4, 4]  DType: int8

        A[0]
        [       -128    -95     65      -11     ]
        1-D array  Shape: [4]  DType: int8

        A[0, 1]
        -95
        0-D array  Shape: [0]  DType: int8

        A[Slice(1,3)]
        [[      8       -72     -116    45      ]
        [      45      111     -30     4       ]]
        2-D array  Shape: [2, 4]  DType: int8

        A[1, Slice(2,4)]
        [       -116    45      ]
        1-D array  Shape: [2]  DType: int8

        A[Slice(1,3), Slice(1,3)]
        [[      -72     -116    ]
        [      111     -30     ]]
        2-D array  Shape: [2, 2]  DType: int8

        A.item(0,1) as Scalar
        -95

        ==============================
        A is a vector
        [       43      -127    -30     -111    ]
        1-D array  Shape: [4]  DType: int8

        A[0]
        43
        0-D array  Shape: [0]  DType: int8

        A[Slice(1,3)]
        [       -127    -30     ]
        1-D array  Shape: [2]  DType: int8

        A.item(0) as Scalar
        43

        ==============================
        A is a 3darray
        [[[     -22     47      22      110     ]
        [     88      6       -105    39      ]
        [     -22     51      105     67      ]
        [     -61     -116    60      -44     ]]
        [[     33      65      125     -35     ]
        [     -65     123     57      64      ]
        [     38      -110    33      98      ]
        [     -59     -17     68      -6      ]]
        [[     -68     -58     -37     -86     ]
        [     -4      101     104     -113    ]
        [     103     1       4       -47     ]
        [     124     -2      -60     -105    ]]
        [[     114     -110    0       -30     ]
        [     -58     105     7       -10     ]
        [     112     -116    66      69      ]
        [     83      -96     -124    48      ]]]
        3-D array  Shape: [4, 4, 4]  DType: int8

        A[0]
        [[      -22     47      22      110     ]
        [      88      6       -105    39      ]
        [      -22     51      105     67      ]
        [      -61     -116    60      -44     ]]
        2-D array  Shape: [4, 4]  DType: int8

        A[0, 1]
        [       88      6       -105    39      ]
        1-D array  Shape: [4]  DType: int8

        A[0, 1, 2]
        -105
        0-D array  Shape: [0]  DType: int8

        A[Slice(1,3)]
        [[[     33      65      125     -35     ]
        [     -65     123     57      64      ]
        [     38      -110    33      98      ]
        [     -59     -17     68      -6      ]]
        [[     -68     -58     -37     -86     ]
        [     -4      101     104     -113    ]
        [     103     1       4       -47     ]
        [     124     -2      -60     -105    ]]]
        3-D array  Shape: [2, 4, 4]  DType: int8

        A[1, Slice(2,4)]
        [[      38      -110    33      98      ]
        [      -59     -17     68      -6      ]]
        2-D array  Shape: [2, 4]  DType: int8

        A[Slice(1,3), Slice(1,3), 2]
        [[      57      33      ]
        [      104     4       ]]
        2-D array  Shape: [2, 2]  DType: int8

        A.item(0,1,2) as Scalar
        -105
        ```

        Args:
            slices: A series of either Slice or Int.

        Returns:
            An ndarray with a smaller or equal dimension of the original one.
        """

        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim:
            raise Error(
                String(
                    "Error: number of slices {} \n"
                    "is greater than number of dimension of array {}!"
                ).format(n_slices, self.ndim)
            )
        var slice_list: List[Slice] = List[Slice]()

        var count_int = 0  # Count the number of Int in the argument

        for i in range(len(slices)):
            if slices[i].isa[Slice]():
                slice_list.append(slices[i]._get_ptr[Slice]()[0])
            elif slices[i].isa[Int]():
                count_int += 1
                var int: Int = slices[i]._get_ptr[Int]()[0]
                slice_list.append(Slice(int, int + 1, 1))

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim, 1))

        var narr: Self = self.__getitem__(slice_list)

        if count_int == self.ndim:
            narr.ndim = 0
            narr.shape._buf[0] = 0

        return narr

    fn __getitem__(self, indices: NDArray[DType.index]) raises -> Self:
        """
        Get items from 0-th dimension of an ndarray of indices.

        If the original array is of shape (i,j,k) and
        the indices array is of shape (l,m,n), then the output array
        will be of shape (l,m,n,j,k).

        Example:
        ```console
        >>>var a = nm.arange[i8](6)
        >>>print(a)
        [       0       1       2       3       4       5       ]
        1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True
        >>>print(a[nm.array[isize]("[4, 2, 5, 1, 0, 2]")])
        [       4       2       5       1       0       2       ]
        1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True

        var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
        print(b)
        [[[     0       1       2       ]
          [     3       4       5       ]]
         [[     6       7       8       ]
          [     9       10      11      ]]]
        3-D array  Shape: [2, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
        print(b[nm.array[isize]("[2, 0, 1]")])
        [[[     0       0       0       ]
          [     0       67      95      ]]
         [[     0       1       2       ]
          [     3       4       5       ]]
         [[     6       7       8       ]
          [     9       10      11      ]]]
        3-D array  Shape: [3, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
        ```

        Args:
            indices: Array of intable values.

        Returns:
            NDArray with items from the array of indices.
        """

        # Get the shape of resulted array
        var shape = NDArrayShape.join(indices.shape, self.shape._pop(0))

        var result = NDArray[dtype](shape)
        var size_per_item = self.size // self.shape[0]

        # Fill in the values
        for i in range(indices.size):
            if indices.item(i) >= self.shape[0]:
                raise Error(
                    String(
                        "index {} with value {} is out of boundary [0, {})"
                    ).format(i, indices.item(i), self.shape[0])
                )
            memcpy(
                result._buf.ptr + i * size_per_item,
                self._buf.ptr + indices.item(i) * size_per_item,
                size_per_item,
            )

        return result

    fn __getitem__(self, indices: List[Int]) raises -> Self:
        # TODO: Use trait IntLike when it is supported by Mojo.
        """
        Get items from 0-th dimension of an array. It is an overload of
        `__getitem__(self, indices: NDArray[DType.index]) raises -> Self`.

        Example:
        ```console
        >>>var a = nm.arange[i8](6)
        >>>print(a)
        [       0       1       2       3       4       5       ]
        1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True
        >>>print(a[List[Int](4, 2, 5, 1, 0, 2)])
        [       4       2       5       1       0       2       ]
        1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True

        var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
        print(b)
        [[[     0       1       2       ]
          [     3       4       5       ]]
         [[     6       7       8       ]
          [     9       10      11      ]]]
        3-D array  Shape: [2, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
        print(b[List[Int](2, 0, 1)])
        [[[     0       0       0       ]
          [     0       67      95      ]]
         [[     0       1       2       ]
          [     3       4       5       ]]
         [[     6       7       8       ]
          [     9       10      11      ]]]
        3-D array  Shape: [3, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
        ```

        Args:
            indices: A list of Int.

        Returns:
            NDArray with items from the list of indices.
        """

        var indices_array = NDArray[DType.index](shape=Shape(len(indices)))
        for i in range(len(indices)):
            (indices_array._buf.ptr + i).init_pointee_copy(indices[i])

        return self[indices_array]

    fn __getitem__(self, mask: NDArray[DType.bool]) raises -> Self:
        # TODO: Extend the mask into multiple dimensions.
        """
        Get item from an array according to a mask array.

        If array shape is equal to mask shape, it returns a flattened array of
        the values where mask is True.

        If array shape is not equal to mask shape, it returns items from the
        0-th dimension of the array where mask is True.

        Example:
        ```console
        >>>var a = nm.arange[i8](6)
        >>>print(a)
        [       0       1       2       3       4       5       ]
        1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True
        >>>print(a[nm.array[boolean]("[1,0,1,1,0,1]")])
        [       0       2       3       5       ]
        1-D array  Shape: [4]  DType: int8  C-cont: True  F-cont: True  own data: True

        var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
        print(b)
        [[[     0       1       2       ]
          [     3       4       5       ]]
         [[     6       7       8       ]
          [     9       10      11      ]]]
        3-D array  Shape: [2, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
        >>>print(b[nm.array[boolean]("[0,1]")])
        [[[     6       7       8       ]
          [     9       10      11      ]]]
        3-D array  Shape: [1, 2, 3]  DType: int8  C-cont: True  F-cont: True  own data: True
        ```

        Args:
            mask: NDArray with Dtype.bool.

        Returns:
            NDArray with items from the mask.
        """

        # CASE 1:
        # if array shape is equal to mask shape,
        # return a flattened array of the values where mask is True
        if mask.shape == self.shape:
            var len_of_result = 0

            # Count number of True
            for i in range(mask.size):
                if mask.item(i):
                    len_of_result += 1

            # Change the first number of the ndshape
            var result = NDArray[dtype](shape=NDArrayShape(len_of_result))

            # Fill in the values
            var offset = 0
            for i in range(mask.size):
                if mask.item(i):
                    (result._buf.ptr + offset).init_pointee_copy(
                        self._buf.ptr[i]
                    )
                    offset += 1

            return result

        # CASE 2:
        # if array shape is not equal to mask shape,
        # return items from the 0-th dimension of the array where mask is True
        if mask.ndim > 1:
            raise Error(String("Currently we only support 1-d mask array."))

        if mask.shape[0] != self.shape[0]:
            raise Error(
                String(
                    "Shape 0 of mask ({}) does not match that of array ({})."
                ).format(mask.shape[0], self.shape[0])
            )

        var len_of_result = 0

        # Count number of True
        for i in range(mask.size):
            if mask.item(i):
                len_of_result += 1

        # Change the first number of the ndshape
        var shape = self.shape
        shape._buf[0] = len_of_result

        var result = NDArray[dtype](shape)
        var size_per_item = self.size // self.shape[0]

        # Fill in the values
        var offset = 0
        for i in range(mask.size):
            if mask.item(i):
                memcpy(
                    result._buf.ptr + offset * size_per_item,
                    self._buf.ptr + i * size_per_item,
                    size_per_item,
                )
                offset += 1

        return result

    fn __getitem__(self, mask: List[Bool]) raises -> Self:
        """
        Get items from 0-th dimension of an array according to mask.
        __getitem__(self, mask: NDArray[DType.bool]) raises -> Self.

        Example:
        ```console
        >>>var a = nm.arange[i8](6)
        >>>print(a)
        [       0       1       2       3       4       5       ]
        1-D array  Shape: [6]  DType: int8  C-cont: True  F-cont: True  own data: True
        >>>print(a[List[Bool](True, False, True, True, False, True)])
        [       0       2       3       5       ]
        1-D array  Shape: [4]  DType: int8  C-cont: True  F-cont: True  own data: True

        var b = nm.arange[i8](12).reshape(Shape(2, 2, 3))
        print(b)
        [[[     0       1       2       ]
          [     3       4       5       ]]
         [[     6       7       8       ]
          [     9       10      11      ]]]
        3-D array  Shape: [2, 2, 3]  DType: int8  C-cont: True  F-cont: False  own data: True
        >>>print(b[List[Bool](False, True)])
        [[[     6       7       8       ]
          [     9       10      11      ]]]
        3-D array  Shape: [1, 2, 3]  DType: int8  C-cont: True  F-cont: True  own data: True
        ```

        Args:
            mask: A list of boolean values.

        Returns:
            NDArray with items from the mask.
        """

        var mask_array = NDArray[DType.bool](shape=Shape(len(mask)))
        for i in range(len(mask)):
            (mask_array._buf.ptr + i).init_pointee_copy(mask[i])

        return self[mask_array]

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    # TODO: We should make a version that checks nonzero/not_nan
    fn __bool__(self) raises -> Bool:
        """
        If all true return true.
        """
        if self.dtype == DType.bool:
            if self.all():
                return True
            else:
                return False
        raise Error(
            "core:ndarray:NDArray:__bool__: Bool is currently only implemented"
            " for DType.bool"
        )

    fn __int__(self) raises -> Int:
        """Get Int representation of the array.

        Similar to Numpy, only 0-D arrays or length-1 arrays can be converted to
        scalars.

        Example:
        ```console
        > var A = NDArray[dtype](6, random=True)
        > print(int(A))

        Unhandled exception caught during execution: Only 0-D arrays or length-1 arrays can be converted to scalars
        mojo: error: execution exited with a non-zero result: 1

        > var B = NDArray[dtype](1, 1, random=True)
        > print(int(B))
        14
        ```

        Returns:
            Int representation of the array.

        """
        if (self.size == 1) or (self.ndim == 0):
            return int(self.load(0))
        else:
            raise (
                "Only 0-D arrays or length-1 arrays can be converted to scalars"
            )

    fn __pos__(self) raises -> Self:
        """
        Unary positve returns self unless boolean type.
        """
        if self.dtype is DType.bool:
            raise Error(
                "ndarray:NDArrray:__pos__: pos does not accept bool type arrays"
            )
        return self

    fn __neg__(self) raises -> Self:
        """
        Unary negative returns self unless boolean type.

        For bolean use `__invert__`(~)
        """
        if self.dtype is DType.bool:
            raise Error(
                "ndarray:NDArrray:__pos__: pos does not accept bool type arrays"
            )
        return self * Scalar[dtype](-1.0)

    # maybe they don't need conversion with astype.
    @always_inline("nodebug")
    fn __eq__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: NDArray[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence.
        """
        return comparison.equal[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    @always_inline("nodebug")
    fn __eq__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: Scalar[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence.
        """
        return comparison.equal[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence.
        """
        return comparison.equal[dtype](self, other)

    @always_inline("nodebug")
    fn __eq__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise equivalence between scalar and Array.
        """
        return comparison.equal[dtype](self, other)

    @always_inline("nodebug")
    fn __ne__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: NDArray[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise nonequivelence.
        """
        return comparison.not_equal[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    @always_inline("nodebug")
    fn __ne__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: Scalar[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise nonequivelence between scalar and Array.
        """
        return comparison.not_equal[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    @always_inline("nodebug")
    fn __ne__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise nonequivelence.
        """
        return comparison.not_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __ne__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise nonequivelence between scalar and Array.
        """
        return comparison.not_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __lt__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: Scalar[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than.
        """
        return comparison.less[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    @always_inline("nodebug")
    fn __lt__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: NDArray[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than between scalar and Array.
        """
        return comparison.less[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    @always_inline("nodebug")
    fn __lt__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than.
        """
        return comparison.less[dtype](self, other)

    @always_inline("nodebug")
    fn __lt__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than between scalar and Array.
        """
        return comparison.less[dtype](self, other)

    @always_inline("nodebug")
    fn __le__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: Scalar[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to.
        """
        return comparison.less_equal[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    @always_inline("nodebug")
    fn __le__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: NDArray[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        return comparison.less_equal[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    @always_inline("nodebug")
    fn __le__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to.
        """
        return comparison.less_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __le__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        return comparison.less_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __gt__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: Scalar[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than.
        """
        return comparison.greater[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    @always_inline("nodebug")
    fn __gt__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: NDArray[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than between scalar and Array.
        """
        return comparison.greater[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    @always_inline("nodebug")
    fn __gt__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than.
        """
        return comparison.greater[dtype](self, other)

    @always_inline("nodebug")
    fn __gt__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than between scalar and Array.
        """
        return comparison.greater[dtype](self, other)

    @always_inline("nodebug")
    fn __ge__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: Scalar[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than or equal to.
        """
        return comparison.greater_equal[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    @always_inline("nodebug")
    fn __ge__[
        OtherDtype: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDtype](),
    ](self, other: NDArray[OtherDtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        return comparison.greater_equal[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    @always_inline("nodebug")
    fn __ge__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than or equal to.
        """
        return comparison.greater_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __ge__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        return comparison.greater_equal[dtype](self, other)

    fn __add__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array + scalar`.
        """
        return math.add[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    """ ARITHMETIC OPERATORS """

    fn __add__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: NDArray[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array + array`.
        """
        return math.add[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    fn __add__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `array + scalar`.
        """
        return math.add[dtype](self, other)

    fn __add__(self, other: Self) raises -> Self:
        """
        Enables `array + array`.
        """
        return math.add[dtype](self, other)

    fn __radd__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `scalar + array`.
        """
        return math.add[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __radd__(mut self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar + array`.
        """
        return math.add[dtype](self, other)

    # TODO make an inplace version of arithmetic functions for the i dunders
    # Cannot do type coercion for iadd
    fn __iadd__(mut self, other: SIMD[dtype, 1]) raises:
        """
        Enables `array += scalar`.
        """
        self = _af.math_func_one_array_one_SIMD_in_one_array_out[
            dtype, SIMD.__add__
        ](self, other)

    fn __iadd__(mut self, other: Self) raises:
        """
        Enables `array *= array`.
        """
        self = _af.math_func_2_array_in_one_array_out[dtype, SIMD.__add__](
            self, other
        )

    fn __sub__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array - scalar`.
        """
        return math.sub[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __sub__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: NDArray[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array - array`.
        """
        return math.sub[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    fn __sub__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `array - scalar`.
        """
        return math.sub[dtype](self, other)

    fn __sub__(self, other: Self) raises -> Self:
        """
        Enables `array - array`.
        """
        return math.sub[dtype](self, other)

    fn __rsub__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `scalar - array`.
        """
        return math.sub[ResultDType](
            other.cast[ResultDType](), self.astype[ResultDType]()
        )

    fn __rsub__(mut self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar - array`.
        """
        return math.sub[dtype](other, self)

    fn __isub__(mut self, other: SIMD[dtype, 1]) raises:
        """
        Enables `array -= scalar`.
        """
        self = self - other

    fn __isub__(mut self, other: Self) raises:
        """
        Enables `array -= array`.
        """
        self = self - other

    fn __matmul__(self, other: Self) raises -> Self:
        return matmul(self, other)

    fn __mul__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array * scalar`.
        """
        return math.mul[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __mul__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: NDArray[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array * array`.
        """
        return math.mul[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    fn __mul__(self, other: Scalar[dtype]) raises -> Self:
        """
        Enables `array * scalar`.
        """
        return math.mul[dtype](self, other)

    fn __mul__(self, other: Self) raises -> Self:
        """
        Enables `array * array`.
        """
        return math.mul[dtype](self, other)

    fn __rmul__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `scalar * array`.
        """
        return math.mul[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __rmul__(mut self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar * array`.
        """
        return math.mul[dtype](self, other)

    fn __imul__(mut self, other: SIMD[dtype, 1]) raises:
        """
        Enables `array *= scalar`.
        """
        self = self * other

    fn __imul__(mut self, other: Self) raises:
        """
        Enables `array *= array`.
        """
        self = self * other

    fn __abs__(self) -> Self:
        return abs(self)

    fn __invert__(self) raises -> Self:
        """
        Element-wise inverse (~ or not), only for bools and integral types.
        """
        return bitwise.invert[dtype](self)

    fn __pow__(self, p: Int) -> Self:
        return self._elementwise_pow(p)

    fn __pow__(self, p: Self) raises -> Self:
        if self.size != p.size:
            raise Error(
                String(
                    "Both arrays must have same number of elements! \n"
                    "Self array has {} elements. \n"
                    "Other array has {} elements"
                ).format(self.size, p.size)
            )

        var result = Self(self.shape)

        @parameter
        fn vectorized_pow[simd_width: Int](index: Int) -> None:
            result._buf.ptr.store(
                index,
                self._buf.ptr.load[width=simd_width](index)
                ** p._buf.ptr.load[width=simd_width](index),
            )

        vectorize[vectorized_pow, self.width](self.size)
        return result

    fn __ipow__(mut self, p: Int):
        self = self.__pow__(p)

    fn _elementwise_pow(self, p: Int) -> Self:
        var new_vec = self

        @parameter
        fn array_scalar_vectorize[simd_width: Int](index: Int) -> None:
            new_vec._buf.ptr.store(
                index,
                builtin_math.pow(
                    self._buf.ptr.load[width=simd_width](index), p
                ),
            )

        vectorize[array_scalar_vectorize, self.width](self.size)
        return new_vec

    fn __truediv__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array / scalar`.
        """
        return math.div[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __truediv__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: NDArray[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array / array`.
        """
        return math.div[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    fn __truediv__(self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `array / scalar`.
        """
        return math.div[dtype](self, other)

    fn __truediv__(self, other: Self) raises -> Self:
        """
        Enables `array / array`.
        """
        return math.div[dtype](self, other)

    fn __itruediv__(mut self, s: SIMD[dtype, 1]) raises:
        """
        Enables `array /= scalar`.
        """
        self = self.__truediv__(s)

    fn __itruediv__(mut self, other: Self) raises:
        """
        Enables `array /= array`.
        """
        self = self.__truediv__(other)

    fn __rtruediv__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, s: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `scalar / array`.
        """
        return math.div[ResultDType](
            s.cast[ResultDType](), self.astype[ResultDType]()
        )

    fn __rtruediv__(self, s: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar / array`.
        """
        return math.div[dtype](s, self)

    fn __floordiv__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array // scalar`.
        """
        return math.floor_div[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __floordiv__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: NDArray[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array // array`.
        """
        return math.floor_div[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    fn __floordiv__(self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `array // scalar`.
        """
        return math.floor_div[dtype](self, other)

    fn __floordiv__(self, other: Self) raises -> Self:
        """
        Enables `array // array`.
        """
        return math.floor_div[dtype](self, other)

    fn __ifloordiv__(mut self, s: SIMD[dtype, 1]) raises:
        """
        Enables `array //= scalar`.
        """
        self = self.__floordiv__(s)

    fn __ifloordiv__(mut self, other: Self) raises:
        """
        Enables `array //= array`.
        """
        self = self.__floordiv__(other)

    fn __rfloordiv__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `scalar // array`.
        """
        return math.floor_div[ResultDType](
            other.cast[ResultDType](), self.astype[ResultDType]()
        )

    fn __rfloordiv__(self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar // array`.
        """
        return math.floor_div[dtype](other, self)

    fn __mod__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array % scalar`.
        """
        return math.mod[ResultDType](
            self.astype[ResultDType](), other.cast[ResultDType]()
        )

    fn __mod__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: NDArray[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `array % array`.
        """
        return math.mod[ResultDType](
            self.astype[ResultDType](), other.astype[ResultDType]()
        )

    fn __mod__(mut self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `array % scalar`.
        """
        return math.mod[dtype](self, other)

    fn __mod__(mut self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `array % array`.
        """
        return math.mod[dtype](self, other)

    fn __imod__(mut self, other: SIMD[dtype, 1]) raises:
        """
        Enables `array %= scalar`.
        """
        self = math.mod[dtype](self, other)

    fn __imod__(mut self, other: NDArray[dtype]) raises:
        """
        Enables `array %= array`.
        """
        self = math.mod[dtype](self, other)

    fn __rmod__(mut self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar % array`.
        """
        return math.mod[dtype](other, self)

    fn __rmod__[
        OtherDType: DType,
        ResultDType: DType = TypeCoercion.result[dtype, OtherDType](),
    ](self, other: Scalar[OtherDType]) raises -> NDArray[ResultDType]:
        """
        Enables `scalar % array`.
        """
        return math.mod[ResultDType](
            other.cast[ResultDType](), self.astype[ResultDType]()
        )

    # ===-------------------------------------------------------------------===#
    # IO dunders and other methods
    # Trait implementations
    # ===-------------------------------------------------------------------===#
    fn __str__(self) -> String:
        """
        Enables str(array).
        """
        var res: String
        try:
            res = self._array_to_string(0, 0, GLOBAL_PRINT_OPTIONS)
        except e:
            res = String("Cannot convert array to string") + str(e)

        return res

    fn write_to[W: Writer](self, mut writer: W):
        try:
            writer.write(
                self._array_to_string(0, 0, GLOBAL_PRINT_OPTIONS)
                + "\n"
                + str(self.ndim)
                + "D-array  Shape"
                + str(self.shape)
                + "  Strides"
                + str(self.strides)
                + "  DType: "
                + _concise_dtype_str(self.dtype)
                + "  C-cont: "
                + str(self.flags["C_CONTIGUOUS"])
                + "  F-cont: "
                + str(self.flags["F_CONTIGUOUS"])
                + "  own data: "
                + str(self.flags["OWNDATA"])
            )
        except e:
            writer.write("Cannot convert array to string" + str(e))

    fn __repr__(self) -> String:
        """
        Compute the "official" string representation of NDArray.

        You can construct the array using this representation.

        An example is:
        ```console
        >>>import numojo as nm
        >>>var b = nm.arange[nm.f32](20).reshape(Shape(4, 5))
        >>>print(repr(b))
        numojo.array[f32](
        '''
        [[0.0, 1.0, 2.0, 3.0, 4.0]
         [5.0, 6.0, 7.0, 8.0, 9.0]
         [10.0, 11.0, 12.0, 13.0, 14.0]
         [15.0, 16.0, 17.0, 18.0, 19.0]]
        '''
        )
        ```
        """
        var result: String

        try:
            result = (
                str("numojo.array[")
                + _concise_dtype_str(self.dtype)
                + str('](\n"""\n')
                + self._array_to_string(0, 0, GLOBAL_PRINT_OPTIONS)
                + '\n"""\n)'
            )
        except e:
            result = "Cannot convert array to string" + str(e)

        return result

    fn __len__(self) -> Int:
        """
        Returns length of 0-th dimension.
        """
        return self.shape._buf[0]

    fn __iter__(self) raises -> _NDArrayIter[__origin_of(self), dtype]:
        """
        Iterate over elements of the NDArray and return sub-arrays as view.

        Returns:
            An iterator of NDArray elements.

        Example:
        ```
        >>> var a = nm.random.arange[nm.i8](2 * 3 * 4).reshape(nm.Shape(2, 3, 4))
        >>> for i in a:
        ...     print(i)
        [[      0       1       2       3       ]
         [      4       5       6       7       ]
         [      8       9       10      11      ]]
        2-D array  Shape: [3, 4]  DType: int8  C-cont: True  F-cont: False  own data: False
        [[      12      13      14      15      ]
         [      16      17      18      19      ]
         [      20      21      22      23      ]]
        2-D array  Shape: [3, 4]  DType: int8  C-cont: True  F-cont: False  own data: False
        ```.
        """

        return _NDArrayIter[__origin_of(self), dtype](
            ptr=self._buf.ptr,
            length=self.shape[0],
            stride_of_axis=self.strides[0],
            shape=self.shape._pop(axis=0),
            strides=self.strides._pop(axis=0),
        )

    fn __reversed__(
        self,
    ) raises -> _NDArrayIter[__origin_of(self), dtype, forward=False]:
        """Iterate backwards over elements of the NDArray, returning
        copied value.

        Returns:
            A reversed iterator of NDArray elements.
        """

        return _NDArrayIter[__origin_of(self), dtype, forward=False](
            ptr=self._buf.ptr,
            length=self.shape[0],
            stride_of_axis=self.strides[0],
            shape=self.shape._pop(axis=0),
            strides=self.strides._pop(axis=0),
        )

    fn _adjust_slice(self, slice_list: List[Slice]) raises -> List[Slice]:
        """
        Adjusts the slice values to lie within 0 and dim.
        """
        var n_slices: Int = slice_list.__len__()
        var slices = List[Slice]()
        for i in range(n_slices):
            if i >= self.ndim:
                raise Error("Error: Number of slices exceeds array dimensions")

            var start: Int = 0
            var end: Int = self.shape[i]
            var step: Int = 1
            if slice_list[i].start is not None:
                start = slice_list[i].start.value()
                if start < 0:
                    # start += self.shape[i]
                    raise Error(
                        "Error: Negative indexing in slices not supported"
                        " currently"
                    )

            if slice_list[i].end is not None:
                end = slice_list[i].end.value()
                if end < 0:
                    # end += self.shape[i] + 1
                    raise Error(
                        "Error: Negative indexing in slices not supported"
                        " currently"
                    )
            step = slice_list[i].step.or_else(1)
            if step == 0:
                raise Error("Error: Slice step cannot be zero")

            slices.append(
                Slice(
                    start=Optional(start),
                    end=Optional(end),
                    step=Optional(step),
                )
            )

        return slices^

    fn _array_to_string(
        self,
        dimension: Int,
        offset: Int,
        print_options: PrintOptions,
    ) raises -> String:
        """
        Convert the array to a string.

        Args:
            dimension: The current dimension.
            offset: The offset of the current dimension.
            print_options: The print options.
        """
        var seperator = print_options.separator
        var padding = print_options.padding
        var edge_items = print_options.edge_items

        if self.ndim == 0:
            return str(self.item(0))
        if dimension == self.ndim - 1:
            var result: String = String("[") + padding
            var number_of_items = self.shape[dimension]
            if number_of_items <= edge_items:  # Print all items
                for i in range(number_of_items):
                    var value = self.load[width=1](
                        offset + i * self.strides[dimension]
                    )
                    var formatted_value = self._format_value(
                        value, print_options
                    )
                    result = result + formatted_value
                    if i < (number_of_items - 1):
                        result = result + seperator
                result = result + padding
            else:  # Print first 3 and last 3 items
                for i in range(edge_items // 2):
                    var value = self.load[width=1](
                        offset + i * self.strides[dimension]
                    )
                    var formatted_value = self._format_value(
                        value, print_options
                    )
                    result = result + formatted_value
                    if i < (edge_items // 2 - 1):
                        result = result + seperator
                result = result + seperator + "..." + seperator
                for i in range(
                    number_of_items - edge_items // 2, number_of_items
                ):
                    var value = self.load[width=1](
                        offset + i * self.strides[dimension]
                    )
                    var formatted_value = self._format_value(
                        value, print_options
                    )
                    result = result + formatted_value
                    if i < (number_of_items - 1):
                        result = result + seperator
                result = result + padding
            result = result + "]"
            return result
        else:
            var result: String = str("[")
            var number_of_items = self.shape[dimension]
            if number_of_items <= edge_items:  # Print all items
                for i in range(number_of_items):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.strides[dimension].__int__(),
                            print_options,
                        )
                    if i > 0:
                        result = (
                            result
                            + str(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset + i * self.strides[dimension].__int__(),
                                print_options,
                            )
                        )
                    if i < (number_of_items - 1):
                        result = result + "\n"
            else:  # Print first 3 and last 3 items
                for i in range(edge_items // 2):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.strides[dimension].__int__(),
                            print_options,
                        )
                    if i > 0:
                        result = (
                            result
                            + str(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset + i * self.strides[dimension].__int__(),
                                print_options,
                            )
                        )
                    if i < (number_of_items - 1):
                        result += "\n"
                result = result + "...\n"
                for i in range(
                    number_of_items - edge_items // 2, number_of_items
                ):
                    result = (
                        result
                        + str(" ") * (dimension + 1)
                        + self._array_to_string(
                            dimension + 1,
                            offset + i * self.strides[dimension].__int__(),
                            print_options,
                        )
                    )
                    if i < (number_of_items - 1):
                        result = result + "\n"
            result = result + "]"
            return result

    fn _format_value[
        dtype: DType
    ](self, value: Scalar[dtype], print_options: PrintOptions) raises -> String:
        """
        Format a single value based on the print options.

        Args:
            value: The value to format.
            print_options: The print options.

        Returns:
            The formatted value as a string.
        """
        var sign = print_options.sign
        var float_format = print_options.float_format
        var nan_string = print_options.nan_string
        var inf_string = print_options.inf_string
        var formatted_width = print_options.formatted_width

        @parameter
        if is_floattype[dtype]():
            if isnan(value):
                return nan_string.rjust(formatted_width)
            if isinf(value):
                return inf_string.rjust(formatted_width)
            if float_format == "scientific":
                return format_floating_scientific(
                    value, print_options.precision, sign
                )
            else:
                return format_floating_precision(
                    value, print_options.precision, sign
                ).rjust(formatted_width)
        else:
            var formatted = str(value)
            if sign and value > 0:
                formatted = "+" + formatted
            return formatted.rjust(formatted_width)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn vdot(self, other: Self) raises -> SIMD[dtype, 1]:
        """
        Inner product of two vectors.
        """
        if self.size != other.size:
            raise Error("The lengths of two vectors do not match.")

        var sum = Scalar[dtype](0)
        for i in range(self.size):
            sum = sum + self.load(i) * other.load(i)
        return sum

    fn mdot(self, other: Self) raises -> Self:
        """
        Dot product of two matrix.
        Matrix A: M * N.
        Matrix B: N * L.
        """

        if (self.ndim != 2) or (other.ndim != 2):
            raise Error(
                String(
                    "The array should have only two dimensions (matrix).\n"
                    "The self array has {} dimensions.\n"
                    "The orther array has {} dimensions"
                ).format(self.ndim, other.ndim)
            )

        if self.shape[1] != other.shape[0]:
            raise Error(
                String(
                    "Second dimension of A does not match first dimension of"
                    " B.\nA is {}x{}. \nB is {}x{}."
                ).format(
                    self.shape[0], self.shape[1], other.shape[0], other.shape[1]
                )
            )

        var new_matrix = Self(Shape(self.shape[0], other.shape[1]))
        for row in range(self.shape[0]):
            for col in range(other.shape[1]):
                new_matrix.__setitem__(
                    Item(row, col),
                    self[row : row + 1, :].vdot(other[:, col : col + 1]),
                )
        return new_matrix

    fn row(self, id: Int) raises -> Self:
        """Get the ith row of the matrix."""

        if self.ndim > 2:
            raise Error(
                String(
                    "The number of dimension is {}.\nIt should be 2."
                ).format(self.ndim)
            )

        var width = self.shape[1]
        var buffer = Self(Shape(width))
        for i in range(width):
            buffer.store(i, self._buf.ptr.load[width=1](i + id * width))
        return buffer

    fn col(self, id: Int) raises -> Self:
        """Get the ith column of the matrix."""

        if self.ndim > 2:
            raise Error(
                String(
                    "The number of dimension is {}.\nIt should be 2."
                ).format(self.ndim)
            )

        var width = self.shape[1]
        var height = self.shape[0]
        var buffer = Self(Shape(height))
        for i in range(height):
            buffer.store(i, self._buf.ptr.load[width=1](id + i * width))
        return buffer

    # # * same as mdot
    fn rdot(self, other: Self) raises -> Self:
        """
        Dot product of two matrix.
        Matrix A: M * N.
        Matrix B: N * L.
        """

        if (self.ndim != 2) or (other.ndim != 2):
            raise Error(
                String(
                    "The array should have only two dimensions (matrix)."
                    "The self array is of {} dimensions.\n"
                    "The other array is of {} dimensions."
                ).format(self.ndim, other.ndim)
            )
        if self.shape[1] != other.shape[0]:
            raise Error(
                String(
                    "Second dimension of A ({}) \n"
                    "does not match first dimension of B ({})."
                ).format(self.shape[1], other.shape[0])
            )

        var new_matrix = Self(Shape(self.shape[0], other.shape[1]))
        for row in range(self.shape[0]):
            for col in range(other.shape[1]):
                new_matrix.store(
                    col + row * other.shape[1],
                    self.row(row).vdot(other.col(col)),
                )
        return new_matrix

    fn num_elements(self) -> Int:
        """
        Function to retreive size (compatability).
        """
        return self.size

    fn load(self, owned index: Int) raises -> Scalar[dtype]:
        """
        Safely retrieve i-th item from the underlying buffer.

        `A.load(i)` differs from `A._buf.ptr[i]` due to boundary check.

        Example:
        ```console
        > array.load(15)
        ```
        returns the item of index 15 from the array's data buffer.

        Note that it does not checked against C-order or F-order.
        ```console
        > # A is a 3x3 matrix, F-order (column-major)
        > A.load(3)  # Row 0, Col 1
        > A.item(3)  # Row 1, Col 0
        ```
        """

        if index < 0:
            index += self.size

        if (index >= self.size) or (index < 0):
            raise Error(
                String("Invalid index: index out of bound [0, {}).").format(
                    self.size
                )
            )

        return self._buf.ptr[index]

    fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
        """
        Safely loads a SIMD element of size `width` at `index`
        from the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.load` directly.

        Raises:
            Index out of boundary.
        """

        if (index < 0) or (index >= self.size):
            raise Error(
                String("Invalid index: index out of bound [0, {}).").format(
                    self.size
                )
            )

        return self._buf.ptr.load[width=width](index)

    fn load[width: Int = 1](self, *indices: Int) raises -> SIMD[dtype, width]:
        """
        Safely loads SIMD element of size `width` at given variadic indices
        from the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.load` directly.

        Raises:
            Index out of boundary.
        """

        if len(indices) != self.ndim:
            raise (
                String(
                    "Length of indices {} does not match ndim {}".format(
                        len(indices), self.ndim
                    )
                )
            )

        for i in range(self.ndim):
            if (indices[i] < 0) or (indices[i] >= self.shape[i]):
                raise Error(
                    String(
                        "Invalid index at {}-th dim: "
                        "index out of bound [0, {})."
                    ).format(i, self.shape[i])
                )

        var idx: Int = _get_offset(indices, self.strides)
        return self._buf.ptr.load[width=width](idx)

    fn store(self, owned index: Int, val: Scalar[dtype]) raises:
        """
        Safely store a scalar to i-th item of the underlying buffer.

        `A.store(i, a)` differs from `A._buf.ptr[i] = a` due to boundary check.

        Raises:
            Index out of boundary.

        Example:
        ```console
        > array.store(15, val = 100)
        ```
        sets the item of index 15 of the array's data buffer to 100.

        Note that it does not checked against C-order or F-order.
        """

        if index < 0:
            index += self.size

        if (index >= self.size) or (index < 0):
            raise Error(
                String("Invalid index: index out of bound [0, {}).").format(
                    self.size
                )
            )

        self._buf.ptr[index] = val

    fn store[width: Int](mut self, index: Int, val: SIMD[dtype, width]) raises:
        """
        Safely stores SIMD element of size `width` at `index`
        of the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.store` directly.

        Raises:
            Index out of boundary.
        """

        if (index < 0) or (index >= self.size):
            raise Error(
                String("Invalid index: index out of bound [0, {}).").format(
                    self.size
                )
            )

        self._buf.ptr.store(index, val)

    fn store[
        width: Int = 1
    ](mut self, *indices: Int, val: SIMD[dtype, width]) raises:
        """
        Safely stores SIMD element of size `width` at given variadic indices
        of the underlying buffer.

        To bypass boundary checks, use `self._buf.ptr.store` directly.

        Raises:
            Index out of boundary.
        """

        if len(indices) != self.ndim:
            raise (
                String(
                    "Length of indices {} does not match ndim {}".format(
                        len(indices), self.ndim
                    )
                )
            )

        for i in range(self.ndim):
            if (indices[i] < 0) or (indices[i] >= self.shape[i]):
                raise Error(
                    String(
                        "Invalid index at {}-th dim: "
                        "index out of bound [0, {})."
                    ).format(i, self.shape[i])
                )

        var idx: Int = _get_offset(indices, self.strides)
        self._buf.ptr.store(idx, val)

    # ===-------------------------------------------------------------------===#
    # OTHER METHODS
    # (Sorted alphabetically)
    #
    # TODO: Implement axis parameter for all operations that are along an axis
    #
    # # not urgent: argpartition, byteswap, choose, conj, dump, getfield
    # # partition, put, repeat, searchsorted, setfield, squeeze, swapaxes, take,
    # # tobyets, tofile, view
    # ===-------------------------------------------------------------------===#
    fn T(self, axes: List[Int]) raises -> Self:
        """
        Transpose array of any number of dimensions according to
        arbitrary permutation of the axes.

        If `axes` is not given, it is equal to flipping the axes.

        Defined in `numojo.routines.manipulation.transpose`.
        """
        return numojo.routines.manipulation.transpose(self, axes)

    fn T(self) raises -> Self:
        """
        (overload) Transpose the array when `axes` is not given.
        If `axes` is not given, it is equal to flipping the axes.
        See docstring of `transpose`.

        Defined in `numojo.routines.manipulation.transpose`.
        """
        return numojo.routines.manipulation.transpose(self)

    fn all(self) raises -> Bool:
        """
        If all true return true.
        """
        # make this a compile time check when they become more readable
        if not (self.dtype is DType.bool or self.dtype.is_integral()):
            raise Error("Array elements must be Boolean or Integer.")
        # We might need to figure out how we want to handle truthyness before can do this
        var result: Bool = True

        @parameter
        fn vectorized_all[simd_width: Int](idx: Int) -> None:
            result = result and builtin_bool.all(
                (self._buf.ptr + idx).strided_load[width=simd_width](1)
            )

        vectorize[vectorized_all, self.width](self.size)
        return result

    fn any(self) raises -> Bool:
        """
        True if any true.
        """
        # make this a compile time check
        if not (self.dtype is DType.bool or self.dtype.is_integral()):
            raise Error("Array elements must be Boolean or Integer.")
        var result: Bool = False

        @parameter
        fn vectorized_any[simd_width: Int](idx: Int) -> None:
            result = result or builtin_bool.any(
                (self._buf.ptr + idx).strided_load[width=simd_width](1)
            )

        vectorize[vectorized_any, self.width](self.size)
        return result

    fn argmax(self) raises -> Int:
        """
        Get location in pointer of max value.
        """
        var result: Int = 0
        var max_val: SIMD[dtype, 1] = self.load[width=1](0)
        for i in range(1, self.size):
            var temp: SIMD[dtype, 1] = self.load[width=1](i)
            if temp > max_val:
                max_val = temp
                result = i
        return result

    fn argmin(self) raises -> Int:
        """
        Get location in pointer of min value.
        """
        var result: Int = 0
        var min_val: SIMD[dtype, 1] = self.load[width=1](0)
        for i in range(1, self.size):
            var temp: SIMD[dtype, 1] = self.load[width=1](i)
            if temp < min_val:
                min_val = temp
                result = i
        return result

    fn argsort(self) raises -> NDArray[DType.index]:
        """
        Sort the NDArray and return the sorted indices.

        See `numojo.routines.sorting.argsort()`.

        Returns:
            The indices of the sorted NDArray.
        """

        return sorting.argsort(self)

    fn astype[target: DType](self) raises -> NDArray[target]:
        """
        Convert type of array.
        """
        return creation.astype[target](self)

    # fn clip(self):
    #     pass

    # fn compress(self):
    #     pass

    fn copy(self) raises -> Self:
        # TODO: Add logics for non-contiguous arrays when views are implemented.
        """
        Returns a copy of the array that owns the data.
        The returned array will be continuous in memory.
        """

        if (self.strides == NDArrayStrides(shape=self.shape)) or (
            self.strides == NDArrayStrides(shape=self.shape, order="F")
        ):
            # The strides and shape are matched.
            # It either owns the data or it is a continuous view of another array.
            # The array is continuous in memory. Nothing needs to be changed.
            var result = self
            return result
        else:
            # The strides and shape are not matched.
            # It is a view of another array with different shape and strides.
            if self.flags["C_CONTIGUOUS"]:
                # The array is C-continuous in memory.
                # Can be copied by the last dimension.
                var result = self
                return result

            elif self.flags["F_CONTIGUOUS"]:
                # The array is F-continuous in memory.
                # Can be copied by the first dimension.
                var result = self
                return result
            else:
                # The array is not continuous in memory.
                # Can be copied by item.
                var result = self
                return result

    fn cumprod(self) raises -> NDArray[dtype]:
        """
        Returns cumprod of all items of an array.
        The array is flattened before cumprod.

        Returns:
            Cumprod of all items of an array.
        """
        return cumprod[dtype](self)

    fn cumprod(self, axis: Int) raises -> NDArray[dtype]:
        """
        Returns cumprod of array by axis.

        Args:
            axis: Axis.

        Returns:
            Cumprod of array by axis.
        """
        return cumprod[dtype](self, axis=axis)

    fn cumsum(self) raises -> NDArray[dtype]:
        """
        Returns cumsum of all items of an array.
        The array is flattened before cumsum.

        Returns:
            Cumsum of all items of an array.
        """
        return cumsum[dtype](self)

    fn cumsum(self, axis: Int) raises -> NDArray[dtype]:
        """
        Returns cumsum of array by axis.

        Args:
            axis: Axis.

        Returns:
            Cumsum of array by axis.
        """
        return cumsum[dtype](self, axis=axis)

    fn diagonal(self):
        pass

    fn fill(mut self, val: Scalar[dtype]):
        """
        Fill all items of array with value.
        """

        for i in range(self.size):
            self._buf.ptr[i] = val

    fn flatten(self, order: String = "C") raises -> Self:
        """
        Return a copy of the array collapsed into one dimension.

        Args:
            order: A NDArray.

        Returns:
            The 1 dimensional flattened NDArray.
        """
        return ravel(self, order=order)

    fn item(
        self, owned index: Int
    ) raises -> ref [self._buf.ptr.origin, self._buf.ptr.address_space] Scalar[
        dtype
    ]:
        """
        Return the scalar at the coordinates.

        If one index is given, get the i-th item of the array (not buffer).
        It first scans over the first row, even it is a colume-major array.

        If more than one index is given, the length of the indices must match
        the number of dimensions of the array.

        Args:
            index: Index of item, counted in row-major way.

        Returns:
            A scalar matching the dtype of the array.

        Raises:
            Index is equal or larger than array size.

        Example:
        ```console
        >>> var A = nm.random.randn[nm.f16](2, 2, 2)
        >>> A = A.reshape(A.shape, order="F")
        >>> print(A)
        [[[     0.2446289       0.5419922       ]
          [     0.09643555      -0.90722656     ]]
         [[     1.1806641       0.24389648      ]
          [     0.5234375       1.0390625       ]]]
        3-D array  Shape: [2, 2, 2]  DType: float16  order: F
        >>> for i in range(A.size):
        ...     print(A.item(i))
        0.2446289
        0.5419922
        0.09643555
        -0.90722656
        1.1806641
        0.24389648
        0.5234375
        1.0390625
        >>> print(A.item(0, 1, 1))
        -0.90722656
        ```.
        """

        if index < 0:
            index += self.size

        if (index < 0) or (index >= self.size):
            raise Error(
                String("`index` exceeds array size ({})").format(self.size)
            )

        if self.flags["F_CONTIGUOUS"]:
            # column-major should be converted to row-major
            # The following code can be taken out as a function that
            # convert any index to coordinates according to the order
            var c_stride = NDArrayStrides(shape=self.shape)
            var c_coordinates = List[Int]()
            var idx: Int = index
            for i in range(c_stride.ndim):
                var coordinate = idx // c_stride[i]
                idx = idx - c_stride[i] * coordinate
                c_coordinates.append(coordinate)

            # Get the value by coordinates and the strides
            return (self._buf.ptr + _get_offset(c_coordinates, self.strides))[]

        else:
            return (self._buf.ptr + index)[]

    fn item(
        self, *index: Int
    ) raises -> ref [self._buf.ptr.origin, self._buf.ptr.address_space] Scalar[
        dtype
    ]:
        """
        Return the scalar at the coordinates.

        If one index is given, get the i-th item of the array (not buffer).
        It first scans over the first row, even it is a colume-major array.

        If more than one index is given, the length of the indices must match
        the number of dimensions of the array.

        Args:
            index: The coordinates of the item.

        Returns:
            A scalar matching the dtype of the array.

        Raises:
            Index is equal or larger than size of dimension.

        Example:
        ```
        >>> var A = nm.random.randn[nm.f16](2, 2, 2)
        >>> A = A.reshape(A.shape, order="F")
        >>> print(A)
        [[[     0.2446289       0.5419922       ]
          [     0.09643555      -0.90722656     ]]
         [[     1.1806641       0.24389648      ]
          [     0.5234375       1.0390625       ]]]
        3-D array  Shape: [2, 2, 2]  DType: float16  order: F
        >>> print(A.item(0, 1, 1))
        -0.90722656
        ```.
        """

        if len(index) != self.ndim:
            raise Error(
                String("Number of indices ({}) do not match ndim ({})").format(
                    len(index), self.ndim
                )
            )
        var list_index = List[Int]()
        for i in range(len(index)):
            if index[i] < 0:
                list_index.append(index[i] + self.shape[i])
            else:
                list_index.append(index[i])
            if (list_index[i] < 0) or (list_index[i] >= self.shape[i]):
                raise Error(
                    String("{}-th index exceeds shape size {}").format(
                        i, self.shape[i]
                    )
                )
        return (self._buf.ptr + _get_offset(index, self.strides))[]

    fn itemset(
        mut self, index: Variant[Int, List[Int]], item: Scalar[dtype]
    ) raises:
        """Set the scalar at the coordinates.

        Args:
            index: The coordinates of the item.
                Can either be `Int` or `List[Int]`.
                If `Int` is passed, it is the index of i-th item of the whole array.
                If `List[Int]` is passed, it is the coordinate of the item.
            item: The scalar to be set.

        Note:
            This is similar to `numpy.ndarray.itemset`.
            The difference is that we takes in `List[Int]`, but numpy takes in a tuple.

        An example goes as follows.

        ```
        import numojo as nm

        fn main() raises:
            var A = nm.zeros[nm.i16](3, 3)
            print(A)
            A.itemset(5, 256)
            print(A)
            A.itemset(List(1,1), 1024)
            print(A)
        ```
        ```console
        [[      0       0       0       ]
         [      0       0       0       ]
         [      0       0       0       ]]
        2-D array  Shape: [3, 3]  DType: int16
        [[      0       0       0       ]
         [      0       0       256     ]
         [      0       0       0       ]]
        2-D array  Shape: [3, 3]  DType: int16
        [[      0       0       0       ]
         [      0       1024    256     ]
         [      0       0       0       ]]
        2-D array  Shape: [3, 3]  DType: int16
        ```
        """

        # If one index is given
        if index.isa[Int]():
            var idx = index._get_ptr[Int]()[]
            if idx < self.size:
                if self.flags[
                    "F_CONTIGUOUS"
                ]:  # column-major should be converted to row-major
                    # The following code can be taken out as a function that
                    # convert any index to coordinates according to the order
                    var c_stride = NDArrayStrides(shape=self.shape)
                    var c_coordinates = List[Int]()
                    for i in range(c_stride.ndim):
                        var coordinate = idx // c_stride[i]
                        idx = idx - c_stride[i] * coordinate
                        c_coordinates.append(coordinate)
                    self._buf.ptr.store(
                        _get_offset(c_coordinates, self.strides), item
                    )

                self._buf.ptr.store(idx, item)
            else:
                raise Error(
                    String(
                        "Error: Elements of `index` ({}) \n"
                        "exceed the array size ({})."
                    ).format(idx, self.size)
                )

        else:
            var indices = index._get_ptr[List[Int]]()[]
            # If more than one index is given
            if indices.__len__() != self.ndim:
                raise Error("Error: Length of Indices do not match the shape")
            for i in range(indices.__len__()):
                if indices[i] >= self.shape[i]:
                    raise Error(
                        "Error: Elements of `index` exceed the array shape"
                    )
            self._buf.ptr.store(_get_offset(indices, self.strides), item)

    fn max(self, axis: Int = 0) raises -> Self:
        """
        Max on axis.
        """
        var ndim: Int = self.ndim
        var shape: List[Int] = List[Int]()
        for i in range(ndim):
            shape.append(self.shape[i])
        if axis > ndim - 1:
            raise Error(
                String(
                    "Axis index ({}) must be smaller than "
                    "the rank of the array ({})."
                ).format(axis, ndim)
            )
        var result_shape: List[Int] = List[Int]()
        var axis_size: Int = shape[axis]
        var slices: List[Slice] = List[Slice]()
        for i in range(ndim):
            if i != axis:
                result_shape.append(shape[i])
                slices.append(Slice(0, shape[i], 1))
            else:
                slices.append(Slice(0, 0, 1))

        slices[axis] = Slice(0, 1, 1)
        var result: NDArray[dtype] = self[slices]
        for i in range(1, axis_size):
            slices[axis] = Slice(i, i + 1, 1)
            var arr_slice = self[slices]
            var mask1 = comparison.greater(arr_slice, result)
            var mask2 = comparison.less(arr_slice, result)
            # Wherever result is less than the new slice it is set to zero
            # Wherever arr_slice is greater than the old result it is added to fill those zeros
            result = arithmetic.add(
                result * bool_to_numeric[dtype](mask2),
                arr_slice * bool_to_numeric[dtype](mask1),
            )

        return result

    fn min(self, axis: Int = 0) raises -> Self:
        """
        Min on axis.
        """
        var ndim: Int = self.ndim
        var shape: List[Int] = List[Int]()
        for i in range(ndim):
            shape.append(self.shape[i])
        if axis > ndim - 1:
            raise Error(
                String(
                    "Axis index ({}) must be smaller than "
                    "the rank of the array ({})."
                ).format(axis, ndim)
            )
        var result_shape: List[Int] = List[Int]()
        var axis_size: Int = shape[axis]
        var slices: List[Slice] = List[Slice]()
        for i in range(ndim):
            if i != axis:
                result_shape.append(shape[i])
                slices.append(Slice(0, shape[i], 1))
            else:
                slices.append(Slice(0, 0, 1))

        slices[axis] = Slice(0, 1, 1)
        var result: NDArray[dtype] = self[slices]
        for i in range(1, axis_size):
            slices[axis] = Slice(i, i + 1, 1)
            var arr_slice = self[slices]
            var mask1 = comparison.less(arr_slice, result)
            var mask2 = comparison.greater(arr_slice, result)
            # Wherever result is greater than the new slice it is set to zero
            # Wherever arr_slice is less than the old result it is added to fill those zeros
            result = arithmetic.add(
                result * bool_to_numeric[dtype](mask2),
                arr_slice * bool_to_numeric[dtype](mask1),
            )

        return result

    fn mean(self: Self, axis: Int) raises -> Self:
        """
        Mean of array elements over a given axis.
        Args:
            array: NDArray.
            axis: The axis along which the mean is performed.
        Returns:
            An NDArray.

        """
        return mean(self, axis)

    fn mean(self) raises -> Scalar[dtype]:
        """
        Cumulative mean of a array.

        Returns:
            The cumulative mean of the array as a SIMD Value of `dtype`.
        """
        return cummean[dtype](self)

    # fn nonzero(self):
    #     pass

    fn nditer(self) raises -> _NDIter[__origin_of(self), dtype]:
        """
        (Overload) Return an iterator yielding the array elements according
        to the memory layout of the array.

        ```console
        >>>var a = nm.random.rand[i8](2, 3, min=0, max=100)
        >>>print(a)
        [[      37      8       25      ]
         [      25      2       57      ]]
        2-D array  (2,3)  DType: int8  C-cont: True  F-cont: False  own data: True
        >>>for i in a.nditer():
        ...    print(i, end=" ")
        37 8 25 25 2 57
        ```
        """

        var order: String

        if self.flags["F_CONTIGUOUS"]:
            order = "F"
        else:
            order = "C"

        return self.nditer(order=order)

    fn nditer(self, order: String) raises -> _NDIter[__origin_of(self), dtype]:
        """
        Return an iterator yielding the array elements according to the order.

        ```console
        >>>var a = nm.random.rand[i8](2, 3, min=0, max=100)
        >>>print(a)
        [[      37      8       25      ]
         [      25      2       57      ]]
        2-D array  (2,3)  DType: int8  C-cont: True  F-cont: False  own data: True
        >>>for i in a.nditer():
        ...    print(i, end=" ")
        37 8 25 25 2 57
        ```
        """

        return _NDIter[__origin_of(self), dtype](
            ptr=self._buf.ptr,
            length=self.size,
            ndim=self.ndim,
            strides=self.strides,
            shape=self.shape,
            order=order,
        )

    fn prod(self: Self) raises -> Scalar[dtype]:
        """
        Product of all array elements.
        Returns:
            Scalar.
        """
        return sum(self)

    fn prod(self: Self, axis: Int) raises -> Self:
        """
        Product of array elements over a given axis.
        Args:
            axis: The axis along which the product is performed.
        Returns:
            An NDArray.
        """

        return prod(self, axis=axis)

    fn reshape(self, shape: NDArrayShape, order: String = "C") raises -> Self:
        """
        Returns an array of the same data with a new shape.

        Args:
            shape: Shape of returned array.
            order: Order of the array - Row major `C` or Column major `F`.

        Returns:
            Array of the same data with a new shape.
        """
        return reshape[dtype](self, shape=shape, order=order)

    fn resize(mut self, shape: NDArrayShape) raises:
        """
        In-place change shape and size of array.

        Notes:
        To returns a new array, use `reshape`.

        Args:
            shape: Shape after resize.
        """

        var order = "C" if self.flags["C_CONTIGUOUS"] else "F"

        if shape.size_of_array() > self.size:
            var other = Self(shape=shape, order=order)
            memcpy(other._buf.ptr, self._buf.ptr, self.size)
            for i in range(self.size, other.size):
                (other._buf.ptr + i).init_pointee_copy(0)
            self = other^
        else:
            self.shape = shape
            self.ndim = shape.ndim
            self.size = shape.size_of_array()
            self.strides = NDArrayStrides(shape, order=order)

    fn round(self) raises -> Self:
        """
        Rounds the elements of the array to a whole number.

        Returns:
            An NDArray.
        """
        return rounding.tround[dtype](self)

    fn sort(mut self) raises:
        """
        Sort NDArray using quick sort method.
        It is not guaranteed to be unstable.

        When no axis is given, the array is flattened before sorting.

        See `numojo.sorting.sort` for more information.
        """
        var I = NDArray[DType.index](self.shape)
        self = ravel(self)
        sorting._sort_inplace(self, I, axis=0)

    fn sort(mut self, owned axis: Int) raises:
        """
        Sort NDArray along the given axis using quick sort method.
        It is not guaranteed to be unstable.

        When no axis is given, the array is flattened before sorting.

        See `numojo.sorting.sort` for more information.
        """
        var I = NDArray[DType.index](self.shape)
        sorting._sort_inplace(self, I, axis=axis)

    fn sum(self: Self) raises -> Scalar[dtype]:
        """
        Sum of all array elements.
        Returns:
            Scalar.
        """
        return sum(self)

    fn sum(self: Self, axis: Int) raises -> Self:
        """
        Sum of array elements over a given axis.
        Args:
            axis: The axis along which the sum is performed.
        Returns:
            An NDArray.
        """
        return sum(self, axis=axis)

    fn tolist(self) -> List[Scalar[dtype]]:
        """
        Convert NDArray to a 1-D List.

        Returns:
            A 1-D List.
        """
        var result: List[Scalar[dtype]] = List[Scalar[dtype]]()
        for i in range(self.size):
            result.append(self._buf.ptr[i])
        return result

    fn to_numpy(self) raises -> PythonObject:
        """
        Convert to a numpy array.
        """
        return to_numpy(self)

    fn to_tensor(self) raises -> Tensor[dtype]:
        """
        Convert array to tensor of the same dtype.

        ```mojo
        import numojo as nm
        from numojo.prelude import *

        fn main() raises:
            var a = nm.random.randn[f16](2, 3, 4)
            print(a)
            print(a.to_tensor())

            var b = nm.array[i8]("[[1, 2, 3], [4, 5, 6]]")
            print(b)
            print(b.to_tensor())

            var c = nm.array[boolean]("[[1,0], [0,1]]")
            print(c)
            print(c.to_tensor())
        ```
        """

        return to_tensor(self)

    # TODO: add axis parameter
    fn trace(
        self, offset: Int = 0, axis1: Int = 0, axis2: Int = 1
    ) raises -> NDArray[dtype]:
        """
        Computes the trace of a ndarray.

        Args:
            offset: Offset of the diagonal from the main diagonal.
            axis1: First axis.
            axis2: Second axis.

        Returns:
            The trace of the ndarray.
        """
        return linalg.norms.trace[dtype](self, offset, axis1, axis2)

    fn _transpose(self) raises -> Self:
        """
        Returns a view of transposed array.

        It is unsafe!
        """
        return Self(
            shape=self.shape._flip(),
            buffer=self._buf.ptr,
            offset=0,
            strides=self.strides._flip(),
        )

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        """
        Retreive pointer without taking ownership.
        """
        return self._buf.ptr


# ===----------------------------------------------------------------------===#
# NDArrayIterator
# ===----------------------------------------------------------------------===#


@value
struct _NDArrayIter[
    is_mutable: Bool, //,
    origin: Origin[is_mutable],
    dtype: DType,
    forward: Bool = True,
]:
    """Iterator for NDArray.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        origin: The lifetime of the underlying NDArray data.
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var ptr: UnsafePointer[Scalar[dtype]]
    var length: Int
    var stride_of_axis: Int
    var shape: NDArrayShape
    var strides: NDArrayStrides

    fn __init__(
        mut self,
        ptr: UnsafePointer[Scalar[dtype]],
        length: Int,
        stride_of_axis: Int,
        shape: NDArrayShape,
        strides: NDArrayStrides,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.ptr = ptr
        self.stride_of_axis = stride_of_axis
        self.shape = shape
        self.strides = strides

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) raises -> NDArray[dtype]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return NDArray(
                shape=self.shape,
                buffer=self.ptr,
                offset=current_index * self.stride_of_axis,
                strides=self.strides,
            )
        else:
            var current_index = self.index
            self.index -= 1
            return NDArray(
                shape=self.shape,
                buffer=self.ptr,
                offset=current_index * self.stride_of_axis,
                strides=self.strides,
            )

    @always_inline
    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index > 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index


@value
struct _NDIter[
    is_mutable: Bool, //, origin: Origin[is_mutable], dtype: DType
]():
    var ptr: UnsafePointer[Scalar[dtype]]
    var length: Int
    var ndim: Int
    var shape: NDArrayShape
    var strides: NDArrayStrides
    var index: Int
    var order: String

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[dtype]],
        length: Int,
        ndim: Int,
        shape: NDArrayShape,
        strides: NDArrayStrides,
        order: String,
    ):
        self.length = length
        self.ptr = ptr
        self.ndim = ndim
        self.shape = shape
        self.strides = strides
        self.order = order
        self.index = 0

    fn __iter__(self) -> Self:
        return self

    fn __has_next__(self) -> Bool:
        if self.index < self.length:
            return True
        else:
            return False

    fn __next__(mut self) raises -> Scalar[dtype]:
        var current_index = self.index
        self.index += 1

        var remainder = current_index
        var indices = Item(ndim=self.ndim, initialized=False)

        if self.order == "C":
            for i in range(self.ndim):
                indices[i], remainder = divmod(
                    remainder, NDArrayStrides(self.shape, order="C")[i]
                )
        else:
            for i in range(self.ndim - 1, -1, -1):
                indices[i], remainder = divmod(
                    remainder, NDArrayStrides(self.shape, order="F")[i]
                )

        return self.ptr[_get_offset(indices, self.strides)]
