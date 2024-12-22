"""
Implements N-Dimensional Array
"""
# ===----------------------------------------------------------------------=== #
# Implements ROW MAJOR N-DIMENSIONAL ARRAYS
# Last updated: 2024-10-14
# ===----------------------------------------------------------------------=== #


"""
# TODO
1) Generalize mdot, rdot to take any IxJx...xKxL and LxMx...xNxP matrix and matmul it into IxJx..xKxMx...xNxP array.
2) Add vectorization for _get_index
3) Create NDArrayView and remove coefficients.
4) Rename some variables or methods that should not be exposed to users.
"""

from builtin.type_aliases import Origin
from random import rand, random_si64, random_float64
from builtin.math import pow
from builtin.bool import all as allb
from builtin.bool import any as anyb
from algorithm import parallelize, vectorize
from python import Python, PythonObject
from sys import simdwidthof
from collections.optional import Optional
from utils import Variant
from memory import UnsafePointer
from memory import memset_zero, memcpy


import numojo.core._array_funcs as _af
import numojo.routines.sorting as sorting
import numojo.routines.math.arithmetic as arithmetic
import numojo.routines.logic.comparison as comparison
import numojo.routines.math.rounding as rounding
import numojo.routines.bitwise as bitwise
import numojo.routines.linalg as linalg

from numojo.routines.statistics.averages import mean, cummean
from numojo.routines.math.products import prod, cumprod
from numojo.routines.math.sums import sum, cumsum
from numojo.routines.math.extrema import maxT, minT
from ..traits import Backend
from numojo.routines.logic.truth import any
from .utility import (
    _get_index,
    _traverse_iterative,
    _traverse_iterative_setter,
    to_numpy,
    bool_to_numeric,
    is_inttype,
    is_booltype,
)
from numojo.core._math_funcs import Vectorized
from numojo.routines.linalg.products import matmul_parallelized
from numojo.routines.manipulation import reshape
from numojo.core.ndshape import NDArrayShape
from numojo.core.ndstrides import NDArrayStrides

# ===----------------------------------------------------------------------===#
# NDArray
# ===----------------------------------------------------------------------===#


struct NDArray[dtype: DType = DType.float64](
    Stringable, Representable, CollectionElement, Sized, Writable
):
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

    var _buf: UnsafePointer[Scalar[dtype]]
    """Data buffer of the items in the NDArray."""
    var ndim: Int
    """Number of Dimensions."""
    var shape: NDArrayShape
    """Size and shape of NDArray."""
    var size: Int
    """Size of NDArray."""
    var strides: NDArrayStrides
    """Contains offset, strides."""
    var coefficient: NDArrayStrides
    """Contains offset, coefficients for slicing."""
    var datatype: DType
    """The datatype of memory."""
    var order: String
    "Memory layout of array C (C order row major) or F (Fortran order col major)."

    alias width: Int = simdwidthof[dtype]()
    """Vector size of the data type."""

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
        self.size = self.shape.size
        self.strides = NDArrayStrides(shape, order=order)
        self.coefficient = NDArrayStrides(shape, order=order)
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(self.size)
        self.datatype = dtype
        self.order = order

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

    # Why do  these last two constructors exist?
    # constructor when rank, ndim, weights, first_index(offset) are known
    fn __init__(
        mut self,
        ndim: Int,
        offset: Int,
        size: Int,
        shape: List[Int],
        strides: List[Int],
        coefficient: List[Int],
        order: String = "C",
    ) raises:
        """
        Extremely specific NDArray initializer.
        """
        self.ndim = ndim
        self.shape = NDArrayShape(shape)
        self.size = size
        self.strides = NDArrayStrides(strides=strides, offset=0)
        self.coefficient = NDArrayStrides(strides=coefficient, offset=offset)
        self.datatype = dtype
        self.order = order
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(size)
        memset_zero(self._buf, size)

    # for creating views
    fn __init__(
        mut self,
        data: UnsafePointer[Scalar[dtype]],
        ndim: Int,
        offset: Int,
        shape: List[Int],
        strides: List[Int],
        coefficient: List[Int],
        order: String = "C",
    ) raises:
        """
        Extremely specific NDArray initializer.
        """
        self.ndim = ndim
        self.shape = NDArrayShape(shape)
        self.size = self.shape.size
        self.strides = NDArrayStrides(strides, offset=0, order=order)
        self.coefficient = NDArrayStrides(
            coefficient, offset=offset, order=order
        )
        self.datatype = dtype
        self.order = order
        self._buf = data + self.strides.offset

    @always_inline("nodebug")
    fn __copyinit__(mut self, other: Self):
        """
        Copy other into self.
        """
        self.ndim = other.ndim
        self.shape = other.shape
        self.size = other.size
        self.strides = other.strides
        self.coefficient = other.coefficient
        self.datatype = other.datatype
        self.order = other.order
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(other.size)
        memcpy(self._buf, other._buf, other.size)

    @always_inline("nodebug")
    fn __moveinit__(mut self, owned existing: Self):
        """
        Move other into self.
        """
        self.ndim = existing.ndim
        self.shape = existing.shape
        self.size = existing.size
        self.strides = existing.strides
        self.coefficient = existing.coefficient
        self.datatype = existing.datatype
        self.order = existing.order^
        self._buf = existing._buf

    @always_inline("nodebug")
    fn __del__(owned self):
        self._buf.free()

    # ===-------------------------------------------------------------------===#
    # Setter dunders
    # ===-------------------------------------------------------------------===#

    fn set(self, index: Int, val: SIMD[dtype, 1]) raises:
        """
        Linearly retreive a value from the underlying Pointer.

        Example:
        ```console
        > Array.get(15)
        ```
        returns the item of index 15 from the array's data buffer.

        Not that it is different from `item()` as `get` does not checked
        against C-order or F-order.
        ```console
        > # A is a 3x3 matrix, F-order (column-major)
        > A.get(3)  # Row 0, Col 1
        > A.item(3)  # Row 1, Col 0
        ```
        """
        if index >= self.size:
            var message = String(
                "Invalid index: index out of bound. \n"
                "The index is {}. \n"
                "The size is {}"
            ).format(index, self.size)
            raise Error(message)
        if index >= 0:
            return self._buf.store(index, val)
        else:
            return self._buf.store(index + self.size, val)

    # TODO: add support for different dtypes
    fn __setitem__(mut self, idx: Int, val: NDArray[dtype]) raises:
        """
        Set a slice of array with given array.

        Example:
            `arr[1]` returns the second row of the array.
        """
        if self.ndim == 0 and val.ndim == 0:
            self._buf.store(0, val._buf.load(0))

        var slice_list = List[Slice]()
        slice_list.append(Slice(idx, idx + 1))
        if self.ndim > 1:
            for i in range(1, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim))

        var n_slices: Int = len(slice_list)
        var ndims: Int = 0
        var count: Int = 0
        var spec: List[Int] = List[Int]()
        for i in range(n_slices):
            # self._adjust_slice_(slice_list[i], self.shape[i])
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
            # if slice_list[j].step is None:
            #     raise Error(String("Step of slice is None."))
            var slice_len: Int = (
                (slice_list[j].end.value() - slice_list[j].start.value())
                / slice_list[j].step.or_else(1)
            ).__int__()
            nshape.append(slice_len)
            nnum_elements *= slice_len
            # if slice_list[j].step is None:
            #     raise Error(String("Step of slice is None."))
            ncoefficients.append(
                self.strides[j] * slice_list[j].step.or_else(1)
            )
            j += 1

        # We can remove this check after we have support for broadcasting
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
        if self.order == "C":
            noffset = 0
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):  # temp
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]
        elif self.order == "F":
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

    fn __setitem__(mut self, index: Idx, val: SIMD[dtype, 1]) raises:
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
        var idx: Int = _get_index(index, self.coefficient)
        self._buf.store(idx, val)

    # compiler doesn't accept this
    # fn __setitem__(
    #     mut self, mask: NDArray[DType.bool], value: Scalar[dtype]
    # ) raises:
    #     """
    #     Set the value of the array at the indices where the mask is true.
    #     """
    #     if (
    #         mask.shape != self.shape
    #     ):  # this behavious could be removed potentially
    #         raise Error("Mask and array must have the same shape")

    #     for i in range(mask.size):
    #         if mask._buf.load[width=1](i):
    #             print(value)
    #             self._buf.store[width=1](i, value)

    fn __setitem__(mut self, owned *slices: Slice, val: NDArray[dtype]) raises:
        """
        Retreive slices of an array from variadic slices.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).
        """
        var slice_list: List[Slice] = List[Slice]()
        for i in range(slices.__len__()):
            slice_list.append(slices[i])
        self.__setitem__(slices=slice_list, val=val)

    fn __setitem__(
        mut self, owned slices: List[Slice], val: NDArray[dtype]
    ) raises:
        """
        Sets the slices of an array from list of slices and array.

        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).
        """
        var n_slices: Int = len(slices)
        var ndims: Int = 0
        var count: Int = 0
        var spec: List[Int] = List[Int]()
        var slice_list: List[Slice] = self._adjust_slice_(slices)
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
            # if slice_list[j].step is None:
            #     raise Error(String("Step of slice is None."))
            var slice_len: Int = (
                (slice_list[j].end.value() - slice_list[j].start.value())
                / slice_list[j].step.or_else(1)
            ).__int__()
            nshape.append(slice_len)
            nnum_elements *= slice_len
            # if slice_list[j].step is None:
            #     raise Error(String("Step of slice is None."))
            ncoefficients.append(
                self.strides[j] * slice_list[j].step.or_else(1)
            )
            j += 1

        # We can remove this check after we have support for broadcasting
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
        if self.order == "C":
            noffset = 0
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):  # temp
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slice_list.__len__()):
                noffset += slice_list[i].start.value() * self.strides[i]
        elif self.order == "F":
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

    ### compiler doesn't accept this.
    # fn __setitem__(self, owned *slices: Variant[Slice, Int], val: NDArray[dtype]) raises:
    #     """
    #     Get items by a series of either slices or integers.
    #     """
    #     var n_slices: Int = slices.__len__()
    #     if n_slices > self.ndim:
    #         raise Error("Error: No of slices greater than rank of array")
    #     var slice_list: List[Slice] = List[Slice]()

    #     var count_int = 0
    #     for i in range(len(slices)):
    #         if slices[i].isa[Slice]():
    #             slice_list.append(slices[i]._get_ptr[Slice]()[0])
    #         elif slices[i].isa[Int]():
    #             count_int += 1
    #             var int: Int = slices[i]._get_ptr[Int]()[0]
    #             slice_list.append(Slice(int, int + 1))

    #     if n_slices < self.ndim:
    #         for i in range(n_slices, self.ndim):
    #             var size_at_dim: Int = self.shape[i]
    #             slice_list.append(Slice(0, size_at_dim))

    #     self.__setitem__(slices=slice_list, val=val)

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
            self.set(int(index.get(i)), rebind[Scalar[dtype]](val.get(i)))

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
            if mask._buf.load(i):
                self._buf.store(i, val._buf.load(i))

    # ===-------------------------------------------------------------------===#
    # Getter dunders
    # ===-------------------------------------------------------------------===#
    fn get(self, index: Int) raises -> SIMD[dtype, 1]:
        """
        Linearly retreive a value from the underlying Pointer.

        Example:
        ```console
        > Array.get(15)
        ```
        returns the item of index 15 from the array's data buffer.

        Not that it is different from `item()` as `get` does not checked
        against C-order or F-order.
        ```console
        > # A is a 3x3 matrix, F-order (column-major)
        > A.get(3)  # Row 0, Col 1
        > A.item(3)  # Row 1, Col 0
        ```
        """
        if index >= self.size:
            raise Error(
                String(
                    "Invalid index: index out of bound!\n"
                    "The index is {}."
                    "The size of the array is {}"
                ).format(index, self.size)
            )
        if index >= 0:
            return self._buf.load[width=1](index)
        else:
            return self._buf.load[width=1](index + self.size)

    fn __getitem__(self, idx: Int) raises -> Self:
        """
        Retreive a slice of the array corresponding to the index at the first dimension.

        Example:
            `arr[1]` returns the second row of the array.
        """

        var slice_list = List[Slice]()
        slice_list.append(Slice(idx, idx + 1))

        # 0-d array always return itself
        if self.ndim == 0:
            return self

        if self.ndim > 1:
            for i in range(1, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim))

        var narr: Self = self.__getitem__(slice_list)

        if self.ndim == 1:
            narr.ndim = 0
            narr.shape._buf[0] = 0

        return narr

    fn __getitem__(self, index: Idx) raises -> SIMD[dtype, 1]:
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
        var idx: Int = _get_index(index, self.coefficient)
        return self._buf.load[width=1](idx)

    fn _adjust_slice_(self, slice_list: List[Slice]) raises -> List[Slice]:
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
            # if slice_list[i].step is None:
            #     raise Error(String("Step of slice is None."))
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
                slice_list.append(Slice(0, self.shape[i]))

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

        var slices: List[Slice] = self._adjust_slice_(slice_list)
        for i in range(slices.__len__()):
            if (
                slices[i].start.value() >= self.shape[i]
                or slices[i].end.value() > self.shape[i]
            ):
                raise Error("Error: Slice value exceeds the array shape")
            # if slices[i].step is None:
            #     raise Error(String("Step of slice is None."))
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
            # if slices[j].step is None:
            #     raise Error(String("Step of slice is None."))
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
        if self.order == "C":
            noffset = 0
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):  # temp
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slices.__len__()):
                noffset += slices[i].start.value() * self.strides[i]

        elif self.order == "F":
            noffset = 0
            nstrides.append(1)
            for i in range(0, ndims - 1):
                nstrides.append(nstrides[i] * nshape[i])
            for i in range(slices.__len__()):
                noffset += slices[i].start.value() * self.strides[i]

        var narr = Self(
            ndims,
            noffset,
            nnum_elements,
            nshape,
            nstrides,
            ncoefficients,
            order=self.order,
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
                slice_list.append(Slice(int, int + 1))

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                var size_at_dim: Int = self.shape[i]
                slice_list.append(Slice(0, size_at_dim))

        var narr: Self = self.__getitem__(slice_list)

        if count_int == self.ndim:
            narr.ndim = 0
            narr.shape._buf[0] = 0

        return narr

    fn __getitem__(self, index: List[Int]) raises -> Self:
        """
        Get items of array from a list of indices.

        It always gets the first dimension.

        Example:
        ```console
        > var A = nm.NDArray[nm.i8](3,random=True)
        > print(A)
        [       14      97      -59     ]
        1-D array  Shape: [3]  DType: int8
        >
        > print(A[List[Int](2,1,0,1,2)])
        [       -59     97      14      97      -59     ]
        1-D array  Shape: [5]  DType: int8
        >
        > var B = nm.NDArray[nm.i8](3, 3,random=True)
        > print(B)
        [[      -4      112     -94     ]
        [      -48     -40     66      ]
        [      -2      -94     -18     ]]
        2-D array  Shape: [3, 3]  DType: int8
        >
        > print(B[List[Int](2,1,0,1,2)])
        [[      -2      -94     -18     ]
        [      -48     -40     66      ]
        [      -4      112     -94     ]
        [      -48     -40     66      ]
        [      -2      -94     -18     ]]
        2-D array  Shape: [5, 3]  DType: int8
        >
        > var C = nm.NDArray[nm.i8](3, 3, 3, random=True)
        > print(C)
        [[[     -126    -88     -79     ]
        [     14      78      99      ]
        [     -32     3       -42     ]]
        [[     56      -45     -71     ]
        [     -13     18      -102    ]
        [     4       83      26      ]]
        [[     61      -73     86      ]
        [     -125    -84     66      ]
        [     32      21      53      ]]]
        3-D array  Shape: [3, 3, 3]  DType: int8
        >
        > print(C[List[Int](2,1,0,1,2)])
        [[[     61      -73     86      ]
        [     -125    -84     66      ]
        [     32      21      53      ]]
        [[     56      -45     -71     ]
        [     -13     18      -102    ]
        [     4       83      26      ]]
        [[     -126    -88     -79     ]
        [     14      78      99      ]
        [     -32     3       -42     ]]
        [[     56      -45     -71     ]
        [     -13     18      -102    ]
        [     4       83      26      ]]
        [[     61      -73     86      ]
        [     -125    -84     66      ]
        [     32      21      53      ]]]
        3-D array  Shape: [5, 3, 3]  DType: int8
        ```

        Args:
            index: List[Int].

        Returns:
            NDArray with items from the list of indices.
        """

        # Shape of the result should be
        # Number of indice * shape from dim-1
        # So just change the first number of the ndshape
        var ndshape = self.shape
        ndshape[0] = len(index)
        ndsize = 1
        for i in range(ndshape.ndim):
            ndsize *= int(ndshape._buf[i])
        var result = NDArray[dtype](ndshape)
        var size_per_item = ndsize // len(index)

        # Fill in the values
        for i in range(len(index)):
            for j in range(size_per_item):
                result._buf.store(
                    i * size_per_item + j, self[int(index[i])].item(j)
                )

        return result

    fn __getitem__(self, index: NDArray[DType.index]) raises -> Self:
        """
        Get items of array from an array of indices.

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

        var new_index = List[Int]()
        for i in index:
            new_index.append(int(i.item(0)))

        return self.__getitem__(new_index)

    fn __getitem__(self, mask: NDArray[DType.bool]) raises -> Self:
        """
        Get items of array corresponding to a mask.

        Example:
            ```
            var A = numojo.core.NDArray[numojo.i16](6, random=True)
            var mask = A > 0
            print(A)
            print(mask)
            print(A[mask])
            ```

        Args:
            mask: NDArray with Dtype.bool.

        Returns:
            NDArray with items from the mask.
        """
        var true: List[Int] = List[Int]()
        for i in range(mask.size):
            if mask._buf.load[width=1](i):
                true.append(i)

        var result = Self(Shape(true.__len__()))
        for i in range(true.__len__()):
            result._buf.store(i, self.get(true[i]))

        return result

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    # We should make a version that checks nonzero/not_nan
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
            return int(self.get(0))
        else:
            raise (
                "Only 0-D arrays or length-1 arrays can be converted to scalars"
            )

    fn __pos__(self) raises -> Self:
        """
        Unary positve returens self unless boolean type.
        """
        if self.dtype is DType.bool:
            raise Error(
                "ndarray:NDArrray:__pos__: pos does not except bool type arrays"
            )
        return self

    fn __neg__(self) raises -> Self:
        """
        Unary negative returens self unless boolean type.

        For bolean use `__invert__`(~)
        """
        if self.dtype is DType.bool:
            raise Error(
                "ndarray:NDArrray:__pos__: pos does not except bool type arrays"
            )
        return self * -1.0

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise equivelence.
        """
        return comparison.equal[dtype](self, other)

    @always_inline("nodebug")
    fn __eq__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise equivelence between scalar and Array.
        """
        return comparison.equal[dtype](self, other)

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

    fn __add__(mut self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `array + scalar`.
        """
        return math.add[dtype](self, other)

    fn __add__(mut self, other: Self) raises -> Self:
        """
        Enables `array + array`.
        """
        return math.add[dtype](self, other)

    fn __radd__(mut self, rhs: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar + array`.
        """
        return math.add[dtype](self, rhs)

    # TODO make an inplace version of arithmetic functions for the i dunders
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

    fn __sub__(self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `array - scalar`.
        """
        return math.sub[dtype](self, other)

    fn __sub__(self, other: Self) raises -> Self:
        """
        Enables `array - array`.
        """
        return math.sub[dtype](self, other)

    fn __rsub__(self, s: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar - array`.
        """
        return math.sub[dtype](s, self)

    fn __isub__(mut self, s: SIMD[dtype, 1]) raises:
        """
        Enables `array -= scalar`.
        """
        self = self - s

    fn __isub__(mut self, s: Self) raises:
        """
        Enables `array -= array`.
        """
        self = self - s

    fn __matmul__(self, other: Self) raises -> Self:
        return matmul_parallelized(self, other)

    fn __mul__(self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `array * scalar`.
        """
        return math.mul[dtype](self, other)

    fn __mul__(self, other: Self) raises -> Self:
        """
        Enables `array * array`.
        """
        return math.mul[dtype](self, other)

    fn __rmul__(self, s: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar * array`.
        """
        return math.mul[dtype](self, s)

    fn __imul__(mut self, s: SIMD[dtype, 1]) raises:
        """
        Enables `array *= scalar`.
        """
        self = self * s

    fn __imul__(mut self, s: Self) raises:
        """
        Enables `array *= array`.
        """
        self = self * s

    fn __abs__(self) -> Self:
        return abs(self)

    fn __invert__(self) raises -> Self:
        """
        Elementwise inverse (~ or not), only for bools and integral types.
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
            result._buf.store(
                index,
                self._buf.load[width=simd_width](index)
                ** p.load[width=simd_width](index),
            )

        vectorize[vectorized_pow, self.width](self.size)
        return result

    fn __ipow__(mut self, p: Int):
        self = self.__pow__(p)

    fn _elementwise_pow(self, p: Int) -> Self:
        var new_vec = self

        @parameter
        fn array_scalar_vectorize[simd_width: Int](index: Int) -> None:
            new_vec._buf.store(
                index, pow(self._buf.load[width=simd_width](index), p)
            )

        vectorize[array_scalar_vectorize, self.width](self.size)
        return new_vec

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

    fn __rtruediv__(self, s: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar / array`.
        """
        return math.div[dtype](s, self)

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

    fn __rfloordiv__(self, s: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar // array`.
        """
        return math.floor_div[dtype](s, self)

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

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#
    fn __str__(self) -> String:
        """
        Enables str(array).
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        try:
            writer.write(
                self._array_to_string(0, 0)
                + "\n"
                + str(self.ndim)
                + "-D array  "
                + self.shape.__str__()
                + "  DType: "
                + self.dtype.__str__()
            )
        except e:
            writer.write("Cannot convert array to string")

    fn __repr__(self) -> String:
        """
        Compute the "official" string representation of NDArray.
        An example is:
        ```
        fn main() raises:
            var A = NDArray[DType.int8](List[Scalar[DType.int8]](14,97,-59,-4,112,), shape=List[Int](5,))
            print(repr(A))
        ```
        It prints what can be used to construct the array itself:
        ```console
        NDArray[DType.int8](List[Scalar[DType.int8]](14,97,-59,-4,112,), shape=List[Int](5,))
        ```.
        """
        try:
            var result: String = str("NDArray[DType.") + str(self.dtype) + str(
                "](List[Scalar[DType."
            ) + str(self.dtype) + str("]](")
            if self.size > 6:
                for i in range(6):
                    result = result + str(self.load[width=1](i)) + str(",")
                result = result + " ... "
            else:
                for i in self:
                    result = result + str(i) + str(",")
            result = result + str("), shape=List[Int](")
            for i in range(self.shape.ndim):
                result = result + str(self.shape._buf[i]) + ","
            result = result + str("))")
            return result
        except e:
            print("Cannot convert array to string", e)
            return ""

    # Should len be size or number of dimensions instead of the first dimension shape?
    fn __len__(self) -> Int:
        return int(self.size)

    fn __iter__(self) raises -> _NDArrayIter[__origin_of(self), dtype]:
        """Iterate over elements of the NDArray, returning copied value.

        Returns:
            An iterator of NDArray elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _NDArrayIter[__origin_of(self), dtype](
            array=self,
            length=self.shape[0],
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
            array=self,
            length=self.shape[0],
        )

    fn _array_to_string(self, dimension: Int, offset: Int) raises -> String:
        if self.ndim == 0:
            return str(self.item(0))
        if dimension == self.ndim - 1:
            var result: String = str("[\t")
            var number_of_items = self.shape[dimension]
            if number_of_items <= 6:  # Print all items
                for i in range(number_of_items):
                    result = (
                        result
                        + self.load[width=1](
                            offset + i * self.strides[dimension]
                        ).__str__()
                    )
                    result = result + "\t"
            else:  # Print first 3 and last 3 items
                for i in range(3):
                    result = (
                        result
                        + self.load[width=1](
                            offset + i * self.strides[dimension]
                        ).__str__()
                    )
                    result = result + "\t"
                result = result + "...\t"
                for i in range(number_of_items - 3, number_of_items):
                    result = (
                        result
                        + self.load[width=1](
                            offset + i * self.strides[dimension]
                        ).__str__()
                    )
                    result = result + "\t"
            result = result + "]"
            return result
        else:
            var result: String = str("[")
            var number_of_items = self.shape[dimension]
            if number_of_items <= 6:  # Print all items
                for i in range(number_of_items):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.strides[dimension].__int__(),
                        )
                    if i > 0:
                        result = (
                            result
                            + str(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset + i * self.strides[dimension].__int__(),
                            )
                        )
                    if i < (number_of_items - 1):
                        result = result + "\n"
            else:  # Print first 3 and last 3 items
                for i in range(3):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.strides[dimension].__int__(),
                        )
                    if i > 0:
                        result = (
                            result
                            + str(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset + i * self.strides[dimension].__int__(),
                            )
                        )
                    if i < (number_of_items - 1):
                        result += "\n"
                result = result + "...\n"
                for i in range(number_of_items - 3, number_of_items):
                    result = (
                        result
                        + str(" ") * (dimension + 1)
                        + self._array_to_string(
                            dimension + 1,
                            offset + i * self.strides[dimension].__int__(),
                        )
                    )
                    if i < (number_of_items - 1):
                        result = result + "\n"
            result = result + "]"
            return result

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
            sum = sum + self.get(i) * other.get(i)
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
                    Idx(row, col),
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
            buffer.set(i, self._buf.load[width=1](i + id * width))
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
            buffer.set(i, self._buf.load[width=1](id + i * width))
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
                new_matrix.set(
                    col + row * other.shape[1],
                    self.row(row).vdot(other.col(col)),
                )
        return new_matrix

    fn num_elements(self) -> Int:
        """
        Function to retreive size (compatability).
        """
        return self.size

    # should this return the List[Int] shape and self.shape be used instead of making it a no input function call?
    # * We fix return dtype of this array shape for the linalg solve module.
    # fn shape(self) -> NDArrayShape[i32]:
    #     """
    #     Get the shape as an NDArray Shape.

    #     To get a list of shape call this then list
    #     """
    #     return self.shape

    fn load[width: Int = 1](self, index: Int) -> SIMD[dtype, width]:
        """
        Loads a SIMD element of size `width` at the given index `index`.
        """
        return self._buf.load[width=width](index)

    # # TODO: we should add checks to make sure user don't load out of bound indices, but that will overhead, figure out later
    fn load[width: Int = 1](self, *index: Int) raises -> SIMD[dtype, width]:
        """
        Loads a SIMD element of size `width` at given variadic indices argument.
        """
        var idx: Int = _get_index(index, self.coefficient)
        return self._buf.load[width=width](idx)

    fn store[width: Int](mut self, index: Int, val: SIMD[dtype, width]):
        """
        Stores the SIMD element of size `width` at index `index`.
        """
        self._buf.store(index, val)

    fn store[
        width: Int = 1
    ](mut self, *index: Int, val: SIMD[dtype, width]) raises:
        """
        Stores the SIMD element of size `width` at the given variadic indices argument.
        """
        var idx: Int = _get_index(index, self.coefficient)
        self._buf.store(idx, val)

    # # not urgent: argpartition, byteswap, choose, conj, dump, getfield
    # # partition, put, repeat, searchsorted, setfield, squeeze, swapaxes, take,
    # # tobyets, tofile, view
    # TODO: Implement axis parameter for all

    # ===-------------------------------------------------------------------===#
    # Operations along an axis
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
            result = result and allb(
                (self._buf + idx).strided_load[width=simd_width](1)
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
            result = result or anyb(
                (self._buf + idx).strided_load[width=simd_width](1)
            )

        vectorize[vectorized_any, self.width](self.size)
        return result

    fn argmax(self) -> Int:
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

    fn argmin(self) -> Int:
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

    fn astype[type: DType](self) raises -> NDArray[type]:
        """
        Convert type of array.
        """
        # I wonder if we can do this operation inplace instead of allocating memory.
        var narr: NDArray[type] = NDArray[type](self.shape, order=self.order)

        @parameter
        if type == DType.bool:

            @parameter
            fn vectorized_astype[simd_width: Int](idx: Int) -> None:
                (narr.unsafe_ptr() + idx).strided_store[width=simd_width](
                    self.load[simd_width](idx).cast[type](), 1
                )

            vectorize[vectorized_astype, self.width](self.size)
        else:

            @parameter
            if self.dtype == DType.bool:

                @parameter
                fn vectorized_astypenb_from_b[
                    simd_width: Int
                ](idx: Int) -> None:
                    narr.store[simd_width](
                        idx,
                        (self._buf + idx)
                        .strided_load[width=simd_width](1)
                        .cast[type](),
                    )

                vectorize[vectorized_astypenb_from_b, self.width](self.size)
            else:

                @parameter
                fn vectorized_astypenb[simd_width: Int](idx: Int) -> None:
                    narr.store[simd_width](
                        idx, self.load[simd_width](idx).cast[type]()
                    )

                vectorize[vectorized_astypenb, self.width](self.size)

        return narr

    # fn clip(self):
    #     pass

    # fn compress(self):
    #     pass

    # fn copy(self):
    #     pass

    fn cumprod(self) -> Scalar[dtype]:
        """
        Cumulative product of a array.

        Returns:
            The cumulative product of the array as a SIMD Value of `dtype`.
        """
        return cumprod[dtype](self)

    fn cumsum(self) -> Scalar[dtype]:
        """
        Cumulative Sum of a array.

        Returns:
            The cumulative sum of the array as a SIMD Value of `dtype`.
        """
        return cumsum[dtype](self)

    fn diagonal(self):
        pass

    fn fill(mut self, val: Scalar[dtype]):
        """
        Fill all items of array with value.
        """

        for i in range(self.size):
            self._buf[i] = val

    fn flatten(mut self) raises:
        """
        Convert shape of array to one dimensional.
        """
        self.shape = NDArrayShape(self.size, size=self.size)
        self.strides = NDArrayStrides(shape=self.shape, offset=0)

    fn item(self, *index: Int) raises -> SIMD[dtype, 1]:
        """
        Return the scalar at the coordinates.

        If one index is given, get the i-th item of the array.
        It first scans over the first row, even it is a colume-major array.

        If more than one index is given, the length of the indices must match
        the number of dimensions of the array.

        Example:
        ```console
        > var A = nm.NDArray[dtype](3, 3, random=True, order="F")
        > print(A)
        [[      14      -4      -48     ]
        [      97      112     -40     ]
        [      -59     -94     66      ]]
        2-D array  Shape: [3, 3]  DType: int8

        > for i in A:
        >     print(i)  # Return rows
        [       14      -4      -48     ]
        1-D array  Shape: [3]  DType: int8
        [       97      112     -40     ]
        1-D array  Shape: [3]  DType: int8
        [       -59     -94     66      ]
        1-D array  Shape: [3]  DType: int8

        > for i in range(A.size()):
        >    print(A.item(i))  # Return 0-d arrays
        c strides Stride: [3, 1]
        14
        c strides Stride: [3, 1]
        -4
        c strides Stride: [3, 1]
        -48
        c strides Stride: [3, 1]
        97
        c strides Stride: [3, 1]
        112
        c strides Stride: [3, 1]
        -40
        c strides Stride: [3, 1]
        -59
        c strides Stride: [3, 1]
        -94
        c strides Stride: [3, 1]
        66
        ==============================
        ```

        Args:
            index: The coordinates of the item.

        Returns:
            A scalar matching the dtype of the array.
        """

        # If one index is given
        if index.__len__() == 1:
            if index[0] < self.size:
                if (
                    self.order == "F"
                ):  # column-major should be converted to row-major
                    # The following code can be taken out as a function that
                    # convert any index to coordinates according to the order
                    var c_stride = NDArrayStrides(shape=self.shape)
                    var c_coordinates = List[Int]()
                    var idx: Int = index[0]
                    for i in range(c_stride.ndim):
                        var coordinate = idx // c_stride[i]
                        idx = idx - c_stride[i] * coordinate
                        c_coordinates.append(coordinate)
                    return self._buf.load[width=1](
                        _get_index(c_coordinates, self.strides)
                    )

                return self._buf.load[width=1](index[0])
            else:
                raise Error(
                    String(
                        "Error: Elements of `index` ({}) \n"
                        "exceed the array size ({})"
                    ).format(index[0], self.size)
                )

        # If more than one index is given
        if index.__len__() != self.ndim:
            raise Error(
                String(
                    "Error: Length of Indices ({}) \n"
                    "do not match the shape ({})"
                ).format(index.__len__(), self.ndim)
            )
        for i in range(index.__len__()):
            if index[i] >= self.shape[i]:
                raise Error(
                    String(
                        "Error: Elements of `index` ({}) \n"
                        "exceed the array shape ({}) \n"
                        "for {}-th dimension."
                    ).format(index[i], self.shape[i], i)
                )
        return self._buf.load[width=1](_get_index(index, self.strides))

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
                if (
                    self.order == "F"
                ):  # column-major should be converted to row-major
                    # The following code can be taken out as a function that
                    # convert any index to coordinates according to the order
                    var c_stride = NDArrayStrides(shape=self.shape)
                    var c_coordinates = List[Int]()
                    for i in range(c_stride.ndim):
                        var coordinate = idx // c_stride[i]
                        idx = idx - c_stride[i] * coordinate
                        c_coordinates.append(coordinate)
                    self._buf.store(
                        _get_index(c_coordinates, self.strides), item
                    )

                self._buf.store(idx, item)
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
            self._buf.store(_get_index(indices, self.strides), item)

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
                slices.append(Slice(0, shape[i]))
            else:
                slices.append(Slice(0, 0))

        slices[axis] = Slice(0, 1)
        var result: NDArray[dtype] = self[slices]
        for i in range(1, axis_size):
            slices[axis] = Slice(i, i + 1)
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
                slices.append(Slice(0, shape[i]))
            else:
                slices.append(Slice(0, 0))

        slices[axis] = Slice(0, 1)
        var result: NDArray[dtype] = self[slices]
        for i in range(1, axis_size):
            slices[axis] = Slice(i, i + 1)
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

    fn prod(self: Self, axis: Int) raises -> Self:
        """
        Product of array elements over a given axis.
        Args:
            array: NDArray.
            axis: The axis along which the product is performed.
        Returns:
            An NDArray.
        """

        return prod(self, axis)

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
        self = flatten(self)
        sorting._sort_inplace(
            self,
            I,
        )

    fn sort(mut self, owned axis: Int) raises:
        """
        Sort NDArray along the given axis using quick sort method.
        It is not guaranteed to be unstable.

        When no axis is given, the array is flattened before sorting.

        See `numojo.sorting.sort` for more information.
        """
        var I = NDArray[DType.index](self.shape)
        sorting._sort_inplace(self, I, axis=axis)

    fn sum(self: Self, axis: Int) raises -> Self:
        """
        Sum of array elements over a given axis.
        Args:
            axis: The axis along which the sum is performed.
        Returns:
            An NDArray.
        """
        return sum(self, axis)

    fn tolist(self) -> List[Scalar[dtype]]:
        """
        Convert NDArray to a 1-D List.

        Returns:
            A 1-D List.
        """
        var result: List[Scalar[dtype]] = List[Scalar[dtype]]()
        for i in range(self.size):
            result.append(self._buf[i])
        return result

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

    # Technically it only changes the ArrayDescriptor and not the fundamental data
    fn reshape(mut self, *shape: Int, order: String = "C") raises:
        """
        Reshapes the NDArray to given Shape.

        Args:
            shape: Variadic list of shape.
            order: Order of the array - Row major `C` or Column major `F`.
        """
        var s: VariadicList[Int] = shape
        reshape[dtype](self, s, order=order)

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        """
        Retreive pointer without taking ownership.
        """
        return self._buf

    fn to_numpy(self) raises -> PythonObject:
        """
        Convert to a numpy array.
        """
        return to_numpy(self)


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
    var array: NDArray[dtype]
    var length: Int

    fn __init__(
        mut self,
        array: NDArray[dtype],
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.array = array

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) raises -> NDArray[dtype]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.array.__getitem__(current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.array.__getitem__(current_index)

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
