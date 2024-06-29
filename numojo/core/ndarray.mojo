"""
# ===----------------------------------------------------------------------=== #
# Implements ROW MAJOR N-DIMENSIONAL ARRAYS
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #
"""

"""
# TODO
1) Add NDArray, NDArrayShape constructor overload for List, VariadicList types etc to cover more cases
3) Generalize mdot, rdot to take any IxJx...xKxL and LxMx...xNxP matrix and matmul it into IxJx..xKxMx...xNxP array.
4) Vectorize row(), col() to retrieve rows and columns for 2D arrays
5) Add __getitem__() overload for (Slice, Int)
7) Add vectorization for _get_index
8) Write more explanatory Error("") statements
9) Vectorize the for loops inside getitem or move these checks to compile time to try and remove the overhead from constantly checking
10) Add List[Int] and Variadic[Int] Shape args for __init__ to make it more flexible
"""

from random import rand
from builtin.math import pow
from builtin.bool import all as allb
from builtin.bool import any as anyb
from algorithm import parallelize, vectorize

import . _array_funcs as _af
from ..math.statistics.stats import mean, prod, sum
from ..math.statistics.cumulative_reduce import (
    cumsum,
    cumprod,
    cummean,
    maxT,
    minT,
)
from ..math.check import any, all
from ..math.arithmetic import abs
from .ndarray_utils import _get_index, _traverse_iterative, to_numpy
from .utility_funcs import is_inttype


@register_passable("trivial")
struct NDArrayShape[dtype: DType = DType.int32](Stringable):
    """Implements the NDArrayShape."""

    # Fields
    var _size: Int  # The total no of elements in the corresponding array
    var _shape: DTypePointer[dtype]  # The shape of the corresponding array
    var _len: Int  # The length of _shape

    @always_inline("nodebug")
    fn __init__(inout self, *shape: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions.

        Args:
            shape: Variable number of integers representing the shape dimensions.
        """
        self._size = 1
        self._len = len(shape)
        self._shape = DTypePointer[dtype].alloc(len(shape))
        memset_zero(self._shape, len(shape))
        for i in range(len(shape)):
            self._shape[i] = shape[i]
            self._size *= shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, *shape: Int, size: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions and a specified size.

        Args:
            shape: Variable number of integers representing the shape dimensions.
            size: The total number of elements in the array.
        """
        self._size = size
        self._len = len(shape)
        self._shape = DTypePointer[dtype].alloc(len(shape))
        memset_zero(self._shape, len(shape))
        var count: Int = 1
        for i in range(len(shape)):
            self._shape[i] = shape[i]
            count *= shape[i]
        if count != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(inout self, shape: List[Int]):
        """
        Initializes the NDArrayShape with a list of shape dimensions.

        Args:
            shape: A list of integers representing the shape dimensions.
        """
        self._size = 1
        self._len = len(shape)
        self._shape = DTypePointer[dtype].alloc(len(shape))
        memset_zero(self._shape, len(shape))
        for i in range(len(shape)):
            self._shape[i] = shape[i]
            self._size *= shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, shape: List[Int], size: Int) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions and a specified size.

        Args:
            shape: A list of integers representing the shape dimensions.
            size: The specified size of the NDArrayShape.
        """
        self._size = (
            size  # maybe I should add a check here to make sure it matches
        )
        self._len = len(shape)
        self._shape = DTypePointer[dtype].alloc(len(shape))
        memset_zero(self._shape, len(shape))
        var count: Int = 1
        for i in range(len(shape)):
            self._shape[i] = shape[i]
            count *= shape[i]
        if count != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(inout self, shape: VariadicList[Int]):
        """
        Initializes the NDArrayShape with a list of shape dimensions.

        Args:
            shape: A list of integers representing the shape dimensions.
        """
        self._size = 1
        self._len = len(shape)
        self._shape = DTypePointer[dtype].alloc(len(shape))
        memset_zero(self._shape, len(shape))
        for i in range(len(shape)):
            self._shape[i] = shape[i]
            self._size *= shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, shape: VariadicList[Int], size: Int) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions and a specified size.

        Args:
            shape: A list of integers representing the shape dimensions.
            size: The specified size of the NDArrayShape.
        """
        self._size = (
            size  # maybe I should add a check here to make sure it matches
        )
        self._len = len(shape)
        self._shape = DTypePointer[dtype].alloc(len(shape))
        memset_zero(self._shape, len(shape))
        var count: Int = 1
        for i in range(len(shape)):
            self._shape[i] = shape[i]
            count *= shape[i]
        if count != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(inout self, shape: NDArrayShape) raises:
        """
        Initializes the NDArrayShape with another NDArrayShape.

        Args:
            shape: Another NDArrayShape to initialize from.
        """
        self._size = shape._size
        self._len = shape._len
        self._shape = DTypePointer[dtype].alloc(shape._len)
        memset_zero(self._shape, shape._len)
        for i in range(shape._len):
            self._shape[i] = shape[i]

    fn __copy__(inout self, other: Self):
        self._size = other._size
        self._len = other._len
        self._shape = DTypePointer[dtype].alloc(other._len)
        memcpy(self._shape, other._shape, other._len)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        if index >= self._len:
            raise Error("Index out of bound")
        if index >= 0:
            return self._shape[index].__int__()
        else:
            return self._shape[self._len + index].__int__()

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, val: Int) raises:
        if index >= self._len:
            raise Error("Index out of bound")
        if index >= 0:
            self._shape[index] = val
        else:
            self._shape[self._len + index] = val

    @always_inline("nodebug")
    fn size(self) -> Int:
        return self._size

    @always_inline("nodebug")
    fn len(self) -> Int:
        return self._len

    @always_inline("nodebug")
    fn __str__(self: Self) -> String:
        var result: String = "Shape: ["
        for i in range(self._len):
            if i == self._len - 1:
                result += self._shape[i].__str__()
            else:
                result += self._shape[i].__str__() + ", "
        return result + "]"

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Bool:
        for i in range(self._len):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, val: Int) raises -> Bool:
        for i in range(self._len):
            if self[i] == val:
                return True
        return False

    # can be used for vectorized index calculation
    @always_inline("nodebug")
    fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
        # if index >= self._len:
        # raise Error("Index out of bound")
        return self._shape.load[width=width](index)

    # can be used for vectorized index retrieval
    @always_inline("nodebug")
    fn store[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[dtype, width]) raises:
        # if index >= self._len:
        #     raise Error("Index out of bound")
        self._shape.store[width=width](index, val)

    @always_inline("nodebug")
    fn load_int(self, index: Int) -> Int:
        return self._shape.load[width=1](index).__int__()

    @always_inline("nodebug")
    fn store_int(inout self, index: Int, val: Int):
        self._shape.store[width=1](index, val)


@register_passable("trivial")
struct NDArrayStride[dtype: DType = DType.int32](Stringable):
    """Implements the NDArrayStride."""

    # Fields
    var _offset: Int
    var _stride: DTypePointer[dtype]
    var _len: Int

    @always_inline("nodebug")
    fn __init__(
        inout self, *stride: Int, offset: Int = 0
    ):  # maybe we should add checks for offset?
        self._offset = offset
        self._len = stride.__len__()
        self._stride = DTypePointer[dtype].alloc(stride.__len__())
        for i in range(stride.__len__()):
            self._stride[i] = stride[i]

    @always_inline("nodebug")
    fn __init__(inout self, stride: List[Int], offset: Int = 0):
        self._offset = offset
        self._len = stride.__len__()
        self._stride = DTypePointer[dtype].alloc(self._len)
        memset_zero(self._stride, self._len)
        for i in range(self._len):
            self._stride[i] = stride[i]

    @always_inline("nodebug")
    fn __init__(inout self, stride: VariadicList[Int], offset: Int = 0):
        self._offset = offset
        self._len = stride.__len__()
        self._stride = DTypePointer[dtype].alloc(self._len)
        memset_zero(self._stride, self._len)
        for i in range(self._len):
            self._stride[i] = stride[i]

    @always_inline("nodebug")
    fn __init__(inout self, stride: NDArrayStride):
        self._offset = stride._offset
        self._len = stride._len
        self._stride = DTypePointer[dtype].alloc(stride._len)
        for i in range(self._len):
            self._stride[i] = stride._stride[i]

    @always_inline("nodebug")
    fn __init__(
        inout self, stride: NDArrayStride, offset: Int = 0
    ):  # separated two methods to remove if condition
        self._offset = offset
        self._len = stride._len
        self._stride = DTypePointer[dtype].alloc(stride._len)
        for i in range(self._len):
            self._stride[i] = stride._stride[i]

    @always_inline("nodebug")
    fn __init__(
        inout self, *shape: Int, offset: Int = 0, order: String = "C"
    ) raises:
        self._offset = offset
        self._len = shape.__len__()
        self._stride = DTypePointer[dtype].alloc(self._len)
        memset_zero(self._stride, self._len)
        if order == "C":
            for i in range(self._len):
                var temp: Int = 1
                for j in range(i + 1, self._len):
                    temp = temp * shape[j]
                self._stride[i] = temp
        elif order == "F":
            self._stride[0] = 1
            for i in range(0, self._len - 1):
                self._stride[i + 1] = self._stride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        inout self, shape: List[Int], offset: Int = 0, order: String = "C"
    ) raises:
        self._offset = offset
        self._len = shape.__len__()
        self._stride = DTypePointer[dtype].alloc(self._len)
        memset_zero(self._stride, self._len)
        if order == "C":
            for i in range(self._len):
                var temp: Int = 1
                for j in range(i + 1, self._len):
                    temp = temp * shape[j]
                self._stride[i] = temp
        elif order == "F":
            self._stride[0] = 1
            for i in range(0, self._len - 1):
                self._stride[i + 1] = self._stride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: VariadicList[Int],
        offset: Int = 0,
        order: String = "C",
    ) raises:
        self._offset = offset
        self._len = shape.__len__()
        self._stride = DTypePointer[dtype].alloc(self._len)
        memset_zero(self._stride, self._len)
        if order == "C":
            for i in range(self._len):
                var temp: Int = 1
                for j in range(i + 1, self._len):
                    temp = temp * shape[j]
                self._stride[i] = temp
        elif order == "F":
            self._stride[0] = 1
            for i in range(0, self._len - 1):
                self._stride[i + 1] = self._stride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        inout self,
        owned shape: NDArrayShape,
        offset: Int = 0,
        order: String = "C",
    ) raises:
        self._offset = offset
        self._len = shape._len
        self._stride = DTypePointer[dtype].alloc(shape._len)
        memset_zero(self._stride, shape._len)
        if order == "C":
            if shape._len == 1:
                self._stride[0] = 1
            else:
                for i in range(shape._len):
                    var temp: Int = 1
                    for j in range(i + 1, shape._len):
                        temp = temp * shape[j]
                    self._stride[i] = temp
        elif order == "F":
            self._stride[0] = 1
            for i in range(0, self._len - 1):
                self._stride[i + 1] = self._stride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    fn __copy__(inout self, other: Self):
        self._offset = other._offset
        self._len = other._len
        self._stride = DTypePointer[dtype].alloc(other._len)
        memcpy(self._stride, other._stride, other._len)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        if index >= self._len:
            raise Error("Index out of bound")
        if index >= 0:
            return self._stride[index].__int__()
        else:
            return self._stride[self._len + index].__int__()

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, val: Int) raises:
        if index >= self._len:
            raise Error("Index out of bound")
        if index >= 0:
            self._stride[index] = val
        else:
            self._stride[self._len + index] = val

    @always_inline("nodebug")
    fn len(self) -> Int:
        return self._len

    @always_inline("nodebug")
    fn __str__(self: Self) -> String:
        var result: String = "Stride: ["
        for i in range(self._len):
            if i == self._len - 1:
                result += self._stride[i].__str__()
            else:
                result += self._stride[i].__str__() + ", "
        return result + "]"

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Bool:
        for i in range(self._len):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, val: Int) raises -> Bool:
        for i in range(self._len):
            if self[i] == val:
                return True
        return False

    @always_inline("nodebug")
    fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
        # if index >= self._len:
        #     raise Error("Index out of bound")
        return self._stride.load[width=width](index)

    @always_inline("nodebug")
    fn store[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[dtype, width]) raises:
        # if index >= self._len:
        #     raise Error("Index out of bound")
        self._stride.store[width=width](index, val)

    @always_inline("nodebug")
    fn load_unsafe[width: Int = 1](self, index: Int) -> Int:
        return self._stride.load[width=width](index).__int__()

    @always_inline("nodebug")
    fn store_unsafe[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[dtype, width]):
        self._stride.store[width=width](index, val)


@value
struct _NDArrayIter[
    dtype: DType,
    forward: Bool = True,
]:
    """Iterator for NDArray.

    Parameters:
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.

    Notes:
        Need to add lifetimes after the new release.
    """

    var index: Int
    var ptr: DTypePointer[dtype]
    var length: Int

    fn __init__(
        inout self, 
        unsafe_pointer: DTypePointer[dtype], 
        length: Int,
    ):
        self.index = 0 if forward else length
        self.ptr = unsafe_pointer
        self.length = length
        
    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) -> SIMD[dtype, 1]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.ptr.load[width=1](current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.ptr.load[width=1](current_index)

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index

# ===----------------------------------------------------------------------===#
# NDArray
# ===----------------------------------------------------------------------===#


struct NDArray[dtype: DType = DType.float32](Stringable, Sized):
    """The N-dimensional array (NDArray).

    The array can be uniquely defined by the following:
        1. The data buffer of all items.
        2. The shape of the array.
        3. The stride in each dimension
        4. The number of dimensions
        5. The datatype of the elements
        6. The order of the array: Row vs Columns major
    """

    var data: DTypePointer[dtype]  # Data buffer of the items in the NDArray
    var ndim: Int
    var ndshape: NDArrayShape  # contains size, shape
    var stride: NDArrayStride  # contains offset, strides
    var coefficient: NDArrayStride  # contains offset, coefficient
    var datatype: DType  # The datatype of memory
    var order: String  # Defines 0 for row major, 1 for column major

    alias simd_width: Int = simdwidthof[dtype]()  # Vector size of the data type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    # default constructor
    @always_inline("nodebug")
    fn __init__(
        inout self, *shape: Int, random: Bool = False, order: String = "C"
    ) raises:
        """
        Example:
            NDArray[DType.int8](3,2,4)
            Returns an zero array with shape 3 x 2 x 4.
        """
        self.ndim = shape.__len__()
        # I cannot name self.ndshape as self.shape as lsp gives unrecognized variable error
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        # I gotta make coefficients empty, but let's just keep it like for now
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.datatype = dtype
        self.order = order
        self.data = DTypePointer[dtype].alloc(self.ndshape._size)
        memset_zero(self.data, self.ndshape._size)
        if random:
            rand[dtype](self.data, self.ndshape._size)

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: List[Int],
        random: Bool = False,
        order: String = "C",
    ) raises:
        """
        Example:
            NDArray[DType.float16](VariadicList[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """
        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape._size)
        memset_zero(self.data, self.ndshape._size)
        self.datatype = dtype
        self.order = order
        if random:
            rand[dtype](self.data, self.ndshape._size)

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: VariadicList[Int],
        random: Bool = False,
        order: String = "C",
    ) raises:
        """
        Example:
            NDArray[DType.float16](VariadicList[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """
        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape._size)
        memset_zero(self.data, self.ndshape._size)
        self.datatype = dtype
        self.order = order
        if random:
            rand[dtype](self.data, self.ndshape._size)

    @always_inline("nodebug")
    fn __init__(
        inout self,
        *shape: Int,
        fill: Scalar[dtype],
        order: String = "C",
    ) raises:
        """
        Example:
            NDArray[DType.float16](VariadicList[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """
        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape._size)
        memset_zero(self.data, self.ndshape._size)
        self.datatype = dtype
        self.order = order
        for i in range(self.ndshape._size):
            self.data[i] = fill

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: List[Int],
        fill: Scalar[dtype],
        order: String = "C",
    ) raises:
        """
        Example:
            NDArray[DType.float16](VariadicList[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """
        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape._size)
        memset_zero(self.data, self.ndshape._size)
        self.datatype = dtype
        self.order = order
        for i in range(self.ndshape._size):
            self.data[i] = fill

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: VariadicList[Int],
        fill: Scalar[dtype],
        order: String = "C",
    ) raises:
        """
        Example:
            NDArray[DType.float16](VariadicList[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """
        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape._size)
        memset_zero(self.data, self.ndshape._size)
        self.datatype = dtype
        self.order = order
        for i in range(self.ndshape._size):
            self.data[i] = fill

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: NDArrayShape,
        random: Bool = False,
        order: String = "C",
    ) raises:
        """
        Example:
            NDArray[DType.float16](VariadicList[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """
        self.ndim = shape._len
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, order=order)
        self.coefficient = NDArrayStride(shape, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape._size)
        memset_zero(self.data, self.ndshape._size)
        self.datatype = dtype
        self.order = order
        if random:
            rand[dtype](self.data, self.ndshape._size)

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: NDArrayShape,
        fill: Scalar[dtype],
        order: String = "C",
    ) raises:
        """
        Example:
            NDArray[DType.float16](VariadicList[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """
        self.ndim = shape._len
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, order=order)
        self.coefficient = NDArrayStride(shape, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape._size)
        memset_zero(self.data, self.ndshape._size)
        self.datatype = dtype
        self.order = order
        for i in range(self.ndshape._size):
            self.data[i] = fill

    fn __init__(
        inout self,
        data: List[SIMD[dtype, 1]],
        shape: List[Int],
        order: String = "C",
    ) raises:
        """
        Example:
            `NDArray[DType.int8](List[Int8](1,2,3,4,5,6), shape=List[Int](2,3))`
            Returns an array with shape 3 x 2 with input values.
        """

        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape._size)
        self.datatype = dtype
        self.order = order
        memset_zero(self.data, self.ndshape._size)
        for i in range(self.ndshape._size):
            self.data[i] = data[i]

    # constructor when rank, ndim, weights, first_index(offset) are known
    fn __init__(
        inout self,
        ndim: Int,
        offset: Int,
        size: Int,
        shape: List[Int],
        strides: List[Int],
        coefficient: List[Int],
        order: String = "C",
    ) raises:
        self.ndim = ndim
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(stride=strides, offset=0)
        self.coefficient = NDArrayStride(stride=coefficient, offset=offset)
        self.datatype = dtype
        self.order = order
        self.data = DTypePointer[dtype].alloc(size)
        memset_zero(self.data, size)

    # for creating views
    fn __init__(
        inout self,
        data: DTypePointer[dtype],
        ndim: Int,
        offset: Int,
        shape: List[Int],
        strides: List[Int],
        coefficient: List[Int],
        order: String = "C",
    ) raises:
        self.ndim = ndim
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(strides, offset=0, order=order)
        self.coefficient = NDArrayStride(
            coefficient, offset=offset, order=order
        )
        self.datatype = dtype
        self.order = order
        self.data = data + self.stride._offset

    @always_inline("nodebug")
    fn __copyinit__(inout self, other: Self):
        self.ndim = other.ndim
        self.ndshape = other.ndshape
        self.stride = other.stride
        self.coefficient = other.coefficient
        self.datatype = other.datatype
        self.order = other.order
        self.data = DTypePointer[dtype].alloc(other.ndshape.size())
        memcpy(self.data, other.data, other.ndshape.size())

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned existing: Self):
        self.ndim = existing.ndim
        self.ndshape = existing.ndshape
        self.stride = existing.stride
        self.coefficient = existing.coefficient
        self.datatype = existing.datatype
        self.order = existing.order
        self.data = existing.data

    @always_inline("nodebug")
    fn __del__(owned self):
        self.data.free()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, val: SIMD[dtype, 1]) raises:
        if index >= self.ndshape._size:
            raise Error("Invalid index: index out of bound")
        if index >= 0:
            self.data.store[width=1](index, val)
        else:
            self.data.store[width=1](index + self.ndshape._size, val)

    @always_inline("nodebug")
    fn __setitem__(inout self, *index: Int, val: SIMD[dtype, 1]) raises:
        if index.__len__() != self.ndim:
            raise Error("Error: Length of Indices do not match the shape")
        for i in range(index.__len__()):
            if index[i] >= self.ndshape[i]:
                raise Error("Error: Elements of `index` exceed the array shape")
        var idx: Int = _get_index(index, self.coefficient)
        self.data.store[width=1](idx, val)

    @always_inline("nodebug")
    fn __setitem__(
        inout self,
        index: List[Int],
        val: SIMD[dtype, 1],
    ) raises:
        if index.__len__() != self.ndim:
            raise Error("Error: Length of Indices do not match the shape")
        for i in range(index.__len__()):
            if index[i] >= self.ndshape[i]:
                raise Error("Error: Elements of `index` exceed the array shape")
        var idx: Int = _get_index(index, self.coefficient)
        self.data.store[width=1](idx, val)

    @always_inline("nodebug")
    fn __setitem__(
        inout self,
        index: VariadicList[Int],
        val: SIMD[dtype, 1],
    ) raises:
        if index.__len__() != self.ndim:
            raise Error("Error: Length of Indices do not match the shape")
        for i in range(index.__len__()):
            if index[i] >= self.ndshape[i]:
                raise Error("Error: Elements of `index` exceed the array shape")
        var idx: Int = _get_index(index, self.coefficient)
        self.data.store[width=1](idx, val)

    fn __getitem__(self, index: Int) raises -> SIMD[dtype, 1]:
        """
        Example:
            `arr[15]` returns the 15th item of the array's data buffer.
        """
        if index >= self.ndshape._size:
            raise Error("Invalid index: index out of bound")
        if index >= 0:
            return self.data.load[width=1](index)
        else:
            return self.data.load[width=1](index + self.ndshape._size)

    fn __getitem__(self, *index: Int) raises -> SIMD[dtype, 1]:
        """
        Example:
            `arr[1,2]` returns the item of 1st row and 2nd column of the array.
        """
        if index.__len__() != self.ndim:
            raise Error("Error: Length of Indices do not match the shape")
        for i in range(index.__len__()):
            if index[i] >= self.ndshape[i]:
                raise Error("Error: Elements of `index` exceed the array shape")
        var idx: Int = _get_index(index, self.coefficient)
        return self.data.load[width=1](idx)

    fn __getitem__(self, index: List[Int]) raises -> SIMD[dtype, 1]:
        """
        Example:
            `arr[1,2]` returns the item of 1st row and 2nd column of the array.
        """
        if index.__len__() != self.ndim:
            raise Error("Error: Length of Indices do not match the shape")

        for i in range(index.__len__()):
            if index[i] >= self.ndshape[i]:
                raise Error("Error: Elements of `index` exceed the array shape")
        var idx: Int = _get_index(index, self.coefficient)
        return self.data.load[width=1](idx)

    fn __getitem__(self, index: VariadicList[Int]) raises -> SIMD[dtype, 1]:
        """
        Example:
            `arr[VariadicList[Int](1,2)]` returns the item of 1st row and
                2nd column of the array.
        """
        if index.__len__() != self.ndim:
            raise Error("Error: Length of Indices do not match the shape")

        for i in range(index.__len__()):
            if index[i] >= self.ndshape[i]:
                raise Error("Error: Elements of `index` exceed the array shape")
        var idx: Int = _get_index(index, self.coefficient)
        return self.data.load[width=1](idx)

    fn _adjust_slice_(self, inout span: Slice, dim: Int):
        if span.start < 0:
            span.start = dim + span.start
        if not span._has_end():
            span.end = dim
        elif span.end < 0:
            span.end = dim + span.end
        if span.end > dim:
            span.end = dim
        if span.end < span.start:
            span.start = 0
            span.end = 0

    fn __getitem__(self, owned *slices: Slice) raises -> Self:
        """
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
                slice_list.append(Slice(0, self.ndshape[i]))

        var narr: Self = self[slice_list]
        return narr

    fn __getitem__(self, owned slices: List[Slice]) raises -> Self:
        """
        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).
        """

        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim or n_slices < self.ndim:
            raise Error("Error: No of slices do not match shape")

        var ndims: Int = 0
        var spec: List[Int] = List[Int]()
        var count: Int = 0
        for i in range(slices.__len__()):
            self._adjust_slice_(slices[i], self.ndshape[i])
            spec.append(slices[i].unsafe_indices())
            if slices[i].unsafe_indices() != 1:
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
                count+=1
                j += 1
            if j >= self.ndim:
                break
            nshape.append(slices[j].unsafe_indices())
            nnum_elements *= slices[j].unsafe_indices()
            ncoefficients.append(self.stride[j] * slices[j].step)
            j += 1

        if count == slices.__len__():
            nshape.append(1)
            nnum_elements = 1
            ncoefficients.append(1)

        var noffset: Int = 0
        if self.order == "C":
            noffset = 0
            if ndims == 1:
                nstrides.append(1)
            for i in range(ndims):
                var temp_stride: Int = 1
                for j in range(i + 1, ndims):  # temp
                    temp_stride *= nshape[j]
                nstrides.append(temp_stride)
            for i in range(slices.__len__()):
                noffset += slices[i].start * self.stride[i]

        elif self.order == "F":
            noffset = 0
            nstrides.append(1)
            for i in range(0, ndims - 1):
                nstrides.append(nstrides[i] * nshape[i])
            for i in range(slices.__len__()):
                noffset += slices[i].start * self.stride[i]

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
        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).
        """
        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim:
            raise Error("Error: No of slices greater than rank of array")
        var slice_list: List[Slice] = List[Slice]()
        for i in range(len(slices)):
            if slices[i].isa[Slice]():
                slice_list.append(slices[i]._get_ptr[Slice]()[0])
            elif slices[i].isa[Int]():
                var int: Int = slices[i]._get_ptr[Int]()[0]
                slice_list.append(Slice(int, int + 1))

        if n_slices < self.ndim:
            for i in range(n_slices, self.ndim):
                print(i)
                var size_at_dim: Int = self.ndshape[i]
                slice_list.append(Slice(0, size_at_dim - 1))
        var narr: Self = self[slice_list]
        return narr

    fn __int__(self) -> Int:
        return self.ndshape._size

    fn __pos__(self) raises -> Self:
        return self * 1.0

    fn __neg__(self) raises -> Self:
        return self * -1.0

    fn __eq__(self, other: Self) -> Bool:
        return self.data == other.data

    fn __add__(inout self, other: SIMD[dtype, 1]) raises -> Self:
        return _af._math_func_one_array_one_SIMD_in_one_array_out[
            dtype, SIMD.__add__
        ](self, other)

    fn __add__(inout self, other: Self) raises -> Self:
        return _af._math_func_2_array_in_one_array_out[dtype, SIMD.__add__](
            self, other
        )

    fn __radd__(inout self, rhs: SIMD[dtype, 1]) raises -> Self:
        return _af._math_func_one_array_one_SIMD_in_one_array_out[
            dtype, SIMD.__add__
        ](self, rhs)

    fn __iadd__(inout self, other: SIMD[dtype, 1]) raises:
        self = _af._math_func_one_array_one_SIMD_in_one_array_out[
            dtype, SIMD.__add__
        ](self, other)

    fn __iadd__(inout self, other: Self) raises:
        self = _af._math_func_2_array_in_one_array_out[dtype, SIMD.__add__](
            self, other
        )

    fn __sub__(self, other: SIMD[dtype, 1]) raises -> Self:
        return _af._math_func_one_array_one_SIMD_in_one_array_out[
            dtype, SIMD.__sub__
        ](self, other)

    fn __sub__(self, other: Self) raises -> Self:
        return _af._math_func_2_array_in_one_array_out[dtype, SIMD.__sub__](
            self, other
        )

    fn __rsub__(self, s: SIMD[dtype, 1]) raises -> Self:
        return -(self - s)

    fn __isub__(inout self, s: SIMD[dtype, 1]) raises:
        self = self - s

    fn __mul__(self, other: SIMD[dtype, 1]) raises -> Self:
        return _af._math_func_one_array_one_SIMD_in_one_array_out[
            dtype, SIMD.__mul__
        ](self, other)

    fn __mul__(self, other: Self) raises -> Self:
        return _af._math_func_2_array_in_one_array_out[dtype, SIMD.__mul__](
            self, other
        )

    fn __rmul__(self, s: SIMD[dtype, 1]) raises -> Self:
        return self * s

    fn __imul__(inout self, s: SIMD[dtype, 1]) raises:
        self = self * s

    fn __imul__(inout self, s: Self) raises:
        self = self * s

    fn __abs__(self) -> Self:
        return abs(self)

    fn __pow__(self, p: Int) -> Self:
        return self._elementwise_pow(p)

    fn __pow__(self, p: Self) raises -> Self:
        if self.ndshape._size != p.ndshape._size:
            raise Error("Both arrays must have same number of elements")

        var result = Self(self.ndshape)
        alias nelts = simdwidthof[dtype]()

        @parameter
        fn vectorized_pow[simd_width: Int](index: Int) -> None:
            result.data.store[width=simd_width](
                index,
                self.data.load[width=simd_width](index)
                ** p.load[width=simd_width](index),
            )

        vectorize[vectorized_pow, nelts](self.ndshape._size)
        return result

    fn __ipow__(inout self, p: Int):
        self = self.__pow__(p)

    fn _elementwise_pow(self, p: Int) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_vec = self

        @parameter
        fn tensor_scalar_vectorize[simd_width: Int](index: Int) -> None:
            new_vec.data.store[width=simd_width](
                index, pow(self.data.load[width=simd_width](index), p)
            )

        vectorize[tensor_scalar_vectorize, simd_width](self.ndshape._size)
        return new_vec

    # ! truediv is multiplying instead of dividing right now lol, I don't know why.
    fn __truediv__(self, other: SIMD[dtype, 1]) raises -> Self:
        return _af._math_func_one_array_one_SIMD_in_one_array_out[
            dtype, SIMD.__truediv__
        ](self, other)

    fn __truediv__(self, other: Self) raises -> Self:
        if self.ndshape._size != other.ndshape._size:
            raise Error("No of elements in both arrays do not match")

        return _af._math_func_2_array_in_one_array_out[dtype, SIMD.__truediv__](
            self, other
        )

    fn __itruediv__(inout self, s: SIMD[dtype, 1]) raises:
        self = self.__truediv__(s)

    fn __itruediv__(inout self, other: Self) raises:
        self = self.__truediv__(other)

    fn __rtruediv__(self, s: SIMD[dtype, 1]) raises -> Self:
        return self.__truediv__(s)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __str__(self) -> String:
        try:
            return (
                "\n"
                + self._array_to_string(0, 0)
                + "\n"
                + self.ndshape.__str__()
                + "  DType: "
                + self.dtype.__str__()
                + "\n"
            )
        except e:
            print("Cannot convert array to string", e)
            return ""

    fn __len__(self) -> Int:
        return self.ndshape._size

    fn __iter__(self) -> _NDArrayIter[dtype]:
        """Iterate over elements of the NDArray, returning copied value.

        Returns:
            An iterator of NDArray elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _NDArrayIter[dtype](
            unsafe_pointer=self.data,
            length=len(self),
        )

    fn __reversed__(self) -> _NDArrayIter[dtype, forward=False]:
        """Iterate backwards over elements of the NDArray, returning 
        copied value.

        Returns:
            A reversed iterator of NDArray elements.
        """

        return _NDArrayIter[dtype, forward=False](
            unsafe_pointer=self.data,
            length=len(self),
        )

    fn _array_to_string(self, dimension: Int, offset: Int) raises -> String:
        if dimension == self.ndim - 1:
            var result: String = str("[\t")
            var number_of_items = self.ndshape[dimension]
            if number_of_items <= 6:  # Print all items
                for i in range(number_of_items):
                    result = (
                        result
                        + self.load[width=1](
                            offset + i * self.stride[dimension]
                        ).__str__()
                    )
                    result = result + "\t"
            else:  # Print first 3 and last 3 items
                for i in range(3):
                    result = (
                        result
                        + self.load[width=1](
                            offset + i * self.stride[dimension]
                        ).__str__()
                    )
                    result = result + "\t"
                result = result + "...\t"
                for i in range(number_of_items - 3, number_of_items):
                    result = (
                        result
                        + self.load[width=1](
                            offset + i * self.stride[dimension]
                        ).__str__()
                    )
                    result = result + "\t"
            result = result + "]"
            return result
        else:
            var result: String = str("[")
            var number_of_items = self.ndshape[dimension]
            if number_of_items <= 6:  # Print all items
                for i in range(number_of_items):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.stride[dimension].__int__(),
                        )
                    if i > 0:
                        result = (
                            result
                            + str(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset + i * self.stride[dimension].__int__(),
                            )
                        )
                    if i < (number_of_items - 1):
                        result = result + "\n"
            else:  # Print first 3 and last 3 items
                for i in range(3):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.stride[dimension].__int__(),
                        )
                    if i > 0:
                        result = (
                            result
                            + str(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset + i * self.stride[dimension].__int__(),
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
                            offset + i * self.stride[dimension].__int__(),
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
        if self.ndshape._size != other.ndshape._size:
            raise Error("The lengths of two vectors do not match.")

        var sum = Scalar[dtype](0)
        for i in range(self.ndshape._size):
            sum = sum + self[i] * other[i]
        return sum

    fn mdot(self, other: Self) raises -> Self:
        """
        Dot product of two matrix.
        Matrix A: M * N.
        Matrix B: N * L.
        """

        if (self.ndim != 2) or (other.ndim != 2):
            raise Error("The array should have only two dimensions (matrix).")

        if self.ndshape[1] != other.ndshape[0]:
            raise Error(
                "Second dimension of A does not match first dimension of B."
            )

        var new_matrix = Self(self.ndshape[0], other.ndshape[1])
        for row in range(self.ndshape[0]):
            for col in range(other.ndshape[1]):
                new_matrix.__setitem__(
                    List[Int](row, col),
                    self[row : row + 1, :].vdot(other[:, col : col + 1]),
                )
        return new_matrix

    fn row(self, id: Int) raises -> Self:
        """Get the ith row of the matrix."""

        if self.ndim > 2:
            raise Error("Only support 2-D array (matrix).")

        var width = self.ndshape[1]
        var buffer = Self(width)
        for i in range(width):
            buffer[i] = self.data.load[width=1](i + id * width)
        return buffer

    fn col(self, id: Int) raises -> Self:
        """Get the ith column of the matrix."""

        if self.ndim > 2:
            raise Error("Only support 2-D array (matrix).")

        var width = self.ndshape[1]
        var height = self.ndshape[0]
        var buffer = Self(height)
        for i in range(height):
            buffer[i] = self.data.load[width=1](id + i * width)
        return buffer

    # # * same as mdot
    fn rdot(self, other: Self) raises -> Self:
        """
        Dot product of two matrix.
        Matrix A: M * N.
        Matrix B: N * L.
        """

        if (self.ndim != 2) or (other.ndim != 2):
            raise Error("The array should have only two dimensions (matrix).")
        if self.ndshape._shape[1] != other.ndshape._shape[0]:
            raise Error(
                "Second dimension of A does not match first dimension of B."
            )

        var new_matrix = Self(self.ndshape[0], other.ndshape[1])
        for row in range(self.ndshape[0]):
            for col in range(other.ndshape[1]):
                new_matrix[col + row * other.ndshape[1]] = self.row(row).vdot(
                    other.col(col)
                )
        return new_matrix

    fn size(self) -> Int:
        return self.ndshape._size

    fn num_elements(self) -> Int:
        return self.ndshape._size

    # # TODO: move this initialization to the Fields and constructor
    fn shape(self) -> NDArrayShape:
        return self.ndshape

    fn load[width: Int = 1](self, index: Int) -> SIMD[dtype, width]:
        """
        Loads a SIMD element of size `width` at the given index `index`.
        """
        return self.data.load[width=width](index)

    # # TODO: we should add checks to make sure user don't load out of bound indices, but that will overhead, figure out later
    fn load[width: Int = 1](self, *index: Int) raises -> SIMD[dtype, width]:
        """
        Loads a SIMD element of size `width` at given variadic indices argument.
        """
        var idx: Int = _get_index(index, self.coefficient)
        return self.data.load[width=width](idx)

    fn store[width: Int](inout self, index: Int, val: SIMD[dtype, width]):
        """
        Stores the SIMD element of size `width` at index `index`.
        """
        self.data.store[width=width](index, val)

    fn store[
        width: Int = 1
    ](inout self, *index: Int, val: SIMD[dtype, width]) raises:
        """
        Stores the SIMD element of size `width` at the given variadic indices argument.
        """
        var idx: Int = _get_index(index, self.coefficient)
        self.data.store[width=width](idx, val)

    # # not urgent: argpartition, byteswap, choose, conj, dump, getfield
    # # partition, put, repeat, searchsorted, setfield, squeeze, swapaxes, take,
    # # tobyets, tofile, view
    # TODO: Implement axis parameter for all

    fn all(self) raises -> Bool:
        # make this a compile time check
        if not (self.dtype == DType.bool or is_inttype(dtype)):
            raise Error("Array elements must be Boolean or Integer.")
        # We might need to figure out how we want to handle truthyness before can do this
        alias nelts: Int = simdwidthof[dtype]()
        var result: Bool = True
        @parameter
        fn vectorized_all[simd_width: Int](idx: Int) -> None:
            result = result and allb(self.data.load[width=simd_width](idx) ) 
        vectorize[vectorized_all, nelts](self.ndshape._size)
        return result 

    fn any(self) raises -> Bool:
        # make this a compile time check
        if not (self.dtype == DType.bool or is_inttype(dtype)):
            raise Error("Array elements must be Boolean or Integer.")
        alias nelts: Int = simdwidthof[dtype]()
        var result: Bool = False 
        @parameter
        fn vectorized_any[simd_width: Int](idx: Int) -> None:
            result = result or anyb(self.data.load[width=simd_width](idx) ) 
        vectorize[vectorized_any, nelts](self.ndshape._size)
        return result

    fn argmax(self) -> Int:
        var result: Int = 0
        var max_val: SIMD[dtype, 1] = self.load[width=1](0)
        for i in range(1, self.ndshape._size):
            var temp: SIMD[dtype, 1] = self.load[width=1](i) 
            if  temp > max_val:
                max_val = temp 
                result = i
        return result

    fn argmin(self) -> Int:
        var result: Int = 0
        var min_val: SIMD[dtype, 1] = self.load[width=1](0)
        for i in range(1, self.ndshape._size):
            var temp: SIMD[dtype, 1] = self.load[width=1](i) 
            if  temp < min_val:
                min_val = temp 
                result = i
        return result

    fn argsort(self):
        pass

    fn astype[type: DType](inout self) raises -> NDArray[type]:
        # I wonder if we can do this operation inplace instead of allocating memory.
        alias nelts = simdwidthof[dtype]()
        var narr: NDArray[type] = NDArray[type](self.ndshape, random=False, order=self.order)
        narr.datatype = type
        @parameter
        fn vectorized_astype[width: Int](idx: Int) -> None:
            narr.store[width](idx, self.load[width](idx).cast[type]())

        vectorize[vectorized_astype, nelts](self.ndshape._size)    
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

    fn fill(inout self, val: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()

        @parameter
        fn vectorized_fill[simd_width: Int](index: Int) -> None:
            self.data.store[width=simd_width](index, val)

        vectorize[vectorized_fill, simd_width](self.ndshape._size)
        return self

    fn flatten(inout self, inplace: Bool = False) raises -> Self:
        # inplace has some problems right now
        # if inplace:
        #     self.ndshape = NDArrayShape(self.ndshape._size, size=self.ndshape._size)
        #     self.stride = NDArrayStride(shape = self.ndshape, offset=0)
        #     return self

        var res: NDArray[dtype] = NDArray[dtype](self.ndshape._size, random=False)
        alias simd_width: Int = simdwidthof[dtype]()
        @parameter
        fn vectorized_flatten[simd_width: Int](index: Int) -> None:
            res.data.store[width=simd_width](
                index, self.data.load[width=simd_width](index)
            )

        vectorize[vectorized_flatten, simd_width](self.ndshape._size)
        return res

    fn item(self, *indices: Int) raises -> SIMD[dtype, 1]:  # I should add
        if indices.__len__() == 1:
            return self.data.load[width=1](indices[0])
        else:
            return self.data.load[width=1](_get_index(indices, self.stride))

    #TODO:  not finished yet
    fn max(self, axis: Int = 0) raises -> Self: 
        var ndim: Int = self.ndim
        var shape: List[Int] = List[Int]()
        for i in range(ndim):
            shape.append(self.ndshape[i])
        if axis > ndim - 1:
            raise Error("axis cannot be greater than the rank of the array")
        var result_shape: List[Int] = List[Int]()
        var axis_size: Int = shape[axis]
        var slices: List[Slice] = List[Slice]()
        for i in range(ndim):
            if i != axis:
                result_shape.append(shape[i])
                slices.append(Slice(0, shape[i]))
            else:
                slices.append(Slice(0, 0))
        print(result_shape.__str__())
        var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(result_shape))

        for i in range(axis_size):
            slices[axis] = Slice(i, i + 1)
            var arr_slice = self[slices]
            result += maxT(arr_slice)

        return result

    fn min(self, axis: Int = 0):
        pass

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

    # fn ravel(self):
    #     pass

    # fn resize(self):
    #     pass

    # fn round(self):
    #     pass

    # fn sort(self):
    #     pass

    fn sum(self: Self, axis: Int) raises -> Self:
        """
        Sum of array elements over a given axis.
        Args:
            axis: The axis along which the sum is performed.
        Returns:
            An NDArray.
        """
        return sum(self, axis)

    # fn stdev(self):
    #     pass

    # fn tolist(self):
    #     pass

    # fn tostring(self):
    #     pass

    # fn trace(self):
    #     pass

    # fn transpose(self):
    #     pass

    # fn variance(self):
    #     pass

    # Technically it only changes the ArrayDescriptor and not the fundamental data
    fn reshape(inout self, *Shape: Int, order: String = "C") raises:
        """
        Reshapes the NDArray to given Shape.

        Args:
            Shape: Variadic list of shape.
            order: Order of the array - Row major `C` or Column major `F`.
        """
        var num_elements_new: Int = 1
        var ndim_new: Int = 0
        for i in Shape:
            num_elements_new *= i
            ndim_new += 1

        if self.ndshape._size != num_elements_new:
            raise Error("Cannot reshape: Number of elements do not match.")

        var shape_new: List[Int] = List[Int]()

        for i in range(ndim_new):
            shape_new.append(Shape[i])
            var temp: Int = 1
            for j in range(i + 1, ndim_new):  # temp
                temp *= Shape[j]

        self.ndim = ndim_new
        self.ndshape = NDArrayShape(shape=shape_new)
        self.stride = NDArrayStride(shape=shape_new, order=order)
        self.order = order

    fn unsafe_ptr(self) -> DTypePointer[dtype, 0]:
        return self.data

    fn to_numpy(self) -> PythonObject:
        return to_numpy(self)
