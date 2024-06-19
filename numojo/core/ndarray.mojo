"""
# ===----------------------------------------------------------------------=== #
# Implements ROW MAJOR N-DIMENSIONAL ARRAYS
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #
"""

"""
# TODO
1) Add NDArray, NDArrayShape constructor overload for List, VariadicList types etc to cover more cases
2) Remove the redundant shape from fields and combine it with NDArrayShape to make it easy to do self.shape() == other.shape()
3) Generalize mdot, rdot to take any IxJx...xKxL and LxMx...xNxP matrix and matmul it into IxJx..xKxMx...xNxP array.
4) Vectorize row(), col() to retrieve rows and columns for 2D arrays
5) Add __getitem__() overload for (Slice, Int)
6) Add support for Column Major
7) Add vectorization for _get_index
8) Write more explanatory Error("") statements
9) Vectorize the for loops inside getitem or move these checks to compile time to try and remove the overhead from constantly checking
"""

from random import rand
from builtin.math import pow
from algorithm import parallelize, vectorize

from .ndarray_utils import *
from .ndarrayview import NDArrayView

# ===----------------------------------------------------------------------===#
# NDArrayShape
# ===----------------------------------------------------------------------===#

# Keep an option for user to change this
alias ALLOWED = 10


@register_passable("trivial")
struct NDArrayShape(Stringable):
    """Implements the NDArrayShape."""

    # Fields
    var _size: Int
    var _shape: StaticIntTuple[ALLOWED]

    @always_inline("nodebug")
    fn __init__(inout self, *shape: Int):
        self._shape = StaticIntTuple[ALLOWED]()
        self._size = 1
        for i in range(min(len(shape), ALLOWED)):
            self._shape[i] = shape[i]
            self._size *= shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, shape: VariadicList[Int]):
        self._shape = StaticIntTuple[ALLOWED]()
        self._size = 1
        for i in range(min(len(shape), ALLOWED)):
            self._shape[i] = shape[i]
            self._size *= shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, shape: List[Int]):
        self._shape = StaticIntTuple[ALLOWED]()
        self._size = 1
        for i in range(min(len(shape), ALLOWED)):
            self._shape[i] = shape[i]
            self._size *= shape[i]

    @always_inline("nodebug")
    fn __init__[length: Int](inout self, shape: StaticIntTuple[length]):
        self._shape = StaticIntTuple[ALLOWED]()
        self._size = 1
        for i in range(min(length, ALLOWED)):
            self._shape[i] = shape[i]
            self._size *= shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, shape: StaticIntTuple[ALLOWED], size: Int):
        self._shape = shape
        self._size = size

    @always_inline("nodebug")
    fn __init__(inout self, owned shape: NDArrayShape):
        self._shape = StaticIntTuple[ALLOWED]()
        self._size = 1
        for i in range(min(len(self._shape), ALLOWED)):
            self._shape[i] = shape[i]
            self._size *= shape[i]

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Int:
        # takes care of negative indexing
        if index >= 0:
            return self._shape[index]
        else:
            return self._shape[len(self._shape) + index]

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, value: Int):
        # takes care of negative indexing
        if index >= 0:
            self._shape[index] = value
        else:
            self._shape[len(self._shape) + index] = value

    @always_inline("nodebug")
    fn size(self) -> Int:
        return self._size

    @always_inline("nodebug")
    fn dim(self) -> Int:
        return len(self._shape)

    @always_inline("nodebug")
    fn __str__(self: Self) -> String:
        var result: String = "Shape: ["
        for i in range(len(self._shape)):
            if self._shape[i] == 0:
                result = result[:-2]
                break
            result += self._shape[i].__str__() + ", "
        return result + "]"
        # return "Shape: " + self._shape.__str__()

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        for i in range(len(self._shape)):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, value: Int) -> Bool:
        for i in range(len(self._shape)):
            if self[i] == value:
                return True
        return False

    # FIGURE OUT HOW TO LOAD SIMD VECTORS OUT OF THIS
    @always_inline("nodebug")
    fn load_unsafe(self, idx: Int) -> Int:
        return self._shape[idx]

    @always_inline("nodebug")
    fn store_unsafe(inout self, idx: Int, value: Int):
        self._shape[idx] = value


@register_passable("trivial")
struct NDArrayStrides(Stringable):
    """Implements the NDArrayStrides."""

    # Fields
    var _offset: Int
    var _stride: StaticIntTuple[ALLOWED]

    @always_inline("nodebug")
    fn __init__(inout self, *shape: Int, ndim: Int, offset: Int = 0):
        self._offset = offset
        self._stride = StaticIntTuple[ALLOWED]()
        for i in range(min(ndim, ALLOWED)):
            var temp: Int = 1
            for j in range(i + 1, ndim):  # temp
                temp *= shape[j]
            self._stride[i] = temp

    @always_inline("nodebug")
    fn __init__(
        inout self, shape: VariadicList[Int], ndim: Int, offset: Int = 0
    ):
        self._offset = offset
        self._stride = StaticIntTuple[ALLOWED]()
        for i in range(min(ndim, ALLOWED)):
            var temp: Int = 1
            for j in range(i + 1, ndim):  # temp
                temp *= shape[j]
            self._stride[i] = temp

    @always_inline("nodebug")
    fn __init__(inout self, shape: List[Int], ndim: Int, offset: Int = 0):
        self._offset = offset
        self._stride = StaticIntTuple[ALLOWED]()
        for i in range(min(ndim, ALLOWED)):
            var temp: Int = 1
            for j in range(i + 1, ndim):  # temp
                temp *= shape[j]
            self._stride[i] = temp

    @always_inline("nodebug")
    fn __init__[
        length: Int
    ](inout self, shape: StaticIntTuple[length], ndim: Int, offset: Int = 0):
        self._offset = offset
        self._stride = StaticIntTuple[ALLOWED]()
        for i in range(min(ndim, ALLOWED)):
            var temp: Int = 1
            for j in range(i + 1, ndim):  # temp
                temp *= shape[j]
            self._stride[i] = temp

    @always_inline("nodebug")
    fn __init__(inout self, stride: StaticIntTuple[ALLOWED], offset: Int = 0):
        self._offset = offset
        self._stride = stride

    @always_inline("nodebug")
    fn __init__(inout self, owned shape: NDArrayShape, ndim: Int, offset: Int = 0):
        self._offset = offset
        self._stride = StaticIntTuple[ALLOWED]()
        
        if ndim == 1: # TODO: make sure this is present in all __init__() to account for 1D arrays. 
            self._stride[0] = 1
        else:
            for i in range(min(ndim, ALLOWED)):
                var temp: Int = 1
                for j in range(
                    i + 1, ndim
                ):  # make sure i don't need to add min() here
                    temp *= shape[j]
                self._stride[i] = temp

    fn __init__(inout self, owned stride: NDArrayStrides, offset: Int = 0):
        self._offset = offset
        self._stride = stride._stride

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Int:
        # takes care of negative indexing
        if index >= 0:
            return self._stride[index]
        else:
            return self._stride[len(self._stride) + index]

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, value: Int):
        # takes care of negative indexing
        if index >= 0:
            self._stride[index] = value
        else:
            self._stride[len(self._stride) + index] = value

    @always_inline("nodebug")
    fn size(self) -> Int:
        var result = 1
        for i in range(len(self._stride)):
            result *= self._stride[i]
        return result

    @always_inline("nodebug")
    fn dim(self) -> Int:
        return len(self._stride)

    @always_inline("nodebug")
    fn __str__(self: Self) -> String:
        var result: String = "Stride: ["
        for i in range(len(self._stride)):
            if self._stride[i] == 0:
                result = result[:-2]
                break
            result += self._stride[i].__str__() + ", "
        return result + "]"
        # return "Stride: " + self._stride.__str__()

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        for i in range(len(self._stride)):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, value: Int) -> Bool:
        for i in range(len(self._stride)):
            if self[i] == value:
                return True
        return False

    # FIGURE OUT HOW TO LOAD SIMD VECTORS OUT OF THIS
    @always_inline("nodebug")
    fn load_unsafe(self, idx: Int) -> Int:
        return self._stride[idx]

    @always_inline("nodebug")
    fn store_unsafe(inout self, idx: Int, value: Int):
        self._stride[idx] = value


# ===----------------------------------------------------------------------===#
# arrayDescriptor
# ===----------------------------------------------------------------------===#


# @register_passable("trivial")
# struct arrayDescriptor[MAX: Int](Stringable):
#     """Implements the arrayDescriptor (dope vector) that stores the metadata of an ND_arrArray.
#     """

#     # Fields
#     var _ndim: Int  # Number of dimensions of the array
#     var _size: Int  # Number of elements in the array
#     var _offset: Int  # Offset of array data in buffer.
#     # when traversing an array.
#     var _coefficients: StaticIntTuple[MAX]  # coefficients
#     var _dtype: DType  # data type of the array
#     var _order: Int  # order of the array 0 - Row Major, 1 - Column Major

#     # ===-------------------------------------------------------------------===#
#     # Life cycle methods
#     # ===-------------------------------------------------------------------===#
#     fn __init__(
#         inout self,
#         ndim: Int,
#         size: Int,
#         offset: Int,
#         shape: StaticIntTuple[MAX],
#         strides: StaticIntTuple[MAX],
#         dtype: DType,
#         order: Int = 0,
#     ):
#         self._ndim = ndim
#         self._size = size
#         self._offset = offset
#         self._shape = shape
#         self._strides = strides
#         self._coefficients = strides
#         self._dtype = dtype
#         self._order = order

#     fn __init__(
#         inout self,
#         ndim: Int,
#         size: Int,
#         offset: Int,
#         coefficients: StaticIntTuple[MAX],
#         dtype: DType,
#         order: Int = 0,
#     ):
#         self._ndim = ndim
#         self._offset = offset
#         self._size = size
#         self._shape = shape
#         self._strides = strides
#         self._coefficients = coefficients
#         self._dtype = dtype
#         self._order = order

#     fn __str__(self: Self) -> String:
#         return "Shape: [" + self._shape.__str__() + "]"

# implement getter and setters for shape, strides, coefficients


# ===----------------------------------------------------------------------===#
# NDArray
# ===----------------------------------------------------------------------===#


# * COLUMN MAJOR INDEXING
struct NDArray[dtype: DType = DType.float32](Stringable):
    """The N-dimensional array (NDArray).

    The array can be uniquely defined by three parameters:
        1. The data buffer of all items.
        2. The shape of the array.
        3. Is the array row-major ('C') or column-major ('F')?
            Currently, we only implement methods using row-major.
    """

    var data: DTypePointer[dtype]  # Data buffer of the items in the NDArray
    var ndim: Int
    var ndshape: NDArrayShape  # contains size, shape
    var stride: NDArrayStrides  # contains offset, strides
    var coefficients: NDArrayStrides  # contains offset, coefficients
    var datatype: DType  # The datatype of memory
    var order: Int  # Defines 0 for row major, 1 for column major

    alias simd_width: Int = simdwidthof[dtype]()  # Vector size of the data type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    # default constructor
    @always_inline("nodebug")
    fn __init__(inout self, *shape: Int, random: Bool = False):
        """
        Example:
            NDArray[DType.int8](3,2,4)
            Returns an zero array with shape 3 x 2 x 4.
        """
        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(
            shape
        )  # for some reason lsp shows error for using self.shape name, so keep it as ndshape for now
        self.stride = NDArrayStrides(shape, self.ndim)
        self.coefficients = NDArrayStrides(
            shape, self.ndim
        )  # I gotta make it empty, but let's just keep it like for tnow
        self.datatype = dtype
        self.order = 0
        self.data = DTypePointer[dtype].alloc(self.ndshape._size)
        memset_zero(self.data, self.ndshape._size)

        if random:
            rand[dtype](self.data, self.ndshape._size)

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: VariadicList[Int],
        random: Bool = False,
        value: SIMD[dtype, 1] = SIMD[dtype, 1](0),
    ):
        """
        Example:
            NDArray[DType.float16](VariadicList[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """
        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(
            shape
        )  # for some reason lsp shows error for using self.shape name, so keep it as ndshape for now
        self.stride = NDArrayStrides(shape, self.ndim)
        self.coefficients = NDArrayStrides(
            shape, self.ndim
        )  # I gotta make it empty, but let's just keep it like for tnow
        self.data = DTypePointer[dtype].alloc(self.ndshape._size)
        memset_zero(self.data, self.ndshape._size)
        self.datatype = dtype
        self.order = 0

        if random:
            rand[dtype](self.data, self.ndshape._size)

        else:
            for i in range(self.ndshape._size):
                self.data[i] = value.cast[dtype]()

    # fn __init__(inout self, data: List[SIMD[dtype, 1]], shape: List[Int]):
    #     """
    #     Example:
    #         `NDArray[DType.int8](List[Int8](1,2,3,4,5,6), shape=List[Int](2,3))`
    #         Returns an array with shape 3 x 2 with input values.
    #     """

    #     var dimension: Int = shape.__len__()
    #     var first_index: Int = 0
    #     var size: Int = 1
    #     var shapeInfo: List[Int] = List[Int]()
    #     var strides: List[Int] = List[Int]()

    #     for i in range(dimension):
    #         shapeInfo.append(shape[i])
    #         size *= shape[i]
    #         var temp: Int = 1
    #         for j in range(i + 1, dimension):  # temp
    #             temp *= shape[j]
    #         strides.append(temp)

    #     self.data = DTypePointer[dtype].alloc(size)
    #     memset_zero(self.data, size)
    #     for i in range(size):
    #         self.data[i] = data[i]
    #     self.info = arrayDescriptor[dtype](
    #         dimension, first_index, size, shapeInfo, strides
    #     )

    # fn __init__(inout self, shape: List[Int], random: Bool = False):
    #     """
    #     Example:
    #         NDArray[DType.float16](List[Int](3, 2, 4), random=True)
    #         Returns an array with shape 3 x 2 x 4 and randomly values.
    #     """

    #     var dimension: Int = shape.__len__()
    #     var first_index: Int = 0
    #     var size: Int = 1
    #     var shapeInfo: List[Int] = List[Int]()
    #     var strides: List[Int] = List[Int]()

    #     for i in range(dimension):
    #         shapeInfo.append(shape[i])
    #         size *= shape[i]
    #         var temp: Int = 1
    #         for j in range(i + 1, dimension):  # temp
    #             temp *= shape[j]
    #         strides.append(temp)

    #     self.data = DTypePointer[dtype].alloc(size)
    #     memset_zero(self.data, size)
    #     self.info = arrayDescriptor[dtype](
    #         dimension, first_index, size, shapeInfo, strides
    #     )
    #     if random:
    #         rand[dtype](self.data, size)

    fn __init__(
        inout self,
        shape: NDArrayShape,
        random: Bool = False,
        value: SIMD[dtype, 1] = SIMD[dtype, 1](0),  # make it Optional[]
    ):
        self.ndim = 0
        for i in range(len(shape._shape)):
            if shape[i] != 0:
                self.ndim += 1

        self.ndshape = shape
        # self.ndshape = NDArrayShape(
            # shape
        # )  # for some reason lsp shows error for using self.shape name, so keep it as ndshape for now
        self.stride = NDArrayStrides(shape, self.ndim)
        self.coefficients = self.stride
        # self.coefficients = NDArrayStrides(
            # shape, self.ndim
        # )  # I gotta make it empty, but let's just keep it like for tnow
        self.data = DTypePointer[dtype].alloc(self.ndshape._size)
        memset_zero(self.data, self.ndshape._size)
        self.datatype = dtype
        self.order = 0

        if random:
            rand[dtype](self.data, self.ndshape._size)

        else:
            for i in range(self.ndshape._size):
                self.data[i] = value.cast[dtype]()

    # constructor when rank, ndim, weights, first_index(offset) are known
    fn __init__(
        inout self,
        ndim: Int,
        offset: Int,
        size: Int,
        shape: StaticIntTuple[ALLOWED],
        strides: StaticIntTuple[ALLOWED],
        coefficients: StaticIntTuple[ALLOWED],
    ):
        self.ndim = ndim
        self.ndshape = NDArrayShape(
            shape, size
        )  # for some reason lsp shows error for using self.shape name, so keep it as ndshape for now
        self.stride = NDArrayStrides(strides, offset=offset)
        self.coefficients = NDArrayStrides(
            coefficients, offset=offset
        )  # I gotta make it empty, but let's just keep it like for tnow
        self.datatype = dtype
        self.order = 0
        self.data = DTypePointer[dtype].alloc(size)
        memset_zero(self.data, size)

    # for creating views
    fn __init__(
        inout self,
        data: DTypePointer[dtype],
        ndim: Int,
        offset: Int,
        size: Int,
        shape: StaticIntTuple[ALLOWED],
        strides: StaticIntTuple[ALLOWED],
        coefficients: StaticIntTuple[ALLOWED],
    ):
        self.ndim = ndim
        self.ndshape = NDArrayShape(
            shape, size
        )  # for some reason lsp shows error for using self.shape name, so keep it as ndshape for now
        self.stride = NDArrayStrides(strides, offset=offset)
        self.coefficients = NDArrayStrides(
            coefficients, offset=offset
        )  # I gotta make it empty, but let's just keep it like for tnow
        self.datatype = dtype
        self.order = 0
        self.data = data + self.stride._offset

    @always_inline("nodebug")
    fn __copyinit__(inout self, other: Self):
        self.ndim = other.ndim
        self.ndshape = other.ndshape
        self.stride = other.stride
        self.coefficients = other.coefficients
        self.datatype = other.datatype
        self.order = other.order
        self.data = DTypePointer[dtype].alloc(other.ndshape.size())
        memcpy(self.data, other.data, other.ndshape.size())

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned existing: Self):
        self.ndim = existing.ndim
        self.ndshape = existing.ndshape
        self.stride = existing.stride
        self.coefficients = existing.coefficients
        self.datatype = existing.datatype
        self.order = existing.order
        self.data = existing.data

    # @always_inline("nodebug")
    # fn __del__(owned self):
    #     self.data.free()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, value: SIMD[dtype, 1]) raises:
        if index >= self.ndshape._size:
            raise Error("Invalid index: index out of bound")
        self.data[index] = value

    # lsp shows unused variable because of the if condition, be careful with this
    @always_inline("nodebug")
    fn __setitem__(inout self, *index: Int, value: SIMD[dtype, 1]) raises:
        var idx: Int = _get_index(index, self.coefficients._stride)
        if idx >= self.ndshape._size:
            raise Error("Invalid index: index out of bound")

        self.data[idx] = value

    @always_inline("nodebug")
    fn __setitem__(
        inout self,
        indices: List[Int],
        val: SIMD[dtype, 1],
    ) raises:
        var idx: Int = _get_index(indices, self.coefficients._stride)
        if idx >= self.ndshape._size:
            raise Error("Invalid index: index out of bound")

        self.data[idx] = val

    fn __getitem__(self, index: Int) raises -> SIMD[dtype, 1]:
        """
        Example:
            `arr[15]` returns the 15th item of the array's data buffer.
        """
        if index >= self.ndshape._size:
            raise Error("Invalid index: index out of bound")

        return self.data[index]

    fn __getitem__(self, *indices: Int) raises -> SIMD[dtype, 1]:
        """
        Example:
            `arr[1,2]` returns the item of 1st row and 2nd column of the array.
        """
        if indices.__len__() != self.ndshape._size:
            raise Error("Error: Length of Indices do not match the shape")

        for i in range(indices.__len__()):
            if indices[i] >= self.ndshape._shape[i]:
                raise Error(
                    "Error: Elements of Indices exceed the shape values"
                )

        var idx: Int = _get_index(indices, self.coefficients._stride)
        return self.data[idx]

    # same as above, but explicit VariadicList
    fn __getitem__(self, indices: VariadicList[Int]) raises -> SIMD[dtype, 1]:
        """
        Example:
            `arr[VariadicList[Int](1,2)]` returns the item of 1st row and
                2nd column of the array.
        """
        if indices.__len__() != self.ndim:
            raise Error("Error: Length of Indices do not match the shape")

        for i in range(indices.__len__()):
            if indices[i] >= self.ndshape._shape[i]:
                raise Error(
                    "Error: Elements of Indices exceed the shape values"
                )

        var index: Int = _get_index(indices, self.coefficients._stride)
        return self.data[index]

    fn __getitem__(
        self,
        indices: List[Int],
        offset: Int,
        coefficients: StaticIntTuple[ALLOWED],
    ) -> SIMD[dtype, 1]:
        """
        Example:
            `arr[List[Int](1,2), 1, List[Int](1,1)]` returns the item of
            1st row and 3rd column of the array.
        """
        var index: Int = offset + _get_index(indices, coefficients)
        return self.data.__getitem__(index)

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
        if n_slices > self.ndim or n_slices < self.ndim:
            print("Error: No of slices do not match shape")

        var ndims: Int = 0
        var spec: StaticIntTuple[ALLOWED] = StaticIntTuple[ALLOWED]()
        for i in range(slices.__len__()):
            self._adjust_slice_(slices[i], self.ndshape._shape[i])
            spec[i] = slices[i].unsafe_indices()
            if slices[i].unsafe_indices() != 1:
                ndims += 1

        var nshape: StaticIntTuple[ALLOWED] = StaticIntTuple[ALLOWED]()
        var ncoefficients: StaticIntTuple[ALLOWED] = StaticIntTuple[ALLOWED]()
        var nstrides: StaticIntTuple[ALLOWED] = StaticIntTuple[ALLOWED]()
        var nnum_elements: Int = 1

        var j: Int = 0
        for i in range(ndims):
            while spec[j] == 1:
                j += 1
            if j >= self.ndim:
                break
            nshape[i] = slices[j].unsafe_indices()
            nnum_elements *= slices[j].unsafe_indices()
            ncoefficients[i] = self.coefficients._stride[j] * slices[j].step
            j += 1

        for k in range(ndims):
            var temp: Int = 1
            for j in range(k + 1, ndims):  # temp
                temp *= nshape[j]
            nstrides[k] = temp

        # row major
        var noffset: Int = 0
        for i in range(slices.__len__()):
            var temp: Int = 1
            for j in range(i + 1, slices.__len__()):
                temp *= self.ndshape._shape[j]
            noffset += slices[i].start * temp

        # var narr = Self(
        #     self.data,
        #     ndims,
        #     noffset,
        #     nnum_elements,
        #     nshape,
        #     nstrides,
        #     ncoefficients,
        # )

        var narr = Self(
            ndims, noffset, nnum_elements, nshape, nstrides, ncoefficients
        )

        # Starting index to traverse the new array
        var index = StaticIntTuple[ALLOWED]()
        for i in range(ndims):
            index[i] = 0

        _traverse_iterative[dtype](
            self, narr, ndims, nshape, ncoefficients, nstrides, noffset, index, 0
        )

        return narr

    fn __getitem__(self, owned slices: List[Slice]) raises -> Self:
        """
        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).
        """
        var n_slices: Int = slices.__len__()
        if n_slices > self.ndim or n_slices < self.ndim:
            print("Error: No of slices do not match shape")

        var ndims: Int = 0
        var spec: StaticIntTuple[ALLOWED] = StaticIntTuple[ALLOWED]()
        for i in range(slices.__len__()):
            self._adjust_slice_(slices[i], self.ndshape._shape[i])
            spec[i] = slices[i].unsafe_indices()
            if slices[i].unsafe_indices() != 1:
                ndims += 1

        var nshape: StaticIntTuple[ALLOWED] = StaticIntTuple[ALLOWED]()
        var ncoefficients: StaticIntTuple[ALLOWED] = StaticIntTuple[ALLOWED]()
        var nstrides: StaticIntTuple[ALLOWED] = StaticIntTuple[ALLOWED]()
        var nnum_elements: Int = 1

        var j: Int = 0
        for i in range(ndims):
            while spec[j] == 1:
                j += 1
            if j >= self.ndim:
                break
            nshape[i] = slices[j].unsafe_indices()
            nnum_elements *= slices[j].unsafe_indices()
            ncoefficients[i] = self.coefficients._stride[j] * slices[j].step
            j += 1

        for k in range(ndims):
            var temp: Int = 1
            for j in range(k + 1, ndims):  # temp
                temp *= nshape[j]
            nstrides[k] = temp

        # row major
        var noffset: Int = 0
        for i in range(slices.__len__()):
            var temp: Int = 1
            for j in range(i + 1, slices.__len__()):
                temp *= self.ndshape._shape[j]
            noffset += slices[i].start * temp

        var narr = Self(
            ndims, noffset, nnum_elements, nshape, nstrides, ncoefficients
        )

        var index = StaticIntTuple[ALLOWED]()
        for i in range(ndims):
            index[i] = 0

        _traverse_iterative[dtype](
            self, narr, ndims, nshape, ncoefficients, nstrides, noffset, index, 0
        )

        return narr

    fn __getitem__(self, owned *slices: Variant[Slice, Int]) raises -> Self:
        """
        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).
        """
        var slice_list: List[Slice] = List[Slice]()
        for i in range(len(slices)):
            if slices[i].isa[Slice]():
                slice_list.append(slices[i]._get_ptr[Slice]()[0])
            elif slices[i].isa[Int]():
                var int: Int = slices[i]._get_ptr[Int]()[0]
                slice_list.append(Slice(int, int+1))
        var narr: Self = self[slice_list]
        return narr

    fn __int__(self) -> Int:
        return self.ndshape._size

    fn __pos__(self) -> Self:
        return self * 1.0

    fn __neg__(self) -> Self:
        return self * -1.0

    fn __eq__(self, other: Self) -> Bool:
        return self.data == other.data

    fn _elementwise_scalar_arithmetic[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width], SIMD[dtype, width]
        ) -> SIMD[dtype, width]
    ](self, s: SIMD[dtype, 1]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = self

        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array.data.store[width=simd_width](
                idx,
                func[dtype, simd_width](
                    SIMD[dtype, simd_width](s),
                    self.data.load[width=simd_width](idx),
                ),
            )

        vectorize[elemwise_vectorize, simd_width](self.ndshape._size)
        return new_array

    fn _elementwise_array_arithmetic[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width], SIMD[dtype, width]
        ) -> SIMD[dtype, width]
    ](self, other: Self) -> Self:
        alias simd_width = simdwidthof[dtype]()
        var new_vec = self

        @parameter
        fn elemwise_arithmetic[simd_width: Int](index: Int) -> None:
            new_vec.data.store[width=simd_width](
                index,
                func[dtype, simd_width](
                    self.data.load[width=simd_width](index),
                    other.data.load[width=simd_width](index),
                ),
            )

        vectorize[elemwise_arithmetic, simd_width](self.ndshape._size)
        return new_vec

    fn __add__(inout self, other: SIMD[dtype, 1]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__add__](other)

    fn __add__(inout self, other: Self) -> Self:
        return self._elementwise_array_arithmetic[SIMD.__add__](other)

    fn __radd__(inout self, s: SIMD[dtype, 1]) -> Self:
        return self + s

    fn __iadd__(inout self, s: SIMD[dtype, 1]):
        self = self + s

    fn __sub__(self, other: SIMD[dtype, 1]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__sub__](other)

    fn __sub__(self, other: Self) -> Self:
        return self._elementwise_array_arithmetic[SIMD.__sub__](other)

    fn __rsub__(self, s: SIMD[dtype, 1]) -> Self:
        return -(self - s)

    fn __isub__(inout self, s: SIMD[dtype, 1]):
        self = self - s

    fn __mul__(self, s: SIMD[dtype, 1]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__mul__](s)

    fn __mul__(self, other: Self) -> Self:
        return self._elementwise_array_arithmetic[SIMD.__mul__](other)

    fn __rmul__(self, s: SIMD[dtype, 1]) -> Self:
        return self * s

    fn __imul__(inout self, s: SIMD[dtype, 1]):
        self = self * s

    fn reduce_sum(self) -> SIMD[dtype, 1]:
        var reduced = SIMD[dtype, 1](0.0)
        alias simd_width: Int = simdwidthof[dtype]()

        @parameter
        fn vectorize_reduce[simd_width: Int](idx: Int) -> None:
            reduced += self.data.load[width=simd_width](idx).reduce_add()

        vectorize[vectorize_reduce, simd_width](self.ndshape._size)
        return reduced

    fn reduce_mul(self) -> SIMD[dtype, 1]:
        var reduced = SIMD[dtype, 1](0.0)
        alias simd_width: Int = simdwidthof[dtype]()

        @parameter
        fn vectorize_reduce[simd_width: Int](idx: Int) -> None:
            reduced *= self.data.load[width=simd_width](idx).reduce_mul()

        vectorize[vectorize_reduce, simd_width](self.ndshape._size)
        return reduced

    # fn __abs__(self) -> Self:
    #     var result = Self(self.ndshape._shape)
    #     alias nelts = simdwidthof[dtype]()

    #     @parameter
    #     fn vectorized_abs[simd_width: Int](idx: Int) -> None:
    #         result.data.store[width=simd_width](
    #             idx, abs(self.data.load[width=simd_width](idx))
    #         )

    #     vectorize[vectorized_abs, nelts](self.info._size)
    #     return result

    # all elements raised to some integer power
    fn __pow__(self, p: Int) -> Self:
        return self._elementwise_pow(p)

    # fn __pow__(self, p: Self) raises -> Self:
    #     if (
    #         self.ndshape._size != p.ndshape._size
    #     ):  # This lets us do the operation as long as they have same no of elements
    #         raise Error("Both arrays must have same number of elements")

    #     var result = Self(self.ndshape._shape)
    #     alias nelts = simdwidthof[dtype]()

    #     @parameter
    #     fn vectorized_pow[simd_width: Int](idx: Int) -> None:
    #         result.data.store[width=simd_width](
    #             idx,
    #             self.data.load[width=simd_width](idx)
    #             ** p.load[width=simd_width](idx),
    #         )

    #     return result

    fn __ipow__(inout self, p: Int):
        self = self.__pow__(p)

    fn _elementwise_pow(self, p: Int) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_vec = self

        @parameter
        fn tensor_scalar_vectorize[simd_width: Int](idx: Int) -> None:
            new_vec.data.store[width=simd_width](
                idx, pow(self.data.load[width=simd_width](idx), p)
            )

        vectorize[tensor_scalar_vectorize, simd_width](self.ndshape._size)
        return new_vec

    # ! truediv is multiplying instead of dividing right now lol, I don't know why.
    fn __truediv__(self, s: SIMD[dtype, 1]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__truediv__](s)

    fn __truediv__(self, other: Self) raises -> Self:
        if self.ndshape._size != other.ndshape._size:
            raise Error("No of elements in both arrays do not match")

        return self._elementwise_array_arithmetic[SIMD.__truediv__](other)

    fn __itruediv__(inout self, s: SIMD[dtype, 1]):
        self = self.__truediv__(s)

    fn __itruediv__(inout self, other: Self) raises:
        self = self.__truediv__(other)

    fn __rtruediv__(self, s: SIMD[dtype, 1]) -> Self:
        return self.__truediv__(s)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __str__(self) -> String:
        return (
            "\n"
            + self._array_to_string(0, 0)
            + "\n"
            + self.ndshape.__str__()
            + "  DType: "
            + self.dtype.__str__()
            + "\n"
        )

    fn __len__(inout self) -> Int:
        return self.ndshape._size

    fn _array_to_string(self, dimension: Int, offset: Int) -> String:
        if dimension == self.ndim - 1:
            var result: String = str("[\t")
            var number_of_items = self.ndshape._shape[dimension]
            if number_of_items <= 6:  # Print all items
                for i in range(number_of_items):
                    result = (
                        result
                        + self.data.load[width=1](
                            offset + i * self.coefficients._stride[dimension]
                        ).__str__()
                    )
                    result = result + "\t"
            else:  # Print first 3 and last 3 items
                for i in range(3):
                    result = (
                        result
                        + self.data[
                            offset + i * self.coefficients._stride[dimension]
                        ].__str__()
                    )
                    result = result + "\t"
                result = result + "...\t"
                for i in range(number_of_items - 3, number_of_items):
                    result = (
                        result
                        + self.data[
                            offset + i * self.coefficients._stride[dimension]
                        ].__str__()
                    )
                    result = result + "\t"
            result = result + "]"
            return result
        else:
            var result: String = str("[")
            var number_of_items = self.ndshape._shape[dimension]
            if number_of_items <= 6:  # Print all items
                for i in range(number_of_items):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.coefficients._stride[dimension],
                        )
                    if i > 0:
                        result = (
                            result
                            + str(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset
                                + i * self.coefficients._stride[dimension],
                            )
                        )
                    if i < (number_of_items - 1):
                        result = result + "\n"
            else:  # Print first 3 and last 3 items
                for i in range(3):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.coefficients._stride[dimension],
                        )
                    if i > 0:
                        result = (
                            result
                            + str(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset
                                + i * self.coefficients._stride[dimension],
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
                            offset + i * self.coefficients._stride[dimension],
                        )
                    )
                    if i < (number_of_items - 1):
                        result = result + "\n"
            result = result + "]"
            return result

    # fn _array_to_string(self, dimension: Int, offset: Int) -> String:
    #     if dimension == self.ndim - 1:
    #         var result: String = str("[\t")
    #         var number_of_items = self.ndshape._shape[dimension]
    #         if number_of_items <= 6:  # Print all items
    #             for i in range(number_of_items):
    #                 print("strides: ", offset + i * self.coefficients._stride[dimension])
    #                 result = (
    #                     result
    #                     + self.data.load[width=1](
    #                         offset + i * self.coefficients._stride[dimension]
    #                     ).__str__()
    #                 )
    #                 result = result + "\t"
    #         else:  # Print first 3 and last 3 items
    #             for i in range(3):
    #                 result = (
    #                     result
    #                     + self.data[
    #                         offset + i * self.coefficients._stride[dimension]
    #                     ].__str__()
    #                 )
    #                 result = result + "\t"
    #             result = result + "...\t"
    #             for i in range(number_of_items - 3, number_of_items):
    #                 result = (
    #                     result
    #                     + self.data[
    #                         offset + i * self.coefficients._stride[dimension]
    #                     ].__str__()
    #                 )
    #                 result = result + "\t"
    #         result = result + "]"
    #         return result
    #     else:
    #         var result: String = str("[")
    #         var number_of_items = self.ndshape._shape[dimension]
    #         if number_of_items <= 6:  # Print all items
    #             for i in range(number_of_items):
    #                 if i == 0:
    #                     result = result + self._array_to_string(
    #                         dimension + 1,
    #                         offset + i * self.coefficients._stride[dimension],
    #                     )
    #                 if i > 0:
    #                     result = (
    #                         result
    #                         + str(" ") * (dimension + 1)
    #                         + self._array_to_string(
    #                             dimension + 1,
    #                             offset + i * self.coefficients._stride[dimension],
    #                         )
    #                     )
    #                 if i < (number_of_items - 1):
    #                     result = result + "\n"
    #         else:  # Print first 3 and last 3 items
    #             for i in range(3):
    #                 if i == 0:
    #                     result = result + self._array_to_string(
    #                         dimension + 1,
    #                         offset + i * self.coefficients._stride[dimension],
    #                     )
    #                 if i > 0:
    #                     result = (
    #                         result
    #                         + str(" ") * (dimension + 1)
    #                         + self._array_to_string(
    #                             dimension + 1,
    #                             offset + i * self.coefficients._stride[dimension],
    #                         )
    #                     )
    #                 if i < (number_of_items - 1):
    #                     result += "\n"
    #             result = result + "...\n"
    #             for i in range(number_of_items - 3, number_of_items):
    #                 result = (
    #                     result
    #                     + str(" ") * (dimension + 1)
    #                     + self._array_to_string(
    #                         dimension + 1,
    #                         offset + i * self.coefficients._stride[dimension],
    #                     )
    #                 )
    #                 if i < (number_of_items - 1):
    #                     result = result + "\n"
    #         result = result + "]"
    #         return result

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    # fn vdot(self, other: Self) raises -> SIMD[dtype, 1]:
    #     """
    #     Inner product of two vectors.
    #     """
    #     if self.info._size != other.info._size:
    #         raise Error("The lengths of two vectors do not match.")

    #     var sum = Scalar[dtype](0)
    #     for i in range(self.info._size):
    #         sum = sum + self[i] * other[i]
    #     return sum

    # fn mdot(self, other: Self) raises -> Self:
    #     """
    #     Dot product of two matrix.
    #     Matrix A: M * N.
    #     Matrix B: N * L.
    #     """

    #     if (self.info._ndim != 2) or (other.info._ndim != 2):
    #         raise Error("The array should have only two dimensions (matrix).")

    #     if self.info._shape[1] != other.info._shape[0]:
    #         raise Error(
    #             "Second dimension of A does not match first dimension of B."
    #         )

    #     var new_dims = List[Int](self.info._shape[0], other.info._shape[1])
    #     var new_matrix = Self(new_dims)
    #     for row in range(self.info._shape[0]):
    #         for col in range(other.info._shape[1]):
    #             new_matrix.__setitem__(
    #                 List[Int](row, col),
    #                 self[row : row + 1, :].vdot(other[:, col : col + 1]),
    #             )
    #     return new_matrix

    # fn row(self, id: Int) raises -> Self:
    #     """Get the ith row of the matrix."""

    #     if self.info._ndim > 2:
    #         raise Error("Only support 2-D array (matrix).")

    #     # If bug fixed, then
    #     # Tensor[dtype](shape=width, self.data.offset(something))

    #     var width = self.info._shape[1]
    #     var buffer = Self(width)
    #     for i in range(width):
    #         buffer[i] = self.data[i + id * width]
    #     return buffer

    # fn col(self, id: Int) raises -> Self:
    #     """Get the ith column of the matrix."""

    #     if self.info._ndim > 2:
    #         raise Error("Only support 2-D array (matrix).")

    #     var width = self.info._shape[1]
    #     var height = self.info._shape[0]
    #     var buffer = Self(height)
    #     for i in range(height):
    #         buffer[i] = self.data[id + i * width]
    #     return buffer

    # # * same as mdot
    # fn rdot(self, other: Self) raises -> Self:
    #     """
    #     Dot product of two matrix.
    #     Matrix A: M * N.
    #     Matrix B: N * L.
    #     """

    #     if (self.info._ndim != 2) or (other.info._ndim != 2):
    #         raise Error("The array should have only two dimensions (matrix).")
    #     if self.info._shape[1] != other.info._shape[0]:
    #         raise Error(
    #             "Second dimension of A does not match first dimension of B."
    #         )

    #     var new_dims = List[Int](self.info._shape[0], other.info._shape[1])
    #     var new_matrix = Self(new_dims)
    #     for row in range(new_dims[0]):
    #         for col in range(new_dims[1]):
    #             new_matrix[col + row * new_dims[1]] = self.row(row).vdot(
    #                 other.col(col)
    #             )
    #     return new_matrix

    # fn size(self) -> Int:
    #     return self.info._size

    fn num_elements(self) -> Int:
        return self.ndshape._size

    # # TODO: move this initialization to the Fields and constructor
    fn shape(self) -> NDArrayShape:
        return self.ndshape

    fn load[width: Int](self, idx: Int) -> SIMD[dtype, width]:
        """
        Loads a SIMD element of size `width` at the given index `idx`.
        """
        return self.data.load[width=width](idx)

    # # TODO: we should add checks to make sure user don't load out of bound indices, but that will overhead, figure out later
    fn load[width: Int = 1](self, *indices: Int) -> SIMD[dtype, width]:
        """
        Loads a SIMD element of size `width` at given variadic indices argument.
        """
        var index: Int = _get_index(indices, self.coefficients._stride)
        return self.data.load[width=width](index)

    # fn load[width: Int = 1](self, indices: VariadicList[Int]) -> SIMD[dtype, 1]:
    #     """
    #     Loads a SIMD element of size `width` at given VariadicList of indices.
    #     """
    #     var idx: Int = _get_index(indices, self.info._strides)
    #     return self.data.load[width=width](idx)

    # fn load[width: Int = 1](self, indices: List[Int]) -> SIMD[dtype, 1]:
    #     """
    #     Loads a SIMD element of size `width` at given List of indices.
    #     """
    #     var idx: Int = _get_index(indices, self.info._strides)
    #     return self.data.load[width=width](idx)

    fn store[width: Int](inout self, idx: Int, val: SIMD[dtype, width]):
        """
        Stores the SIMD element of size `width` at index `idx`.
        """
        self.data.store[width=width](idx, val)

    fn store[
        width: Int = 1
    ](inout self, *indices: Int, val: SIMD[dtype, width]):
        """
        Stores the SIMD element of size `width` at the given variadic indices argument.
        """
        var idx: Int = _get_index(indices, self.coefficients._stride)
        self.data.store[width=width](idx, val)

    # fn store[
    #     width: Int = 1
    # ](inout self, indices: VariadicList[Int], val: SIMD[dtype, width]):
    #     """
    #     Stores the SIMD element of size `width` at the given Variadic list of indices.
    #     """
    #     var idx: Int = _get_index(indices, self.info._strides)
    #     self.data.store[width=width](idx, val)

    # # not urgent: argpartition, byteswap, choose, conj, dump, getfield
    # # partition, put, repeat, searchsorted, setfield, squeeze, swapaxes, take,
    # # tobyets, tofile, view

    # fn all(self):
    #     pass

    # fn any(self):
    #     pass

    # fn argmax(self):
    #     pass

    # fn argmin(self):
    #     pass

    # fn argsort(self):
    #     pass

    # fn astype(self):
    #     pass

    # fn clip(self):
    #     pass

    # fn compress(self):
    #     pass

    # fn copy(self):
    #     pass

    # fn cumprod(self):
    #     pass

    # fn cumsum(self):
    #     pass

    # fn diagonal(self):
    #     pass

    # fn fill(self):
    #     pass

    # fn flatten(self):
    #     pass

    # fn item(self):
    #     pass

    # fn max(self):
    #     pass

    # fn min(self):
    #     pass

    # fn mean(self):
    #     pass

    # fn nonzero(self):
    #     pass

    # fn prod(self):
    #     pass

    # fn ravel(self):
    #     pass

    # fn resize(self):
    #     pass

    # fn round(self):
    #     pass

    # fn sort(self):
    #     pass

    # fn sum(self):
    #     pass

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
    fn reshape(inout self, *Shape: Int) raises:
        """
        Reshapes the NDArray to given Shape.

        Args:
            Shape: Variadic list of shape.
        """
        var num_elements_new: Int = 1
        var ndim_new: Int = 0
        for i in Shape:
            num_elements_new *= i
            ndim_new += 1

        if self.ndshape._size != num_elements_new:
            raise Error("Cannot reshape: Number of elements do not match.")

        self.ndim = ndim_new
        var shape_new: StaticIntTuple[ALLOWED] = StaticIntTuple[ALLOWED]()
        var strides_new: StaticIntTuple[ALLOWED] = StaticIntTuple[ALLOWED]()

        for i in range(ndim_new):
            shape_new[i] = Shape[i]
            var temp: Int = 1
            for j in range(i + 1, ndim_new):  # temp
                temp *= Shape[j]
            strides_new[i] = temp

        self.ndshape._shape = shape_new
        self.stride._stride = strides_new
    # self.shape.shape = shape_new # current ndarray doesn't have NDArray shape field

    fn unsafe_ptr(self) -> DTypePointer[dtype, 0]:
        return self.data


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
