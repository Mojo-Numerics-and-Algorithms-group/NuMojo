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

import ._array_funcs as _af

@register_passable("trivial")
struct NDArrayShape[](Stringable):
    """Implements the NDArrayShape."""

    # Fields
    var _size: Int
    var _shape: StaticIntTuple[Int]

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
    fn __init__(
        inout self, owned shape: NDArrayShape, ndim: Int, offset: Int = 0
    ):
        self._offset = offset
        self._stride = StaticIntTuple[ALLOWED]()

        if (
            ndim == 1
        ):  # TODO: make sure this is present in all __init__() to account for 1D arrays.
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

    # Fields
    var _arr: DTypePointer[dtype]  # Data buffer of the items in the NDArray
    alias simd_width: Int = simdwidthof[dtype]()  # Vector size of the data type
    var info: ArrayDescriptor[dtype]  # Infomation regarding the NDArray.

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    # default constructor
    fn __init__(inout self, *shape: Int, random: Bool = False):
        """
        Example:
            NDArray[DType.int8](3,2,4)
            Returns an zero array with shape 3 x 2 x 4.
        """
        var dimension: Int = shape.__len__()
        var first_index: Int = 0
        var size: Int = 1
        var shapeInfo: List[Int] = List[Int]()
        var strides: List[Int] = List[Int]()

        for i in range(dimension):
            shapeInfo.append(shape[i])
            size *= shape[i]
            var temp: Int = 1
            for j in range(i + 1, dimension):  # temp
                temp *= shape[j]
            strides.append(temp)

        self._arr = DTypePointer[dtype].alloc(size)
        memset_zero(self._arr, size)
        self.info = ArrayDescriptor[dtype](
            dimension, first_index, size, shapeInfo, strides
        )
        if random:
            rand[dtype](self._arr, size)

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
        var dimension: Int = shape.__len__()
        var first_index: Int = 0
        var size: Int = 1
        var shapeInfo: List[Int] = List[Int]()
        var strides: List[Int] = List[Int]()

        for i in range(dimension):
            shapeInfo.append(shape[i])
            size *= shape[i]
            var temp: Int = 1
            for j in range(i + 1, dimension):  # temp
                temp *= shape[j]
            strides.append(temp)

        self._arr = DTypePointer[dtype].alloc(size)
        memset_zero(self._arr, size)
        self.info = ArrayDescriptor[dtype](
            dimension, first_index, size, shapeInfo, strides
        )

        if random:
            rand[dtype](self._arr, size)
        else:
            for i in range(size):
                self._arr[i] = value.cast[dtype]()

    fn __init__(inout self, data: List[SIMD[dtype, 1]], shape: List[Int]):
        """
        Example:
            `NDArray[DType.int8](List[Int8](1,2,3,4,5,6), shape=List[Int](2,3))`
            Returns an array with shape 3 x 2 with input values.
        """

        var dimension: Int = shape.__len__()
        var first_index: Int = 0
        var size: Int = 1
        var shapeInfo: List[Int] = List[Int]()
        var strides: List[Int] = List[Int]()

        for i in range(dimension):
            shapeInfo.append(shape[i])
            size *= shape[i]
            var temp: Int = 1
            for j in range(i + 1, dimension):  # temp
                temp *= shape[j]
            strides.append(temp)

        self._arr = DTypePointer[dtype].alloc(size)
        memset_zero(self._arr, size)
        for i in range(size):
            self._arr[i] = data[i]
        self.info = ArrayDescriptor[dtype](
            dimension, first_index, size, shapeInfo, strides
        )

    fn __init__(inout self, shape: List[Int], random: Bool = False):
        """
        Example:
            NDArray[DType.float16](List[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """

        var dimension: Int = shape.__len__()
        var first_index: Int = 0
        var size: Int = 1
        var shapeInfo: List[Int] = List[Int]()
        var strides: List[Int] = List[Int]()

        for i in range(dimension):
            shapeInfo.append(shape[i])
            size *= shape[i]
            var temp: Int = 1
            for j in range(i + 1, dimension):  # temp
                temp *= shape[j]
            strides.append(temp)

        self._arr = DTypePointer[dtype].alloc(size)
        memset_zero(self._arr, size)
        self.info = ArrayDescriptor[dtype](
            dimension, first_index, size, shapeInfo, strides
        )
        if random:
            rand[dtype](self._arr, size)

    fn __init__(
        inout self,
        shape: NDArrayShape,
        random: Bool = False,
        value: SIMD[dtype, 1] = SIMD[dtype, 1](0),  # make it Optional[]
    ):
        var dimension: Int = shape.__len__()
        var first_index: Int = 0
        var size: Int = 1
        var strides: List[Int] = List[Int]()

        for i in range(dimension):
            size *= shape.shape[i]
            var temp: Int = 1
            for j in range(i + 1, dimension):  # temp
                temp *= shape.shape[j]
            strides.append(temp)

        self._arr = DTypePointer[dtype].alloc(size)
        memset_zero(self._arr, size)
        self.info = ArrayDescriptor[dtype](
            dimension, first_index, size, shape.shape, strides
        )
        if random:
            rand[dtype](self._arr, size)
        elif value:
            for i in range(size):
                self._arr[i] = value

    # constructor when rank, ndim, weights, first_index(offset) are known
    fn __init__(
        inout self,
        ndim: Int,
        offset: Int,
        size: Int,
        shape: List[Int],
        strides: List[Int],
        coefficients: List[Int] = List[Int](),
    ):
        self._arr = DTypePointer[dtype].alloc(size)
        memset_zero(self._arr, size)
        self.info = ArrayDescriptor[dtype](
            ndim, offset, size, shape, strides, coefficients
        )

    fn __copyinit__(inout self, new: Self):
        self.info = new.info
        self._arr = DTypePointer[dtype].alloc(new.info.size)
        for i in range(new.info.size):
            self._arr[i] = new._arr[i]

    fn __moveinit__(inout self, owned existing: Self):
        self.info = existing.info
        self._arr = existing._arr

    fn __del__(owned self):
        self._arr.free()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __setitem__(inout self, index: Int, value: SIMD[dtype, 1]) raises:
        if index >= self.info.size:
            raise Error("Invalid index: index out of bound")
        self.store(index, value)

    fn __setitem__(inout self, *index: Int, value: SIMD[dtype, 1]) raises:
        var idx: Int = _get_index(index, self.info.strides)
        if idx >= self.info.size:
            raise Error("Invalid index: index out of bound")
        self.store(idx, value)

    fn __setitem__(
        inout self,
        indices: List[Int],
        val: SIMD[dtype, 1],
    ) raises:
        var idx: Int = _get_index(indices, self.info.strides)
        if idx >= self.info.size:
            raise Error("Invalid index: index out of bound")
        self.store(idx, val)

    fn __getitem__(self, index: Int) raises -> SIMD[dtype, 1]:
        """
        Example:
            `arr[15]` returns the 15th item of the array's data buffer.
        """
        if index >= self.info.size:
            raise Error("Invalid index: index out of bound")
        return self._arr.__getitem__(index)

    fn __getitem__(self, *indices: Int) raises -> SIMD[dtype, 1]:
        """
        Example:
            `arr[1,2]` returns the item of 1st row and 2nd column of the array.
        """
        if indices.__len__() != self.info.ndim:
            raise Error("Error: Length of Indices do not match the shape")

        for i in range(indices.__len__()):
            if indices[i] >= self.info.shape[i]:
                raise Error(
                    "Error: Elements of Indices exceed the shape values"
                )

        var index: Int = _get_index(indices, self.info.strides)
        return self._arr[index]

    # same as above, but explicit VariadicList
    fn __getitem__(self, indices: VariadicList[Int]) raises -> SIMD[dtype, 1]:
        """
        Example:
            `arr[VariadicList[Int](1,2)]` returns the item of 1st row and
                2nd column of the array.
        """
        if indices.__len__() != self.info.ndim:
            raise Error("Error: Length of Indices do not match the shape")

        for i in range(indices.__len__()):
            if indices[i] >= self.info.shape[i]:
                raise Error(
                    "Error: Elements of Indices exceed the shape values"
                )

        var index: Int = _get_index(indices, self.info.strides)
        return self._arr[index]

    fn __getitem__(
        self, indices: List[Int], offset: Int, coefficients: List[Int]
    ) -> SIMD[dtype, 1]:
        """
        Example:
            `arr[List[Int](1,2), 1, List[Int](1,1)]` returns the item of
            1st row and 3rd column of the array.
        """

        var index: Int = offset + _get_index(indices, coefficients)
        return self._arr.__getitem__(index)

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
        if n_slices > self.info.ndim or n_slices < self.info.ndim:
            print("Error: No of slices do not match shape")

        var ndims: Int = 0
        var spec: List[Int] = List[Int]()
        for i in range(slices.__len__()):
            self._adjust_slice_(slices[i], self.info.shape[i])
            spec.append(slices[i].unsafe_indices())
            if slices[i].unsafe_indices() != 1:
                ndims += 1

        var nshape: List[Int] = List[Int]()
        var ncoefficients: List[Int] = List[Int]()
        var nstrides: List[Int] = List[Int]()
        var nnum_elements: Int = 1

        var j: Int = 0
        for i in range(ndims):
            while spec[j] == 1:
                j += 1
            if j >= self.info.ndim:
                break
            nshape.append(slices[j].unsafe_indices())
            nnum_elements *= slices[j].unsafe_indices()
            ncoefficients.append(self.info.strides[j] * slices[j].step)
            j += 1
            # combined the two for loops, this calculates the strides for new array

        for k in range(ndims):
            var temp: Int = 1
            for j in range(k + 1, ndims):  # temp
                temp *= nshape[j]
            nstrides.append(temp)

        # row major
        var noffset: Int = 0
        for i in range(slices.__len__()):
            var temp: Int = 1
            for j in range(i + 1, slices.__len__()):
                temp *= self.info.shape[j]
            noffset += slices[i].start * temp

        var narr = Self(
            ndims, noffset, nnum_elements, nshape, nstrides, ncoefficients
        )

        # Starting index to traverse the new array
        var index = List[Int]()
        for _ in range(ndims):
            index.append(0)
        _traverse_iterative[dtype](
            self, narr, nshape, ncoefficients, nstrides, noffset, index, 0
        )

        return narr
    
    fn __getitem__(self, owned slices: List[Slice]) raises -> Self:
        """
        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).
        """

        var n_slices: Int = slices.__len__()
        if n_slices > self.info.ndim or n_slices < self.info.ndim:
            raise Error("Error: No of slices do not match shape")

        var ndims: Int = 0
        var spec: List[Int] = List[Int]()
        for i in range(slices.__len__()):
            self._adjust_slice_(slices[i], self.info.shape[i])
            spec.append(slices[i].unsafe_indices())
            if slices[i].unsafe_indices() != 1:
                ndims += 1

        var nshape: List[Int] = List[Int]()
        var ncoefficients: List[Int] = List[Int]()
        var nstrides: List[Int] = List[Int]()
        var nnum_elements: Int = 1

        var j: Int = 0
        for i in range(ndims):
            while spec[j] == 1:
                j += 1
            if j >= self.info.ndim:
                break
            nshape.append(slices[j].unsafe_indices())
            nnum_elements *= slices[j].unsafe_indices()
            ncoefficients.append(self.info.strides[j] * slices[j].step)
            j += 1

        for k in range(ndims):
            var temp: Int = 1
            for j in range(k + 1, ndims):  # temp
                temp *= nshape[j]
            nstrides.append(temp)

        # row major
        var noffset: Int = 0
        for i in range(slices.__len__()):
            var temp: Int = 1
            for j in range(i + 1, slices.__len__()):
                temp *= self.info.shape[j]
            noffset += slices[i].start * temp

        var narr = Self(
            ndims, noffset, nnum_elements, nshape, nstrides, ncoefficients
        )

        # Starting index to traverse the new array
        var index = List[Int]()
        for _ in range(ndims):
            index.append(0)
        _traverse_iterative[dtype](
            self, narr, nshape, ncoefficients, nstrides, noffset, index, 0
        )

        return narr

    fn __getitem__(self, owned *slices: Variant[Slice,Int]) raises -> Self:
        """
        Example:
            `arr[1:3, 2:4]` returns the corresponding sliced array (2 x 2).
        """
        var n_slices: Int = slices.__len__()
        if n_slices > self.info.ndim:
            raise Error("Error: No of slices greater than rank of array")
        var slice_list: List[Slice] = List[Slice]()
        for i in range(len(slices)):
            if slices[i].isa[Slice]():
                slice_list.append(slices[i]._get_ptr[Slice]()[0])
            elif slices[i].isa[Int]():
                var int: Int = slices[i]._get_ptr[Int]()[0]
                slice_list.append(Slice(int,int+1))
                # print(int,"=",Slice(int,int+1))
        if n_slices < self.info.ndim:
            for i in range(n_slices,self.info.ndim):
                print(i)
                var size_at_dim: Int = self.info.shape[i]
                slice_list.append(Slice(0,size_at_dim-1))
        var narr: Self = self[slice_list]
        return narr

    fn __int__(self) -> Int:
        return self.info.size

    fn __pos__(self) -> Self:
        return self * 1.0

    fn __neg__(self) -> Self:
        return self * -1.0

    fn __eq__(self, other: Self) -> Bool:
        return self._arr == other._arr

    fn __add__(inout self, other: SIMD[dtype, 1]) -> Self:
        return _af._math_func_one_array_one_SIMD_in_one_array_out[dtype,SIMD.__add__](self,other)

    fn __add__(inout self, other: Self) raises -> Self:
        return _af._math_func_2_array_in_one_array_out[dtype,SIMD.__add__](self,other)

    fn __radd__(inout self, rhs: SIMD[dtype, 1]) -> Self:
        return _af._math_func_one_array_one_SIMD_in_one_array_out[dtype,SIMD.__add__](self,rhs)

    fn __iadd__(inout self, other: SIMD[dtype, 1]):
        self = _af._math_func_one_array_one_SIMD_in_one_array_out[dtype,SIMD.__add__](self,other)

    fn __iadd__(inout self, other: Self)raises:
        self = _af._math_func_2_array_in_one_array_out[dtype,SIMD.__add__](self,other)

    fn __sub__(self, other: SIMD[dtype, 1]) -> Self:
        return _af._math_func_one_array_one_SIMD_in_one_array_out[dtype,SIMD.__sub__](self,other)

    fn __sub__(self, other: Self)raises -> Self:
        return _af._math_func_2_array_in_one_array_out[dtype,SIMD.__sub__](self,other)

    fn __rsub__(self, s: SIMD[dtype, 1]) -> Self:
        return -(self - s)

    fn __isub__(inout self, s: SIMD[dtype, 1]):
        self = self - s

    fn __mul__(self, other: SIMD[dtype, 1]) -> Self:
        return _af._math_func_one_array_one_SIMD_in_one_array_out[dtype,SIMD.__mul__](self,other)

    fn __mul__(self, other: Self)raises -> Self:
        return _af._math_func_2_array_in_one_array_out[dtype,SIMD.__mul__](self,other)

    fn __rmul__(self, s: SIMD[dtype, 1]) -> Self:
        return self * s

    fn __imul__(inout self, s: SIMD[dtype, 1]):
        self = self * s

    fn __imul__(inout self, s: Self)raises:
        self = self * s

    # Same as cumsum consider removing
    # fn reduce_sum(self) -> SIMD[dtype, 1]:
    #     var reduced = SIMD[dtype, 1](0.0)
    #     alias simd_width: Int = simdwidthof[dtype]()

    #     @parameter
    #     fn vectorize_reduce[simd_width: Int](idx: Int) -> None:
    #         reduced += self._arr.load[width=simd_width](idx).reduce_add()

    #     vectorize[vectorize_reduce, simd_width](self.info.size)
    #     return reduced
    
    # Same as cumprod consider removing
    # fn reduce_mul(self) -> SIMD[dtype, 1]:
    #     var reduced = SIMD[dtype, 1](0.0)
    #     alias simd_width: Int = simdwidthof[dtype]()

    #     @parameter
    #     fn vectorize_reduce[simd_width: Int](idx: Int) -> None:
    #         reduced *= self._arr.load[width=simd_width](idx).reduce_mul()

    #     vectorize[vectorize_reduce, simd_width](self.info.size)
    #     return reduced

    fn __abs__(self) -> Self:
        return abs(self)
        # var result = Self(self.info.shape)
        # alias nelts = simdwidthof[dtype]()

        # @parameter
        # fn vectorized_abs[simd_width: Int](idx: Int) -> None:
        #     result._arr.store[width=simd_width](
        #         idx, abs(self._arr.load[width=simd_width](idx))
        #     )

        # vectorize[vectorized_abs, nelts](self.info.size)
        # return result

    # all elements raised to some integer power
    fn __pow__(self, p: Int) -> Self:
        return self._elementwise_pow(p)

    # all elements raised to some corresponding array element power
    fn __pow__(self, p: Self) raises -> Self:
        if (
            self.info.size != p.info.size
        ):  # This lets us do the operation as long as they have same no of elements
            raise Error("Both arrays must have same number of elements")

        var result = Self(self.info.shape)
        alias nelts = simdwidthof[dtype]()

        @parameter
        fn vectorized_pow[simd_width: Int](idx: Int) -> None:
            result._arr.store[width=simd_width](
                idx,
                self._arr.load[width=simd_width](idx)
                ** p.load[width=simd_width](idx),
            )

        return result

    fn __ipow__(inout self, p: Int):
        self = self.__pow__(p)

    fn _elementwise_pow(self, p: Int) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_vec = self

        @parameter
        fn tensor_scalar_vectorize[simd_width: Int](idx: Int) -> None:
            new_vec._arr.store[width=simd_width](
                idx, pow(self._arr.load[width=simd_width](idx), p)
            )

        vectorize[tensor_scalar_vectorize, simd_width](self.info.size)
        return new_vec

    # ! truediv is multiplying instead of dividing right now lol, I don't know why.
    fn __truediv__(self, other: SIMD[dtype, 1]) -> Self:
        return _af._math_func_one_array_one_SIMD_in_one_array_out[dtype, SIMD.__truediv__](self,other)

    fn __truediv__(self, other: Self) raises -> Self:
        if self.info.size != other.info.size:
            raise Error("No of elements in both arrays do not match")

        return _af._math_func_2_array_in_one_array_out[dtype,SIMD.__truediv__](self, other)


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
        return self._array_to_string(0, 0)

    fn __len__(inout self) -> Int:
        return self.info.size

    fn _array_to_string(self, dimension: Int, offset: Int) -> String:
        if dimension == self.info.ndim - 1:
            var result: String = str("[\t")
            var number_of_items = self.info.shape[dimension]
            if number_of_items <= 6:  # Print all items
                for i in range(number_of_items):
                    result = (
                        result
                        + self._arr[
                            offset + i * self.info.strides[dimension]
                        ].__str__()
                    )
                    result = result + "\t"
            else:  # Print first 3 and last 3 items
                for i in range(3):
                    result = (
                        result
                        + self._arr[
                            offset + i * self.info.strides[dimension]
                        ].__str__()
                    )
                    result = result + "\t"
                result = result + "...\t"
                for i in range(number_of_items - 3, number_of_items):
                    result = (
                        result
                        + self._arr[
                            offset + i * self.info.strides[dimension]
                        ].__str__()
                    )
                    result = result + "\t"
            result = result + "]"
            return result
        else:
            var result: String = str("[")
            var number_of_items = self.info.shape[dimension]
            if number_of_items <= 6:  # Print all items
                for i in range(number_of_items):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.info.strides[dimension],
                        )
                    if i > 0:
                        result = (
                            result
                            + str(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset + i * self.info.strides[dimension],
                            )
                        )
                    if i < (number_of_items - 1):
                        result = result + "\n"
            else:  # Print first 3 and last 3 items
                for i in range(3):
                    if i == 0:
                        result = result + self._array_to_string(
                            dimension + 1,
                            offset + i * self.info.strides[dimension],
                        )
                    if i > 0:
                        result = (
                            result
                            + str(" ") * (dimension + 1)
                            + self._array_to_string(
                                dimension + 1,
                                offset + i * self.info.strides[dimension],
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
                            offset + i * self.info.strides[dimension],
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
        if self.info.size != other.info.size:
            raise Error("The lengths of two vectors do not match.")

        var sum = Scalar[dtype](0)
        for i in range(self.info.size):
            sum = sum + self[i] * other[i]
        return sum

    fn mdot(self, other: Self) raises -> Self:
        """
        Dot product of two matrix.
        Matrix A: M * N.
        Matrix B: N * L.
        """

        if (self.info.ndim != 2) or (other.info.ndim != 2):
            raise Error("The array should have only two dimensions (matrix).")

        if self.info.shape[1] != other.info.shape[0]:
            raise Error(
                "Second dimension of A does not match first dimension of B."
            )

        var new_dims = List[Int](self.info.shape[0], other.info.shape[1])
        var new_matrix = Self(new_dims)
        for row in range(self.info.shape[0]):
            for col in range(other.info.shape[1]):
                new_matrix.__setitem__(
                    List[Int](row, col),
                    self[row : row + 1, :].vdot(other[:, col : col + 1]),
                )
        return new_matrix

    fn row(self, id: Int) raises -> Self:
        """Get the ith row of the matrix."""

        if self.info.ndim > 2:
            raise Error("Only support 2-D array (matrix).")

        # If bug fixed, then
        # Tensor[dtype](shape=width, self._arr.offset(something))

        var width = self.info.shape[1]
        var buffer = Self(width)
        for i in range(width):
            buffer[i] = self._arr[i + id * width]
        return buffer

    fn col(self, id: Int) raises -> Self:
        """Get the ith column of the matrix."""

        if self.info.ndim > 2:
            raise Error("Only support 2-D array (matrix).")

        var width = self.info.shape[1]
        var height = self.info.shape[0]
        var buffer = Self(height)
        for i in range(height):
            buffer[i] = self._arr[id + i * width]
        return buffer

    # * same as mdot
    fn rdot(self, other: Self) raises -> Self:
        """
        Dot product of two matrix.
        Matrix A: M * N.
        Matrix B: N * L.
        """

        if (self.info.ndim != 2) or (other.info.ndim != 2):
            raise Error("The array should have only two dimensions (matrix).")
        if self.info.shape[1] != other.info.shape[0]:
            raise Error(
                "Second dimension of A does not match first dimension of B."
            )

        var new_dims = List[Int](self.info.shape[0], other.info.shape[1])
        var new_matrix = Self(new_dims)
        for row in range(new_dims[0]):
            for col in range(new_dims[1]):
                new_matrix[col + row * new_dims[1]] = self.row(row).vdot(
                    other.col(col)
                )
        return new_matrix

    fn size(self) -> Int:
        return self.info.size

    fn num_elements(self) -> Int:
        return self.info.size

    # TODO: move this initialization to the Fields and constructor
    fn shape(self) -> NDArrayShape:
        var shapeNDArray: NDArrayShape = NDArrayShape(self.info.shape)
        return shapeNDArray

    fn load[width: Int](self, idx: Int) -> SIMD[dtype, width]:
        """
        Loads a SIMD element of size `width` at the given index `idx`.
        """
        return self._arr.load[width=width](idx)

    # TODO: we should add checks to make sure user don't load out of bound indices, but that will overhead, figure out later
    fn load[width: Int = 1](self, *indices: Int) -> SIMD[dtype, width]:
        """
        Loads a SIMD element of size `width` at given variadic indices argument.
        """
        var index: Int = _get_index(indices, self.info.strides)
        return self._arr.load[width=width](index)

    fn load[width: Int = 1](self, indices: VariadicList[Int]) -> SIMD[dtype, 1]:
        """
        Loads a SIMD element of size `width` at given VariadicList of indices.
        """
        var idx: Int = _get_index(indices, self.info.strides)
        return self._arr.load[width=width](idx)

    fn load[width: Int = 1](self, indices: List[Int]) -> SIMD[dtype, 1]:
        """
        Loads a SIMD element of size `width` at given List of indices.
        """
        var idx: Int = _get_index(indices, self.info.strides)
        return self._arr.load[width=width](idx)

    fn store[width: Int](inout self, idx: Int, val: SIMD[dtype, width]):
        """
        Stores the SIMD element of size `width` at index `idx`.
        """
        self._arr.store[width=width](idx, val)

    fn store[
        width: Int = 1
    ](inout self, indices: VariadicList[Int], val: SIMD[dtype, width]):
        """
        Stores the SIMD element of size `width` at the given Variadic list of indices.
        """
        var idx: Int = _get_index(indices, self.info.strides)
        self._arr.store[width=width](idx, val)

    fn store[
        width: Int = 1
    ](inout self, *indices: Int, val: SIMD[dtype, width]):
        """
        Stores the SIMD element of size `width` at the given variadic indices argument.
        """
        var idx: Int = _get_index(indices, self.info.strides)
        self._arr.store[width=width](idx, val)

    # not urgent: argpartition, byteswap, choose, conj, dump, getfield
    # partition, put, repeat, searchsorted, setfield, squeeze, swapaxes, take,
    # tobyets, tofile, view

    fn all(self):
        # We might need to figure out how we want to handle truthyness before can do this
        pass
            

    fn any(self):
        pass

    fn argmax(self):
        pass

    fn argmin(self):
        pass

    fn argsort(self):
        pass

    fn astype(self):
        pass

    fn clip(self):
        pass

    fn compress(self):
        pass

    fn copy(self):
        pass

    fn cumprod(self)->Scalar[dtype]:
        """
        Cumulative product of a array.
        
        Returns:
            The cumulative product of the array as a SIMD Value of `dtype`.
        """
        return cumprod[dtype](self)

    fn cumsum(self)->Scalar[dtype]:
        """
        Cumulative Sum of a array.
        
        Returns:
            The cumulative sum of the array as a SIMD Value of `dtype`.
        """
        return cumsum[dtype](self)

    fn diagonal(self):
        pass

    fn fill(self):
        pass

    fn flatten(self):
        pass

    fn item(self):
        pass

    fn max(self):
        pass

    fn min(self):
        pass

    fn mean(self:Self,axis: Int)raises->Self:
        """
        Mean of array elements over a given axis.
        Args:
            array: NDArray.
            axis: The axis along which the mean is performed.
        Returns:
            An NDArray.

        """
        return mean(self,axis)

    fn mean(self)raises->Scalar[dtype]:
        """
        Cumulative mean of a array.
        
        Returns:
            The cumulative mean of the array as a SIMD Value of `dtype`.
        """
        return cummean[dtype](self)

    fn nonzero(self):
        pass

    fn prod(self:Self,axis: Int)raises->Self:
        """
        Product of array elements over a given axis.
        Args:
            array: NDArray.
            axis: The axis along which the product is performed.
        Returns:
            An NDArray.
        """

        return prod(self,axis)

    fn ravel(self):
        pass

    fn resize(self):
        pass

    fn round(self):
        pass

    fn sort(self):
        pass

    fn sum(self:Self,axis: Int)raises->Self:
        """
        Sum of array elements over a given axis.
        Args:
            axis: The axis along which the sum is performed.
        Returns:
            An NDArray.
        """
        return sum(self,axis)

    fn stdev(self):
        pass

    fn tolist(self):
        pass

    fn tostring(self):
        pass

    fn trace(self):
        pass

    fn transpose(self):
        pass

    fn variance(self):
        pass

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

        if self.info.size != num_elements_new:
            raise Error("Cannot reshape: Number of elements do not match.")

        self.info.ndim = ndim_new
        var shape_new: List[Int] = List[Int]()
        var strides_new: List[Int] = List[Int]()

        for i in range(ndim_new):
            shape_new.append(Shape[i])
            var temp: Int = 1
            for j in range(i + 1, ndim_new):  # temp
                temp *= Shape[j]
            strides_new.append(temp)

        self.info.shape = shape_new
        self.info.strides = strides_new
        # self.shape.shape = shape_new # current ndarray doesn't have NDArray shape field

    fn unsafe_ptr(self) -> DTypePointer[dtype, 0]:
        return self._arr
