############################################################################################
# * ROW MAJOR ND ARRAYS
# * Last updated: 2024-06-13
############################################################################################

"""Implements basic object methods for working with N-dimensional arrays.

"""

from random import rand
from testing import assert_raises

from builtin.math import pow

fn _get_index(indices: VariadicList[Int], weights: List[Int]) -> Int:
    var idx: Int = 0
    for i in range(weights.__len__()):
        idx += indices[i] * weights[i]
    return idx


fn _get_index(indices: List[Int], weights: List[Int]) -> Int:
    var idx: Int = 0
    for i in range(weights.__len__()):
        idx += indices[i] * weights[i]
    return idx

# TODO: need to figure out why mojo crashes at the iterative calling step
# fn _traverse_iterative[
#     dtype: DType
# ](
#     orig: Array[dtype],
#     inout narr: Array[dtype],
#     ndim: List[Int],
#     weights: List[Int],
#     offset: Int,
#     inout index: List[Int],
#     depth: Int,
# ):
#     if depth == ndim.__len__():
#         var idx = offset + _get_index(index, weights)
#         var temp = orig._arr.load[width=1](idx)
#         narr._arr[idx] = temp
#         return

#     for i in range(ndim[depth]):
#         index[depth] = i
#         var newdepth = depth + 1
#         _traverse_iterative(
#             orig, narr, ndim, weights, offset, index, newdepth
#         )

fn _traverse_iterative[
    dtype: DType
](  
    orig: NDArray[dtype],
    inout narr: NDArray[dtype],
    ndim: List[Int],
    coefficients: List[Int],
    strides: List[Int],
    offset: Int,
    inout index: List[Int],
    depth: Int,
):
    if depth == ndim.__len__():
        var idx = offset + _get_index(index, coefficients)
        var nidx = _get_index(index, strides)
        var temp = orig._arr.load[width=1](idx)
        # narr._arr.__setitem__(nidx, temp)
        # narr._arr[nidx] = temp
        narr.__setitem__(nidx, temp)
        # narr.__setitem__(index, temp)
        return

    for i in range(ndim[depth]):
        index[depth] = i
        var newdepth = depth + 1
        _traverse_iterative(
            orig, narr, ndim, coefficients, strides, offset, index, newdepth
        )

@value
struct NDArrayShape[dtype:DType = DType.int32](Stringable):
    var shape: List[Int]

    fn __init__(inout self, shape: List[Int]):
        self.shape = shape

    fn __init__(inout self, num:Int):
        self.shape = List[Int](num)

    fn __init__(inout self, shape:VariadicList[Int]):
        self.shape = List[Int]()
        for i in range(shape.__len__()):
            self.shape.append(shape[i])

    fn __eq__(self, other: Self) -> Bool:
        if self.shape.__len__() != other.shape.__len__():
            return False

        for i in range(self.shape.__len__()):
            if self.shape[i] != other.shape[i]:
                return False
        return True

    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    fn __str__(self) -> String:
        return self.shape.__str__()

@value
struct arrayDescriptor[dtype: DType = DType.float32]():

    var ndim: Int  # Number of dimensions of the array
    var offset: Int  # Offset of array data in buffer.
    var size: Int  # Number of elements in the array
    var shape: List[Int]  # size of each dimension
    var strides: List[Int]  # Tuple of bytes to step in each dimension 
                            # when traversing an array.
    var coefficients: List[Int]  # coefficients

    fn __init__(
        inout self,
        ndim: Int,
        offset: Int,
        size: Int,
        shape: List[Int],
        strides: List[Int],
        coefficients: List[Int] = List[Int](),
    ):
        self.ndim = ndim
        self.offset = offset
        self.size = size
        self.shape = shape
        self.strides = strides
        self.coefficients = coefficients

# * COLUMN MAJOR INDEXING
struct NDArray[dtype: DType = DType.float32](Stringable):
    """The N-dimensional array (NDArray).

    The array can be uniquely defined by three parameters:
        1. The data buffer of all items.
        2. The shape of the array.
        3. Is the array row-major ('C') or column-major ('F')?
            Currently, we only implement methods using row-major.
    """

    var _arr: DTypePointer[dtype]  # Data buffer of the items in the NDArray
    alias simd_width: Int = simdwidthof[dtype]()  # Vector size of the data type
    var info: arrayDescriptor[dtype]  # Infomation regarding the NDArray.

    # default constructor
    fn __init__(inout self, *shape: Int):
        """
        Example:
            NDArray[DType.int8](3,2,4)
            Returns an empty array with shape 3 x 2 x 4.
        """
        var dimension: Int = shape.__len__()
        var first_index: Int = 0
        var size: Int = 1
        var shapeInfo: List[Int] = List[Int]()
        var strides: List[Int] = List[Int]()

        print(shape[0], shape[1])
        for i in range(dimension):
            shapeInfo.append(shape[i])
            size *= shape[i]
            var temp: Int = 1
            for j in range(i + 1):  # temp
                temp *= shape[j]
            strides.append(temp)

        self._arr = DTypePointer[dtype].alloc(size)
        memset_zero(self._arr, size)
        self.info = arrayDescriptor[dtype](
            dimension, first_index, size, shapeInfo, strides
        )
    
    fn __init__(inout self, shape: NDArrayShape, random: Bool = False, value: SIMD[dtype, 1] = SIMD[dtype, 1](0)):

        var dimension: Int = shape.shape.__len__()
        var first_index: Int = 0
        var size: Int = 1
        var shapeInfo: List[Int] = List[Int]()
        var strides: List[Int] = List[Int]()

        for i in range(dimension):
            shapeInfo.append(shape.shape[i])
            size *= shape.shape[i]
            var temp: Int = 1
            for j in range(i + 1, dimension):  # temp
                temp *= shape.shape[j]
            strides.append(temp)

        self._arr = DTypePointer[dtype].alloc(size)
        memset_zero(self._arr, size)
        for i in range(size):
            self._arr[i] = value
        self.info = arrayDescriptor[dtype](
            dimension, first_index, size, shapeInfo, strides
        )
        if random:
            rand[dtype](self._arr, size)
    

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
        self.info = arrayDescriptor[dtype](
            ndim, offset, size, shape, strides, coefficients
        )

    fn __init__(inout self, shape: VariadicList[Int], random: Bool = False):
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
            for j in range(i + 1, shape.__len__()):  # temp
                temp *= shape[j]
            strides.append(temp)

        self._arr = DTypePointer[dtype].alloc(size)
        memset_zero(self._arr, size)
        self.info = arrayDescriptor[dtype](
            dimension, first_index, size, shapeInfo, strides
        )

        if random:
            rand[dtype](self._arr, size)

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
            for j in range(i + 1):  # temp
                temp *= shape[j]
            strides.append(temp)

        self._arr = DTypePointer[dtype].alloc(size)
        memset_zero(self._arr, size)
        self.info = arrayDescriptor[dtype](
            dimension, first_index, size, shapeInfo, strides
        )
        if random:
            rand[dtype](self._arr, size)

    fn __copyinit__(inout self, new: Self):
        self.info = new.info
        self._arr = DTypePointer[dtype].alloc(new.info.size)
        for i in range(new.info.size):
            self._arr[i] = new._arr[i]

    fn __moveinit__(inout self, owned existing: Self):
        self.info = existing.info
        self._arr = existing._arr

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

    fn __setitem__(inout self, idx: Int, val: SIMD[dtype, 1]):
        self._arr.__setitem__(idx, val)

    fn __setitem__(
        inout self,
        indices: List[Int],
        val: SIMD[dtype, 1],
    ):
        var index: Int = _get_index(indices, self.info.strides)
        self._arr.__setitem__(index, val)

    fn __getitem__(self, idx: Int) -> SIMD[dtype, 1]:
        """
        Example:
            `arr[15]` returns the 15th item of the array's data buffer.
        """
        return self._arr.__getitem__(idx)

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
        return self._arr[index]

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
        for _ in range(ndims):
            while spec[j] == 1:
                j += 1
            if j >= self.info.ndim:
                break
            nshape.append(slices[j].unsafe_indices())
            nnum_elements *= slices[j].unsafe_indices()
            ncoefficients.append(self.info.strides[j] * slices[j].step)
            j += 1

        for i in range(ndims):
            var temp: Int = 1
            for j in range(i + 1, ndims):  # temp
                temp *= nshape[j]
            nstrides.append(temp)

        # row major
        var noffset: Int = 0
        for i in range(slices.__len__()):
            var temp: Int = 1
            for j in range(i + 1, slices.__len__()):
                temp *= self.info.shape[j]
            noffset += slices[i].start * temp

        var narr = Self(ndims, noffset, nnum_elements, nshape, nstrides, ncoefficients)
        var index = List[Int]()
        for _ in range(ndims):
            index.append(0)
        
        _traverse_iterative[dtype](
            self, narr, nshape, ncoefficients, nstrides, noffset, index, 0
        )
        return narr

    # I have to implement some kind of Index struct like the tensor Index() so that we don't have to write VariadicList everytime
    # fn __setitem__(inout self, indices:VariadicList[Int], value:Scalar[dtype]) raises:

    #     if indices.__len__() != self._shape.__len__():
    #         with assert_raises():
    #             raise "Error: Indices do not match the shape"

    #     for i in range(indices.__len__()):
    #         if indices[i] >= self._shape[i]:
    #             with assert_raises():
    #                 raise "Error: Indices do not match the shape"

    #     var index: Int = 0
    #     for i in range(indices.__len__()):
    #         var temp: Int = 1
    #         for j in range(i+1, self._shape.__len__()):
    #             temp *= self._shape[j]
    #         index += (indices[i] * temp)

    #     self._arr[index] = value

    # fn __del__(owned self):
    #     self._arr.free()

    fn shape(self) -> NDArrayShape:
        var shapeNDArray: NDArrayShape = NDArrayShape(self.info.shape)
        return shapeNDArray

    fn load[width:Int](self, idx:Int) -> SIMD[dtype, width]:
        return self._arr.load[width=width](idx)

    # fn load[width:Int = 1](self, *indices:Int) -> SIMD[dtype, width]:
    #     var index: Int = _get_index(indices, self.info.strides)
    #     return self._arr.load[width=width](index)

    # fn load[width:Int = 1](self, indices:VariadicList[Int]) -> SIMD[dtype, 1]:
    #     var index: Int = _get_index(indices, self.info.strides)
    #     return self._arr.load[width=width](index)

    fn store[width:Int](inout self, idx:Int, val:SIMD[dtype, width]):
        self._arr.store[width=width](idx, val)

    # fn store[width:Int = 1](self, indices:VariadicList[Int], val:SIMD[dtype, width]):
    #     var index: Int = _get_index(indices, self.info.strides)
    #     self._arr.store[width=width](index, val)

    fn size(self) -> Int:
        return self.info.size

    fn __len__(inout self) -> Int:
        return self.info.size

    fn __int__(self) -> Int:
        return self.info.size

    fn __pos__(self) -> Self:
        return self * 1.0

    fn __neg__(self) -> Self:
        return self * -1.0

    fn __str__(self) -> String:
        return self._array_to_string(0, 0)

    fn _array_to_string(self, dimension:Int, offset:Int) -> String:
        if dimension == self.info.ndim - 1:
            var result: String = str("[\t")
            var number_of_items = self.info.shape[dimension]
            if number_of_items <= 6:  # Print all items
                for i in range(number_of_items):
                    result = result + self._arr[offset + i * self.info.strides[dimension]].__str__()
                    result = result + "\t"
            else:  # Print first 3 and last 3 items
                for i in range(3):
                    result = result + self._arr[offset + i * self.info.strides[dimension]].__str__()
                    result = result + "\t"
                result = result + "...\t"
                for i in range(number_of_items-3, number_of_items):
                    result = result + self._arr[offset + i * self.info.strides[dimension]].__str__()
                    result = result + "\t"
            result = result + "]"
            return result
        else:
            var result: String = str("[")
            var number_of_items = self.info.shape[dimension]
            if number_of_items <= 6:  # Print all items
                for i in range(number_of_items):
                    if i == 0:
                        result = result + self._array_to_string(dimension + 1, offset + i * self.info.strides[dimension])
                    if i > 0:
                        result = result + str(" ") * (dimension+1) + self._array_to_string(dimension + 1, offset + i * self.info.strides[dimension])
                    if i < (number_of_items-1):
                        result = result + "\n"
            else:  # Print first 3 and last 3 items
                    for i in range(3):
                        if i == 0:
                            result = result + self._array_to_string(dimension + 1, offset + i * self.info.strides[dimension])
                        if i > 0:
                            result = result + str(" ") * (dimension+1) + self._array_to_string(dimension + 1, offset + i * self.info.strides[dimension])
                        if i < (number_of_items-1):
                            result += "\n"
                    result = result + "...\n"
                    for i in range(number_of_items-3, number_of_items):
                        result = result + str(" ") * (dimension+1) + self._array_to_string(dimension + 1, offset + i * self.info.strides[dimension])
                        if i < (number_of_items-1):
                            result = result + "\n"
            result = result + "]"
            return result

    fn __eq__(self, other: Self) -> Bool:
        return self._arr == other._arr

    # # ARITHMETICS
    fn _elementwise_scalar_arithmetic[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width], SIMD[dtype, width]
        ) -> SIMD[dtype, width]
    ](self, s: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = self

        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._arr.store[width=simd_width](
                idx,
                func[dtype, simd_width](
                    SIMD[dtype, simd_width](s),
                    self._arr.load[width=simd_width](idx),
                ),
            )

        vectorize[elemwise_vectorize, simd_width](self.info.size)
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
            new_vec._arr.store[width=simd_width](
                index,
                func[dtype, simd_width](
                    self._arr.load[width=simd_width](index),
                    other._arr.load[width=simd_width](index),
                ),
            )

        vectorize[elemwise_arithmetic, simd_width](self.info.size)
        return new_vec

    fn __add__(inout self, other: Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__add__](other)

    fn __add__(inout self, other: Self) -> Self:
        return self._elementwise_array_arithmetic[SIMD.__add__](other)

    fn __radd__(inout self, s: Scalar[dtype]) -> Self:
        return self + s

    fn __iadd__(inout self, s: Scalar[dtype]):
        self = self + s

    fn __sub__(self, other: Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__sub__](other)

    fn __sub__(self, other: Self) -> Self:
        return self._elementwise_array_arithmetic[SIMD.__sub__](other)

    fn __rsub__(self, s: Scalar[dtype]) -> Self:
        return -(self - s)

    fn __isub__(inout self, s: Scalar[dtype]):
        self = self - s

    fn __mul__(self, s: Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__mul__](s)

    fn __mul__(self, other: Self) -> Self:
        return self._elementwise_array_arithmetic[SIMD.__mul__](other)

    fn __rmul__(self, s: Scalar[dtype]) -> Self:
        return self * s

    fn __imul__(inout self, s: Scalar[dtype]):
        self = self * s

    fn _reduce_sum(self) -> Scalar[dtype]:
        var reduced = Scalar[dtype](0.0)
        alias simd_width: Int = simdwidthof[dtype]()

        @parameter
        fn vectorize_reduce[simd_width: Int](idx: Int) -> None:
            reduced[0] += self._arr.load[width=simd_width](idx).reduce_add()

        vectorize[vectorize_reduce, simd_width](self.info.size)
        return reduced

    fn __matmul__(self, other: Self) -> Scalar[dtype]:
        return self._elementwise_array_arithmetic[SIMD.__mul__](
            other
        )._reduce_sum()

    fn vdot(self, other: Self) raises -> Scalar[dtype]:
        """
        Inner product of two vectors.
        """
        if self.info.size != other.info.size:
            raise Error("The lengths of two vectors do not match.")
        return self._elementwise_array_arithmetic[SIMD.__mul__](
            other
        )._reduce_sum()

    fn mdot(self, other: Self) raises -> Self:
        """
        Dot product of two matrix.
        Matrix A: M * N.
        Matrix B: N * L.
        """

        print("A:", self.info.shape[0], " x ", self.info.shape[1])
        print("B:", other.info.shape[0], " x ", other.info.shape[1])
        print("AB:", self.info.shape[0], " x ", other.info.shape[1])

        if (self.info.ndim != 2) or (other.info.ndim != 2):
            raise Error("The array should have only two dimensions (matrix).")
        if self.info.shape[1] != other.info.shape[0]:
            raise Error("Second dimension of A does not match first dimension of B.")

        var new_dims = List[Int](self.info.shape[0], other.info.shape[1])
        var new_matrix = Self(new_dims)
        for row in range(self.info.shape[0]):
            for col in range(other.info.shape[1]):
                new_matrix.__setitem__(
                                        List[Int](row, col),
                                        self[row:row+1, :].vdot(other[:, col:col+1])
                                        )
        return new_matrix

    fn __pow__(self, p: Int) -> Self:
        return self._elementwise_pow(p)

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
    fn __truediv__(self, s: Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__truediv__](s)

    fn __truediv__(self, other: Self) raises -> Self:
        if self.info.size != other.info.size:
            raise Error("No of elements in both arrays do not match")

        return self._elementwise_array_arithmetic[SIMD.__truediv__](other)

    fn __itruediv__(inout self, s: Scalar[dtype]):
        self = self.__truediv__(s)

    fn __itruediv__(inout self, other: Self) raises:
        self = self.__truediv__(other)

    fn __rtruediv__(self, s: Scalar[dtype]) -> Self:
        return self.__truediv__(s)
