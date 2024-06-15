############################################################################################
# * ROW MAJOR ND ARRAYS
# * Last updated: 2024-06-13
############################################################################################

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
fn _traverse_iterative[
    dtype: DType
](
    orig: Array[dtype],
    inout narr: Array[dtype],
    dims: List[Int],
    weights: List[Int],
    offset_index: Int,
    inout index: List[Int],
    depth: Int,
):
    if depth == dims.__len__():
        var idx = offset_index + _get_index(index, weights)
        var temp = orig._arr.load[width=1](idx)
        narr._arr[idx] = temp
        return

    for i in range(dims[depth]):
        index[depth] = i
        var newdepth = depth + 1
        _traverse_iterative(
            orig, narr, dims, weights, offset_index, index, newdepth
        )


@value
struct arrayDescriptor[dtype: DType = DType.float32]():
    var rank: Int  # No of dimensions
    var dims: List[Int]  # size of each dimension
    var weights: List[Int]  # coefficients
    var num_elements: Int
    var first_index: Int

    fn __init__(
        inout self,
        rank: Int,
        dims: List[Int],
        weights: List[Int],
        first_index: Int,
    ):
        self.rank = rank
        self.dims = dims
        self.weights = weights
        self.num_elements = 1
        self.first_index = first_index
        for i in range(self.dims.__len__()):
            self.num_elements *= self.dims[i]


# * COLUMN MAJOR INDEXING
struct Array[dtype: DType = DType.float32](Stringable):
    var _arr: DTypePointer[dtype]
    var _arrayInfo: arrayDescriptor[dtype]
    alias simd_width: Int = simdwidthof[dtype]()

    # default constructor
    fn __init__(inout self, *dims: Int):
        var weight: List[Int] = List[Int]()
        var dim: List[Int] = List[Int]()

        for i in range(dims.__len__()):
            dim.append(dims[i])
            var temp: Int = 1
            # columns major
            for j in range(i + 1):
                temp *= dims[j]
            weight.append(temp)

        self._arrayInfo = arrayDescriptor[dtype](
            rank=dims.__len__(), dims=dim, weights=weight, first_index=0
        )
        self._arr = DTypePointer[dtype].alloc(self._arrayInfo.num_elements)
        memset_zero(self._arr, self._arrayInfo.num_elements)

    # constructor when rank, dims, weights, first_index(offset) are known
    fn __init__(
        inout self,
        rank: Int,
        dims: List[Int],
        weights: List[Int],
        first_index: Int,
    ):
        self._arrayInfo = arrayDescriptor[dtype](
            rank=rank, dims=dims, weights=weights, first_index=first_index
        )
        self._arr = DTypePointer[dtype].alloc(self._arrayInfo.num_elements)
        memset_zero(self._arr, self._arrayInfo.num_elements)

    fn __init__(inout self, dims: VariadicList[Int], random: Bool = False):
        var weight: List[Int] = List[Int]()
        var dim: List[Int] = List[Int]()
        for i in range(dims.__len__()):
            dim.append(dims[i])
            var temp: Int = 1
            # columns major
            # for j in range(i):
            #     temp *= dims[j]
            # weight.append(temp)
            for j in range(i + 1, dims.__len__()):
                temp *= dims[j]
            weight.append(temp)

        self._arrayInfo = arrayDescriptor[dtype](
            rank=dims.__len__(), dims=dim, weights=weight, first_index=0
        )
        self._arr = DTypePointer[dtype].alloc(self._arrayInfo.num_elements)
        memset_zero(self._arr, self._arrayInfo.num_elements)
        if random:
            rand[dtype](self._arr, self._arrayInfo.num_elements)

    fn __init__(inout self, dims: List[Int], random: Bool = False):
        var weight: List[Int] = List[Int]()
        for i in range(dims.__len__()):
            var temp: Int = 1
            for j in range(i):
                temp *= dims[j]
            weight.append(temp)

        self._arrayInfo = arrayDescriptor[dtype](
            rank=dims.__len__(), dims=dims, weights=weight, first_index=0
        )
        self._arr = DTypePointer[dtype].alloc(self._arrayInfo.num_elements)
        memset_zero(self._arr, self._arrayInfo.num_elements)
        if random:
            rand[dtype](self._arr, self._arrayInfo.num_elements)

    fn __copyinit__(inout self, new: Self):
        self._arrayInfo = new._arrayInfo
        self._arr = DTypePointer[dtype].alloc(self._arrayInfo.num_elements)
        for i in range(self._arrayInfo.num_elements):
            self._arr[i] = new._arr[i]

    fn __moveinit__(inout self, owned existing: Self):
        self._arrayInfo = existing._arrayInfo
        self._arr = existing._arr
        existing._arr = DTypePointer[dtype]()

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

    fn __setitem__(inout self, indices: List[Int], val: SIMD[dtype, 1]):
        var index: Int = _get_index(indices, self._arrayInfo.weights)
        self._arr[index] = val

    fn __setitem__(inout self, indices: VariadicList[Int], val: SIMD[dtype, 1]):
        var index: Int = _get_index(indices, self._arrayInfo.weights)
        self._arr[index] = val

    fn __getitem__(inout self, idx: Int) -> SIMD[dtype, 1]:
        return self._arr.__getitem__(idx)

    fn __getitem__(inout self, *indices: Int) raises -> SIMD[dtype, 1]:
        if indices.__len__() != self._arrayInfo.rank:
            raise Error("Error: Length of Indices do not match the shape")

        for i in range(indices.__len__()):
            if indices[i] >= self._arrayInfo.dims[i]:
                raise Error(
                    "Error: Elements of Indices exceed the shape values"
                )

        var index: Int = self._arrayInfo.first_index + _get_index(
            indices, self._arrayInfo.weights
        )
        return self._arr[index]

    # same as above, but explicit VariadicList
    fn __getitem__(self, indices: VariadicList[Int]) raises -> SIMD[dtype, 1]:
        if indices.__len__() != self._arrayInfo.rank:
            raise Error("Error: Length of Indices do not match the shape")

        for i in range(indices.__len__()):
            if indices[i] >= self._arrayInfo.dims[i]:
                raise Error(
                    "Error: Elements of Indices exceed the shape values"
                )

        var index: Int = self._arrayInfo.first_index + _get_index(
            indices, self._arrayInfo.weights
        )
        return self._arr[index]

    fn __getitem__(self, owned *slices: Slice) raises -> Self:
        var n_slices: Int = slices.__len__()
        if n_slices > self._arrayInfo.rank or n_slices < self._arrayInfo.rank:
            print("Error: No of slices do not match shape")

        var nrank: Int = 0
        var spec: List[Int] = List[Int]()
        for i in range(slices.__len__()):
            self._adjust_slice_(slices[i], self._arrayInfo.dims[i])
            spec.append(slices[i].unsafe_indices())
            if slices[i].unsafe_indices() != 1:
                nrank += 1

        var dims: List[Int] = List[Int]()
        var weights: List[Int] = List[Int]()

        var j: Int = 0
        for _ in range(nrank):
            while spec[j] == 1:
                j += 1
            if j >= self._arrayInfo.rank:
                break
            dims.append(slices[j].unsafe_indices())
            weights.append(self._arrayInfo.weights[j] * slices[j].step)
            j += 1

        # columns major
        # var offset_index: Int = 0
        # for i in range(slices.__len__()):
        #     var temp: Int = 1
        #     for j in range(i):
        #         temp *= self._arrayInfo.dims[j]
        #     offset_index += (slices[i].start * temp)

        # row major
        var offset_index: Int = 0
        for i in range(slices.__len__()):
            var temp: Int = 1
            for j in range(i + 1, slices.__len__()):
                temp *= self._arrayInfo.dims[j]
            offset_index += slices[i].start * temp

        var narr = self
        narr._arrayInfo.rank = nrank
        narr._arrayInfo.dims = dims
        narr._arrayInfo.weights = weights
        narr._arrayInfo.first_index = offset_index

        var index = List[Int]()
        for _ in range(nrank):
            index.append(0)
        _traverse_iterative[dtype](
            self, narr, dims, weights, offset_index, index, 0
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

    fn __del__(owned self):
        self._arr.free()

    fn __len__(inout self) -> Int:
        return self._arrayInfo.num_elements

    fn __int__(self) -> Int:
        return self._arrayInfo.num_elements

    fn __pos__(self) -> Self:
        return self * 1.0

    fn __neg__(self) -> Self:
        return self * -1.0

    fn __str__(self) -> String:
        return (
            self._array_to_string(0, self._arrayInfo.first_index)
            + "\n"
            + "Shape: "
            + self._arrayInfo.dims.__str__()
            + "\t"
            + "dtype: "
            + dtype.__str__()
            + "\n"
        )

    fn _array_to_string(self, dimension: Int, offset: Int) -> String:
        if dimension == len(self._arrayInfo.dims) - 1:
            var result: String = str("[\t")
            for i in range(self._arrayInfo.dims[dimension]):
                if i > 0:
                    result = result + "\t"
                result = (
                    result
                    + self._arr[
                        offset + i * self._arrayInfo.weights[dimension]
                    ].__str__()
                )
            result = result + "\t]"
            return result
        else:
            var result: String = str("[")
            for i in range(self._arrayInfo.dims[dimension]):
                result = result + self._array_to_string(
                    dimension + 1,
                    offset + i * self._arrayInfo.weights[dimension],
                )
                if i < (self._arrayInfo.dims[dimension] - 1):
                    result += "\n"
            result = result + "]"
            return result

    # for columns major, haven't finished it yet
    # fn __str__(self) -> String:
    #     return self._array_to_string(0, self._arrayInfo.first_index)

    # fn _array_to_string(self, dimension:Int, offset:Int) -> String :
    #     if dimension == 0:
    #         var result: String = str("[\t")
    #         for i in range(self._arrayInfo.dims[dimension+1]):
    #             if i > 0:
    #                 result = result + "\t"
    #             result = result + self._arr[offset + i * self._arrayInfo.weights[dimension]].__str__()
    #         result = result + "\t]"
    #         return result
    #     else:
    #         var result: String = str("[")
    #         for i in range(self._arrayInfo.dims[dimension]):
    #             result = result + self._array_to_string(dimension - 1, offset + i * self._arrayInfo.weights[dimension])
    #             if i < (self._arrayInfo.dims[dimension]-1):
    #                 result += "\n"
    #         result = result + "]"
    #         return result

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

        vectorize[elemwise_vectorize, simd_width](self._arrayInfo.num_elements)
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

        vectorize[elemwise_arithmetic, simd_width](self._arrayInfo.num_elements)
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

        vectorize[vectorize_reduce, simd_width](self._arrayInfo.num_elements)
        return reduced

    fn __matmul__(inout self, other: Self) -> Scalar[dtype]:
        return self._elementwise_array_arithmetic[SIMD.__mul__](
            other
        )._reduce_sum()

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

        vectorize[tensor_scalar_vectorize, simd_width](
            self._arrayInfo.num_elements
        )
        return new_vec

    # ! truediv is multiplying instead of dividing right now lol, I don't know why.
    fn __truediv__(self, s: Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__truediv__](s)

    fn __truediv__(self, other: Self) raises -> Self:
        if self._arrayInfo.num_elements != other._arrayInfo.num_elements:
            raise Error("No of elements in both arrays do not match")

        return self._elementwise_array_arithmetic[SIMD.__truediv__](other)

    fn __itruediv__(inout self, s: Scalar[dtype]):
        self = self.__truediv__(s)

    fn __itruediv__(inout self, other: Self) raises:
        self = self.__truediv__(other)

    fn __rtruediv__(self, s: Scalar[dtype]) -> Self:
        return self.__truediv__(s)
