############################################################################################
# * COLUMN MAJOR ND ARRAYS
# * Last updated: 2024-06-13
############################################################################################

from random import rand
from testing import assert_raises

from builtin.math import pow

fn _get_index(indices: VariadicList[Int], weights:List[Int]) -> Int:
    var index: Int = 0
    for i in range(weights.__len__()):
        index += indices[i] * weights[i]
    return index

fn _get_index(indices: List[Int], weights:List[Int]) -> Int:
    var index: Int = 0
    for i in range(weights.__len__()):
        index += indices[i] * weights[i]
    return index

# TODO: need to figure out why mojo crashes at the iterative calling step
# fn _traverse_iterative[dtype:DType](inout orig:array[dtype], inout narr:array[dtype], dims:List[Int], weights:List[Int], offset_index:Int,  inout index:List[Int], depth:Int) raises:

#     if depth == dims.__len__():
#         var idx = offset_index + _get_index(index, weights)
#         # narr.__setitem__(index, orig._arr.__getitem__(idx))
#         var temp = orig._arr.load[width=1](idx)
#         # narr._arr.store[width=1](idx, temp)
#         var nidx = _get_index(index, List[Int](1,2)) 
#         narr._arr[nidx] = temp
#         return 

#     for i in range(dims[depth]):
#         index[depth] = i
#         var newdepth = depth + 1
#         _traverse_iterative(orig, narr, dims, weights, offset_index, index, newdepth)

#     print("depth: ", depth)
#     print(index[0], index[1])


@value
struct dataInfo[dtype: DType=DType.float32]():
    var rank:Int # No of dimensions
    var dims:List[Int] # size of each dimension
    var weights:List[Int] # coefficients
    var num_elements: Int
    var first_index: Int

    fn __init__(inout self, rank:Int, dims:List[Int], weights:List[Int], first_index:Int):
        self.rank = rank 
        self.dims = dims
        self.weights = weights
        self.num_elements = 1
        self.first_index = first_index
        for i in range(self.dims.__len__()):
            self.num_elements *= self.dims[i]

# * COLUMN MAJOR INDEXING
struct array[dtype: DType = DType.float32](Stringable):
    var _arr: DTypePointer[dtype]
    var _data_info: dataInfo[dtype]
    alias simd_width: Int = simdwidthof[dtype]()

    # default constructor
    fn __init__(inout self, *dims:Int):
        var weight:List[Int] = List[Int]()
        var dim: List[Int] = List[Int]()

        for i in range(dims.__len__()):
            dim.append(dims[i])
            var temp: Int = 1
            for j in range(i+1):
                temp *= dims[j]
            weight.append(temp)
        
        self._data_info = dataInfo[dtype](rank=dims.__len__(), dims=dim, weights=weight, first_index=0)
        self._arr =  DTypePointer[dtype].alloc(self._data_info.num_elements)
        memset_zero(self._arr, self._data_info.num_elements)

    # constructor when rank, dims, weights, first_index(offset) are known
    fn __init__(inout self, rank:Int, dims:List[Int], weights:List[Int], first_index:Int):
        self._data_info = dataInfo[dtype](rank=rank, dims=dims, weights=weights, first_index=first_index)
        self._arr =  DTypePointer[dtype].alloc(self._data_info.num_elements)
        memset_zero(self._arr, self._data_info.num_elements) 

    # constructor when data is known, but this currently leads to malloc as two struct instances cant share same DTypePointer
    # fn __init__(inout self, inout data:array[dtype], rank:Int, dims:List[Int], weights:List[Int], first_index:Int):
    #     self._data_info = dataInfo[dtype](rank=rank, dims=dims, weights=weights, first_index=first_index)
    #     self._arr = data._arr

    fn __init__(inout self, dims:VariadicList[Int], random:Bool=False):
        var weight:List[Int] = List[Int]()
        var dim: List[Int] = List[Int]()
        for i in range(dims.__len__()):
            dim.append(dims[i])
            var temp: Int = 1
            for j in range(i):
                temp *= dims[j]
            weight.append(temp)
            
        self._data_info = dataInfo[dtype](rank=dims.__len__(), dims=dim, weights=weight, first_index=0)
        self._arr =  DTypePointer[dtype].alloc(self._data_info.num_elements)
        memset_zero(self._arr, self._data_info.num_elements)
        if random:
            rand[dtype](self._arr, self._data_info.num_elements)

    fn __init__(inout self, dims:List[Int], random:Bool=False):
        var weight:List[Int] = List[Int]()
        for i in range(dims.__len__()):
            var temp: Int = 1
            for j in range(i):
                temp *= dims[j]
            weight.append(temp)

        self._data_info = dataInfo[dtype](rank=dims.__len__(), dims=dims, weights=weight, first_index=0)
        self._arr =  DTypePointer[dtype].alloc(self._data_info.num_elements)
        memset_zero(self._arr, self._data_info.num_elements)
        if random:
            rand[dtype](self._arr, self._data_info.num_elements)

    fn __copyinit__(inout self, new: Self):
        self._data_info = new._data_info
        self._arr = DTypePointer[dtype].alloc(self._data_info.num_elements)
        for i in range(self._data_info.num_elements):
            self._arr[i] = new._arr[i]

    fn __moveinit__(inout self, owned existing: Self):
        self._data_info = existing._data_info
        self._arr = existing._arr
        existing._arr = DTypePointer[dtype]()

    fn _adjust_slice_(inout self, inout span:Slice, dim:Int):
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

    fn __setitem__(inout self, idx:Int, val:SIMD[dtype,1]):
        self._arr.__setitem__(idx, val)

    fn __setitem__(inout self, indices: List[Int], val:SIMD[dtype,1]):
        var index:Int = _get_index(indices, self._data_info.weights)
        self._arr[index] = val
    
    fn __setitem__(inout self, indices: VariadicList[Int], val:SIMD[dtype,1]):
        var index:Int = _get_index(indices, self._data_info.weights)
        self._arr[index] = val

    fn __getitem__(inout self, idx:Int) -> SIMD[dtype, 1]:
        return self._arr.__getitem__(idx)

    fn __getitem__(inout self, *indices: Int) raises -> SIMD[dtype,1]:
        if indices.__len__() != self._data_info.rank:
            raise Error("Error: Length of Indices do not match the shape")

        for i in range(indices.__len__()):
            if indices[i] >= self._data_info.dims[i]:
                    raise Error("Error: Elements of Indices exceed the shape values")

        var index:Int = self._data_info.first_index + _get_index(indices, self._data_info.weights)
        return self._arr[index]

    # same as above, but explicit VariadicList
    fn __getitem__(inout self, indices: VariadicList[Int]) raises -> SIMD[dtype,1]:

        if indices.__len__() != self._data_info.rank:
            raise Error("Error: Length of Indices do not match the shape")

        for i in range(indices.__len__()):
            if indices[i] >= self._data_info.dims[i]:
                raise Error("Error: Elements of Indices exceed the shape values")

        var index:Int = self._data_info.first_index + _get_index(indices, self._data_info.weights)
        return self._arr[index]

    fn __getitem__(inout self, owned *slices: Slice) raises -> Self:
        var n_slices:Int = slices.__len__()
        if n_slices > self._data_info.rank or n_slices < self._data_info.rank:
            print("Error: No of slices do not match shape")

        var nrank: Int = 0
        var spec:List[Int] = List[Int]()
        for i in range(slices.__len__()):
            self._adjust_slice_(slices[i], self._data_info.dims[i])
            spec.append(slices[i].unsafe_indices())
            if slices[i].unsafe_indices() != 1:
                nrank += 1 

        var dims: List[Int] = List[Int]()
        var weights: List[Int] = List[Int]()

        var j:Int = 0
        for _ in range(nrank):
            while spec[j] == 1:
                j+=1
            if j >= self._data_info.rank:
                break
            dims.append(slices[j].unsafe_indices())
            weights.append(self._data_info.weights[j] * slices[j].step)
            j+=1
        
        var offset_index: Int = 0
        for i in range(slices.__len__()):
            var temp: Int = 1
            for j in range(i):
                temp *= self._data_info.dims[j]
            offset_index += (slices[i].start * temp)

        var narr = self
        narr._data_info.rank = nrank
        narr._data_info.dims = dims
        narr._data_info.weights = weights
        narr._data_info.first_index = offset_index
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
        return self._data_info.num_elements

    fn __int__(self) -> Int:
        return self._data_info.num_elements

    fn __pos__(self) -> Self:
        return self * 1.0

    fn __neg__(self) -> Self:
        return self * -1.0

    fn __str__(self) -> String:
        return self._array_to_string(0, self._data_info.first_index)

    fn _array_to_string(self, dimension:Int, offset:Int) -> String :
        # if dimension == len(self._data_info.dims) - 1:
        #     return "[" + ", ".join(str(self._arr[offset + i * self._data_info.weights[dimension]]) for i in range(self._data_info.dims[dimension])) + "]"
        # else:
        #     return "[" + ", ".join(self._array_to_string(dimension + 1, offset + i * self._data_info.weights[dimension]) for i in range(self._data_info.dims[dimension])) + "]"

        if dimension == len(self._data_info.dims) - 1:
            var result: String = str("[\t")
            for i in range(self._data_info.dims[dimension]):
                if i > 0:
                    result = result + "\t"
                result = result + self._arr[offset + i * self._data_info.weights[dimension]].__str__()
                # result = result + "0"
            result = result + "\t]"
            return result
        else:
            var result: String = str("[")
            for i in range(self._data_info.dims[dimension]):
                result = result + self._array_to_string(dimension + 1, offset + i * self._data_info.weights[dimension])
                if i < (self._data_info.dims[dimension]-1):
                    result += "\n"
            result = result + "]"
            return result

    fn __eq__(self, other: Self) -> Bool:
        return self._arr == other._arr

    # # ARITHMETICS
    fn _elementwise_scalar_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, s: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = self
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._arr.store[width=simd_width](idx, func[dtype, simd_width](SIMD[dtype, simd_width](s), self._arr.load[width=simd_width](idx)))
        vectorize[elemwise_vectorize, simd_width](self._data_info.num_elements)
        return new_array

    fn _elementwise_array_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, other: Self) -> Self:
        alias simd_width = simdwidthof[dtype]()
        var new_vec = self
        @parameter
        fn vectorized_arithmetic[simd_width:Int](index: Int) -> None:
            new_vec._arr.store[width=simd_width](index, func[dtype, simd_width](self._arr.load[width=simd_width](index), other._arr.load[width=simd_width](index)))

        vectorize[vectorized_arithmetic, simd_width](self._data_info.num_elements)
        return new_vec

    fn __add__(inout self, other:Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__add__](other)

    fn __add__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[SIMD.__add__](other)

    fn __radd__(inout self, s: Scalar[dtype])->Self:
        return self + s

    fn __iadd__(inout self, s: Scalar[dtype]):
        self = self + s

    fn __sub__(self, other:Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__sub__](other)

    fn __sub__(self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[SIMD.__sub__](other)

    fn __rsub__(self, s: Scalar[dtype])->Self:
        return -(self - s)

    fn __isub__(inout self, s: Scalar[dtype]):
        self = self - s

    fn __mul__(self, s: Scalar[dtype])->Self:
        return self._elementwise_scalar_arithmetic[SIMD.__mul__](s)

    fn __mul__(self, other: Self)->Self:
        return self._elementwise_array_arithmetic[SIMD.__mul__](other)

    fn __rmul__(self, s: Scalar[dtype])->Self:
        return self*s

    fn __imul__(inout self, s: Scalar[dtype]):
        self = self*s

    fn _reduce_sum(self) -> Scalar[dtype]:
        var reduced = Scalar[dtype](0.0)
        alias simd_width: Int = simdwidthof[dtype]()
        @parameter
        fn vectorize_reduce[simd_width: Int](idx: Int) -> None:
            reduced[0] += self._arr.load[width = simd_width](idx).reduce_add()
        vectorize[vectorize_reduce, simd_width](self._data_info.num_elements)
        return reduced

    fn __matmul__(inout self, other:Self) -> Scalar[dtype]:
        return self._elementwise_array_arithmetic[SIMD.__mul__](other)._reduce_sum()

    fn __pow__(self, p: Int)->Self:
        return self._elementwise_pow(p)

    fn __ipow__(inout self, p: Int):
        self = self.__pow__(p)

    fn _elementwise_pow(self, p: Int) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_vec = self
        @parameter
        fn tensor_scalar_vectorize[simd_width: Int](idx: Int) -> None:
            new_vec._arr.store[width=simd_width](idx, pow(self._arr.load[width=simd_width](idx), p))
        vectorize[tensor_scalar_vectorize, simd_width](self._data_info.num_elements)
        return new_vec

    # ! truediv is multiplying instead of dividing right now lol, I don't know why.
    # fn __truediv__(inout self, s: Scalar[dtype]) -> Self:
    #     return self._elementwise_scalar_arithmetic[SIMD.__truediv__](s)

    # fn __truediv__(inout self, other:Self) raises -> Self:
    #     if self._num_elements!=other._num_elements:
    #         with assert_raises():
    #             raise Error("No of elements do not match")
        
    #     return self._elementwise_array_arithmetic[SIMD.__truediv__](other)

    # fn __itruediv__(inout self, s: Scalar[dtype]):
    #     self = self.__truediv__(s)

    # fn __itruediv__(inout self, other:Self):
    #     self = self.__truediv__(other)

    # fn __rtruediv__(inout self, s: Scalar[dtype]) -> Self:
    #     return self.__truediv__(s)