from random import rand
from testing import assert_raises

from builtin.math import pow

struct array[dtype: DType = DType.float32](
    # Intable, Stringable
    ):
    var _arr: DTypePointer[dtype]
    var _shape: VariadicList[Int]
    var _num_elements: Int
    alias simd_width: Int = simdwidthof[dtype]()

    # Constructors
    # * I need to figure out how to create data based on user given size
    fn __init__(inout self, *dims:Int):
        self._shape = dims
        self._num_elements = 1
        for i in range(self._shape.__len__()):
            self._num_elements *= self._shape[i]

        self._arr =  DTypePointer[dtype].alloc(self._num_elements)
        memset_zero(self._arr, self._num_elements)

    fn __init__(inout self, dims:VariadicList[Int], random:Bool=False):
        self._shape = dims
        self._num_elements = 1
        for i in range(self._shape.__len__()):
            self._num_elements *= self._shape[i]

        self._arr =  DTypePointer[dtype].alloc(self._num_elements)
        memset_zero(self._arr, self._num_elements)
        if random:
            rand[dtype](self._arr, self._num_elements)

    fn __copyinit__(inout self, new: Self):
        self._shape = new._shape
        self._num_elements = new._num_elements
        self._arr = new._arr

    fn __moveinit__(inout self, owned existing: Self):
        self._shape = existing._shape
        self._num_elements = existing._num_elements
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

    fn __getitem__(inout self, *indices: Int) raises -> SIMD[dtype,1]:

        if indices.__len__() != self._shape.__len__():
            raise Error("Error: Length of Indices do not match the shape")

        for i in range(indices.__len__()):
            if indices[i] >= self._shape[i]:
                    raise Error("Error: Elements of Indices exceed the shape values")

        var index: Int = 0
        for i in range(indices.__len__()):
            var temp: Int = 1
            for j in range(i+1, self._shape.__len__()):
                temp *= self._shape[j]
            index += (indices[i] * temp)
        return self._arr[index]
    
    fn __getitem__(inout self, indices: VariadicList[Int]) raises -> SIMD[dtype,1]:

        if indices.__len__() != self._shape.__len__():
            raise Error("Error: Length of Indices do not match the shape")

        for i in range(indices.__len__()):
            if indices[i] >= self._shape[i]:
                    raise Error("Error: Elements of Indices exceed the shape values")

        var index: Int = 0
        for i in range(indices.__len__()):
            var temp: Int = 1
            for j in range(i+1, self._shape.__len__()):
                temp *= self._shape[j]
            index += (indices[i] * temp)
        return self._arr[index]

    # I have to implement some kind of Index struct so that we don't have to write VariadicList everytime
    fn __setitem__(inout self, indices:VariadicList[Int], value:Scalar[dtype]) raises:

        if indices.__len__() != self._shape.__len__():
            with assert_raises():
                raise "Error: Indices do not match the shape"

        for i in range(indices.__len__()):
            if indices[i] >= self._shape[i]:
                with assert_raises():
                    raise "Error: Indices do not match the shape"

        var index: Int = 0
        for i in range(indices.__len__()):
            var temp: Int = 1
            for j in range(i+1, self._shape.__len__()):
                temp *= self._shape[j]
            index += (indices[i] * temp) 

        self._arr[index] = value

    fn __del__(owned self):
        self._arr.free()

    fn __len__(inout self) -> Int:
        return self._num_elements

    fn __int__(self) -> Int:
        return self._num_elements

    fn __pos__(self) -> Self:
        return self * 1.0

    fn __neg__(self) -> Self:
        return self * -1.0

    # fn __getitem__(inout self, owned row_slice: Slice, owned col_slice: Slice) -> Self:
    #     self._adjust_slice_(row_slice, self._row)
    #     self._adjust_slice_(col_slice, self._col)

    #     var src_ptr = self._arr
    #     var sliced_arr = Self(row_slice.__len__(), col_slice.__len__())

    #     @parameter
    #     fn slice_column(idx_rows: Int):
    #         src_ptr = self._arr.offset(row_slice[idx_rows] * self._col + col_slice[0])

    #         @parameter
    #         fn slice_row[simd_width: Int](idx: Int) -> None:
    #             sliced_arr._arr.store[width=simd_width](
    #                 idx + idx_rows * col_slice.__len__(),
    #                 strided_load[dtype, simd_width](src_ptr, col_slice.step),
    #             )
    #             src_ptr = src_ptr.offset(simd_width * col_slice.step)

    #         vectorize[slice_row, self.simd_width](col_slice.__len__())

    #     parallelize[slice_column](row_slice.__len__(), row_slice.__len__())
    #     return sliced_arr

    # fn __getitem__(inout self, owned row_slice: Slice, col: Int) -> Self:
    #     return self.__getitem__(row_slice, Slice(col, col+1))

    # fn __getitem__(inout self, row: Int, owned col_slice: Slice) -> Self:
    #     return self.__getitem__(Slice(row, row+1), col_slice)

    # fn print(self, prec: Int = 4) -> None:
    #     var rank: Int = 2
    #     var dim0: Int = 0
    #     var dim1: Int = 0
    #     var val: Scalar[dtype] = 0.0
    #     if self._row == 1:
    #         rank = 1
    #         dim0 = 1
    #         dim1 = self._col
    #     else:
    #         dim0 = self._row
    #         dim1 = self._col
    #     if dim0 > 0 and dim1 > 0:
    #         for j in range(dim0):
    #             if rank > 1:
    #                 if j == 0:
    #                     print("  [", end=" ")
    #                 else:
    #                     print("\n   ", end=" ")
    #             print("[", end=" ")
    #             for k in range(dim1):
    #                 if rank == 1:
    #                     val = self._arr.load[width=1](k)
    #                 if rank == 2:
    #                     val = self[j, k]
    #                 var int_str: String
    #                 if val > 0 or val == 0:
    #                     int_str = String(trunc(val).cast[DType.int32]())
    #                 else:
    #                     int_str = "-" + String(trunc(val).cast[DType.int32]())
    #                     val = -val
    #                 var float_str: String
    #                 float_str = String(mod(val, 1))
    #                 var s = int_str + "." + float_str[2 : prec + 2]
    #                 if k == 0:
    #                     print(s, end=" ")
    #                 else:
    #                     print("  ", s, end=" ")
    #             print("]", end=" ")
    #         if rank > 1:
    #             print("]", end =" ")
    #         print()
    #         if rank > 2:
    #             print("]")
    #     print("  Matrix:", self._row, "x", self._col, ",", "DType:", dtype.__str__())
    #     print()

    # fn __str__(self) -> String:
    #     var prec: Int = 4
    #     var str_repr: String = "["
    #     var rank: Int = 2
    #     var dim0: Int = 0
    #     var dim1: Int = 0
    #     var val: Scalar[dtype] = 0.0
    #     if self._row == 1:
    #         rank = 1
    #         dim0 = 1
    #         dim1 = self._col
    #     else:
    #         dim0 = self._row
    #         dim1 = self._col
    #     if dim0 > 0 and dim1 > 0:
    #         for j in range(dim0):
    #             if rank > 1:
    #                 if j == 0:
    #                     str_repr += "["
    #                 else:
    #                     str_repr += "\n   "
    #                 str_repr += "["
    #             str_repr += "["
    #             for k in range(dim1):
    #                 if rank == 1:
    #                     val = self._arr.load[width=1](k)
    #                 if rank == 2:
    #                     val = self[j, k]
    #                 var int_str: String
    #                 if val > 0 or val == 0:
    #                     int_str = String(trunc(val).cast[DType.int32]())
    #                 else:
    #                     int_str = "-" + String(trunc(val).cast[DType.int32]())
    #                     val = -val
    #                 var float_str: String
    #                 float_str = String(mod(val, 1))
    #                 var s = int_str + "." + float_str[2 : prec + 2]
    #                 if k == 0:
    #                     str_repr += s
    #                 else:
    #                     str_repr += "  " + s
    #             str_repr += "]"
    #         if rank > 1:
    #             str_repr += "]"
    #         str_repr += "\n"
    #         if rank > 2:
    #             str_repr += "]"
    #     str_repr += "\n"
    #     str_repr += "  Array: " + String(self._row) + "x" + String(self._col) + ", DType: " + dtype.__str__()
    #     str_repr += "\n"
    #     return str_repr

    fn __eq__(self, other: Self) -> Bool:
        return self._arr == other._arr

    # # ARITHMETICS
    fn _elementwise_scalar_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, s: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(self._shape)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._arr.store[width=simd_width](idx, func[dtype, simd_width](SIMD[dtype, simd_width](s), self._arr.load[width=simd_width](idx)))
        vectorize[elemwise_vectorize, simd_width](self._num_elements)
        return new_array

    fn _elementwise_array_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, other: Self) -> Self:
        alias simd_width = simdwidthof[dtype]()
        var new_vec = Self()
        @parameter
        fn vectorized_arithmetic[simd_width:Int](index: Int) -> None:
            new_vec._arr.store[width=simd_width](index, func[dtype, simd_width](self._arr.load[width=simd_width](index), other._arr.load[width=simd_width](index)))

        vectorize[vectorized_arithmetic, simd_width](self._num_elements)
        return new_vec

    fn __add__(inout self, other:Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__add__](other)

    fn __add__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[SIMD.__add__](other)

    fn __radd__(inout self, s: Scalar[dtype])->Self:
        return self + s

    fn __iadd__(inout self, s: Scalar[dtype]):
        self = self + s

    fn _sub__(inout self, other:Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__sub__](other)

    fn __sub__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[SIMD.__sub__](other)

    # # TODO: I don't know why I am getting error here, so do this later.
    # fn __rsub__(inout self, s: Scalar[dtype])->Self:
    #     return -(self - s)

    # fn __isub__(inout self, s: Scalar[dtype]):
    #     self = self - s

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
        vectorize[vectorize_reduce, simd_width](self._num_elements)
        return reduced

    fn __matmul__(inout self, other:Self) -> Scalar[dtype]:
        return self._elementwise_array_arithmetic[SIMD.__mul__](other)._reduce_sum()

    fn __pow__(self, p: Int)->Self:
        return self._elementwise_pow(p)

    fn __ipow__(inout self, p: Int):
        self = self.__pow__(p)

    fn _elementwise_pow(self, p: Int) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_vec = Self(self._shape)
        @parameter
        fn tensor_scalar_vectorize[simd_width: Int](idx: Int) -> None:
            new_vec._arr.store[width=simd_width](idx, pow(self._arr.load[width=simd_width](idx), p))
        vectorize[tensor_scalar_vectorize, simd_width](self._num_elements)
        return new_vec

    # ! truediv is multiplying instead of dividing right now, I don't know why.
    fn __truediv__(inout self, s: Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[SIMD.__truediv__](s)

    fn __truediv__(inout self, other:Self) raises -> Self:
        if self._num_elements!=other._num_elements:
            with assert_raises():
                raise Error("No of elements do not match")
        
        return self._elementwise_array_arithmetic[SIMD.__truediv__](other)

    # fn __itruediv__(inout self, s: Scalar[dtype]):
    #     self = self.__truediv__(s)

    # fn __itruediv__(inout self, other:Self):
    #     self = self.__truediv__(other)

    # fn __rtruediv__(inout self, s: Scalar[dtype]) -> Self:
    #     return self.__truediv__(s)