from random import rand
from builtin.math import pow
from algorithm import parallelize, vectorize

from .ndarray_utils import _get_index, _traverse_iterative
from .ndarray import NDArrayShape, NDArrayStrides

# Keep an option for user to change this
alias ALLOWED = 10

# ===----------------------------------------------------------------------===#
# NDArrayView
# ===----------------------------------------------------------------------===#

# TODO: Remove __getitem__ to prevent creating view of views, or create a parameter that keeps tracks of all views created from a single parent NDArray.


# * ROW MAJOR INDEXING
struct NDArrayView[dtype: DType = DType.float32](Stringable):
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

    # Commented this out, having .free() method on both NDArray, NDArrayView will cause malloc.
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

    fn __getitem__(self, owned index: Int) raises -> SIMD[dtype, 1]:
        """
        Example:
            `arr[15]` returns the 15th item of the array's data buffer.
        """
        if index >= self.ndshape._size:
            raise Error("Invalid index: index out of bound")

        index = index * 3
        return self.data.__getitem__(index)
        # return self.data[index]

    fn __getitem__(self, *indices: Int) raises -> SIMD[dtype, 1]:
        """
        Example:
            `arr[1,2]` returns the item of 1st row and 2nd column of the array.
        """
        if indices.__len__() != self.ndim:
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

        # * creates a view of a view, Inception max
        var narr = Self(
            self.data,
            ndims,
            noffset,
            nnum_elements,
            nshape,
            nstrides,
            ncoefficients,
        )

        # var narr = Self(
        #     ndims, noffset, nnum_elements, nshape, nstrides, ncoefficients
        # )

        # # Starting index to traverse the new array
        # var index = StaticIntTuple[ALLOWED]()
        # for i in range(ndims):
        #     index[i] = 0

        # _traverse_iterative[dtype](
        #     self,
        #     narr,
        #     ndims,
        #     nshape,
        #     ncoefficients,
        #     nstrides,
        #     noffset,
        #     index,
        #     0,
        # )
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

        # just returns a view for now. Think whether to return a copy instead
        var narr = Self(
            self.data,
            ndims,
            noffset,
            nnum_elements,
            nshape,
            nstrides,
            ncoefficients,
        )

        # var narr = Self(
        #     ndims, noffset, nnum_elements, nshape, nstrides, ncoefficients
        # )

        # var index = StaticIntTuple[ALLOWED]()
        # for i in range(ndims):
        #     index[i] = 0

        # _traverse_iterative[dtype](
        #     self,
        #     narr,
        #     ndims,
        #     nshape,
        #     ncoefficients,
        #     nstrides,
        #     noffset,
        #     index,
        #     0,
        # )

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
                slice_list.append(Slice(int, int + 1))
                print(int, "=", Slice(int, int + 1))
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

    # mayve I should return a resulting abs array of the view or abs view of the view. this is getting out of hands.
    fn __abs__(self):
        alias nelts = simdwidthof[dtype]()

        @parameter
        fn vectorized_abs[simd_width: Int](idx: Int) -> None:
            self.data.store[width=simd_width](
                idx, abs(self.data.load[width=simd_width](idx))
            )

        vectorize[vectorized_abs, nelts](self.ndshape._size)

    # all elements raised to some integer power
    fn __pow__(self, p: Int) -> Self:
        return self._elementwise_pow(p)

    # same dilemma as __abs__
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

    # # Technically it only changes the ArrayDescriptor and not the fundamental data
    # fn reshape(inout self, *Shape: Int) raises:
    #     """
    #     Reshapes the NDArray to given Shape.

    #     Args:
    #         Shape: Variadic list of shape.
    #     """
    #     var num_elements_new: Int = 1
    #     var ndim_new: Int = 0
    #     for i in Shape:
    #         num_elements_new *= i
    #         ndim_new += 1

    #     if self.info._size != num_elements_new:
    #         raise Error("Cannot reshape: Number of elements do not match.")

    #     self.info._ndim = ndim_new
    #     var shape_new: List[Int] = List[Int]()
    #     var strides_new: List[Int] = List[Int]()

    #     for i in range(ndim_new):
    #         shape_new.append(Shape[i])
    #         var temp: Int = 1
    #         for j in range(i + 1, ndim_new):  # temp
    #             temp *= Shape[j]
    #         strides_new.append(temp)

    #     self.info._shape = shape_new
    #     self.info._strides = strides_new
    # self.shape.shape = shape_new # current ndarray doesn't have NDArray shape field

    fn unsafe_ptr(self) -> DTypePointer[dtype, 0]:
        return self.data
