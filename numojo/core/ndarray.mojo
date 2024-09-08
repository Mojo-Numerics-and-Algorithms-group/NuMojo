"""
Implements N-Dimensional Array
"""
# ===----------------------------------------------------------------------=== #
# Implements ROW MAJOR N-DIMENSIONAL ARRAYS
# Last updated: 2024-07-14
# ===----------------------------------------------------------------------=== #


"""
# TODO
1) Generalize mdot, rdot to take any IxJx...xKxL and LxMx...xNxP matrix and matmul it into IxJx..xKxMx...xNxP array.
2) Add vectorization for _get_index
3) Write more explanatory Error("") statements
4) Create NDArrayView and remove coefficients. 
"""

from builtin.type_aliases import AnyLifetime
from random import rand, random_si64, random_float64
from builtin.math import pow
from builtin.bool import all as allb
from builtin.bool import any as anyb
from algorithm import parallelize, vectorize
from python import Python

import . _array_funcs as _af
from ..math.statistics.stats import mean, prod, sum
from ..math.statistics.cumulative_reduce import (
    cumsum,
    cumprod,
    cummean,
    maxT,
    minT,
)
import . sort as sort
import .. math as math
from ..traits import Backend
from ..math.check import any, all
from ..math.arithmetic import abs
from .ndarray_utils import (
    _get_index,
    _traverse_iterative,
    to_numpy,
    bool_to_numeric,
    fill_pointer,
)
from ..math.math_funcs import Vectorized
from .utility_funcs import is_inttype
from ..math.linalg.matmul import matmul_parallelized
from .array_manipulation_routines import reshape


@register_passable("trivial")
struct NDArrayShape[dtype: DType = DType.int32](Stringable):
    """Implements the NDArrayShape."""

    # Fields
    var ndsize: Int
    """Total no of elements in the corresponding array."""
    var ndshape: DTypePointer[dtype]
    """Shape of the corresponding array."""
    var ndlen: Int
    """Length of ndshape."""

    @always_inline("nodebug")
    fn __init__(inout self, *shape: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions.

        Args:
            shape: Variable number of integers representing the shape dimensions.
        """
        self.ndsize = 1
        self.ndlen = len(shape)
        self.ndshape = DTypePointer[dtype].alloc(len(shape))
        memset_zero(self.ndshape, len(shape))
        for i in range(len(shape)):
            self.ndshape[i] = shape[i]
            self.ndsize *= shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, *shape: Int, size: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions and a specified size.

        Args:
            shape: Variable number of integers representing the shape dimensions.
            size: The total number of elements in the array.
        """
        self.ndsize = size
        self.ndlen = len(shape)
        self.ndshape = DTypePointer[dtype].alloc(len(shape))
        memset_zero(self.ndshape, len(shape))
        var count: Int = 1
        for i in range(len(shape)):
            self.ndshape[i] = shape[i]
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
        self.ndsize = 1
        self.ndlen = len(shape)
        self.ndshape = DTypePointer[dtype].alloc(len(shape))
        memset_zero(self.ndshape, len(shape))
        for i in range(len(shape)):
            self.ndshape[i] = shape[i]
            self.ndsize *= shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, shape: List[Int], size: Int) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions and a specified size.

        Args:
            shape: A list of integers representing the shape dimensions.
            size: The specified size of the NDArrayShape.
        """
        self.ndsize = (
            size  # maybe I should add a check here to make sure it matches
        )
        self.ndlen = len(shape)
        self.ndshape = DTypePointer[dtype].alloc(len(shape))
        memset_zero(self.ndshape, len(shape))
        var count: Int = 1
        for i in range(len(shape)):
            self.ndshape[i] = shape[i]
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
        self.ndsize = 1
        self.ndlen = len(shape)
        self.ndshape = DTypePointer[dtype].alloc(len(shape))
        memset_zero(self.ndshape, len(shape))
        for i in range(len(shape)):
            self.ndshape[i] = shape[i]
            self.ndsize *= shape[i]

    @always_inline("nodebug")
    fn __init__(inout self, shape: VariadicList[Int], size: Int) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions and a specified size.

        Args:
            shape: A list of integers representing the shape dimensions.
            size: The specified size of the NDArrayShape.
        """
        self.ndsize = (
            size  # maybe I should add a check here to make sure it matches
        )
        self.ndlen = len(shape)
        self.ndshape = DTypePointer[dtype].alloc(len(shape))
        memset_zero(self.ndshape, len(shape))
        var count: Int = 1
        for i in range(len(shape)):
            self.ndshape[i] = shape[i]
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
        self.ndsize = shape.ndsize
        self.ndlen = shape.ndlen
        self.ndshape = DTypePointer[dtype].alloc(shape.ndlen)
        memset_zero(self.ndshape, shape.ndlen)
        for i in range(shape.ndlen):
            self.ndshape[i] = shape[i]

    fn __copy__(inout self, other: Self):
        """
        Copy from other into self.
        """
        self.ndsize = other.ndsize
        self.ndlen = other.ndlen
        self.ndshape = DTypePointer[dtype].alloc(other.ndlen)
        memcpy(self.ndshape, other.ndshape, other.ndlen)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        """
        Get shape at specified index.
        """
        if index >= self.ndlen:
            raise Error("Index out of bound")
        if index >= 0:
            return self.ndshape[index].__int__()
        else:
            return self.ndshape[self.ndlen + index].__int__()

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, val: Int) raises:
        """
        Set shape at specified index.
        """
        if index >= self.ndlen:
            raise Error("Index out of bound")
        if index >= 0:
            self.ndshape[index] = val
        else:
            self.ndshape[self.ndlen + index] = val

    @always_inline("nodebug")
    fn size(self) -> Int:
        """
        Get Size of array described by arrayshape.
        """
        return self.ndsize

    @always_inline("nodebug")
    fn len(self) -> Int:
        """
        Get number of dimensions of the array described by arrayshape.
        """
        return self.ndlen

    @always_inline("nodebug")
    fn __str__(self: Self) -> String:
        """
        Return a string of the shape of the array described by arrayshape.

        """
        var result: String = "Shape: ["
        for i in range(self.ndlen):
            if i == self.ndlen - 1:
                result += self.ndshape[i].__str__()
            else:
                result += self.ndshape[i].__str__() + ", "
        return result + "]"

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Bool:
        """
        Check if two arrayshapes have identical dimensions.
        """
        for i in range(self.ndlen):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> Bool:
        """
        Check if two arrayshapes don't have identical dimensions.
        """
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, val: Int) raises -> Bool:
        """
        Check if any of the dimensions are equal to a value.
        """
        for i in range(self.ndlen):
            if self[i] == val:
                return True
        return False

    # can be used for vectorized index calculation
    @always_inline("nodebug")
    fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
        """
        SIMD load dimensional information.
        """
        # if index >= self.ndlen:
        # raise Error("Index out of bound")
        return self.ndshape.load[width=width](index)

    # can be used for vectorized index retrieval
    @always_inline("nodebug")
    fn store[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[dtype, width]) raises:
        """
        SIMD store dimensional information.
        """
        # if index >= self.ndlen:
        #     raise Error("Index out of bound")
        self.ndshape.store[width=width](index, val)

    @always_inline("nodebug")
    fn load_int(self, index: Int) -> Int:
        """
        SIMD load dimensional information.
        """
        return self.ndshape.load[width=1](index).__int__()

    @always_inline("nodebug")
    fn store_int(inout self, index: Int, val: Int):
        """
        SIMD store dimensional information.
        """
        self.ndshape.store[width=1](index, val)


@register_passable("trivial")
struct NDArrayStride[dtype: DType = DType.int32](Stringable):
    """Implements the NDArrayStride."""

    # Fields
    var ndoffset: Int
    var ndstride: DTypePointer[dtype]
    var ndlen: Int

    @always_inline("nodebug")
    fn __init__(
        inout self, *stride: Int, offset: Int = 0
    ):  # maybe we should add checks for offset?
        self.ndoffset = offset
        self.ndlen = stride.__len__()
        self.ndstride = DTypePointer[dtype].alloc(stride.__len__())
        for i in range(stride.__len__()):
            self.ndstride[i] = stride[i]

    @always_inline("nodebug")
    fn __init__(inout self, stride: List[Int], offset: Int = 0):
        self.ndoffset = offset
        self.ndlen = stride.__len__()
        self.ndstride = DTypePointer[dtype].alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        for i in range(self.ndlen):
            self.ndstride[i] = stride[i]

    @always_inline("nodebug")
    fn __init__(inout self, stride: VariadicList[Int], offset: Int = 0):
        self.ndoffset = offset
        self.ndlen = stride.__len__()
        self.ndstride = DTypePointer[dtype].alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        for i in range(self.ndlen):
            self.ndstride[i] = stride[i]

    @always_inline("nodebug")
    fn __init__(inout self, stride: NDArrayStride):
        self.ndoffset = stride.ndoffset
        self.ndlen = stride.ndlen
        self.ndstride = DTypePointer[dtype].alloc(stride.ndlen)
        for i in range(self.ndlen):
            self.ndstride[i] = stride.ndstride[i]

    @always_inline("nodebug")
    fn __init__(
        inout self, stride: NDArrayStride, offset: Int = 0
    ):  # separated two methods to remove if condition
        self.ndoffset = offset
        self.ndlen = stride.ndlen
        self.ndstride = DTypePointer[dtype].alloc(stride.ndlen)
        for i in range(self.ndlen):
            self.ndstride[i] = stride.ndstride[i]

    @always_inline("nodebug")
    fn __init__(
        inout self, *shape: Int, offset: Int = 0, order: String = "C"
    ) raises:
        self.ndoffset = offset
        self.ndlen = shape.__len__()
        self.ndstride = DTypePointer[dtype].alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        if order == "C":
            for i in range(self.ndlen):
                var temp: Int = 1
                for j in range(i + 1, self.ndlen):
                    temp = temp * shape[j]
                self.ndstride[i] = temp
        elif order == "F":
            self.ndstride[0] = 1
            for i in range(0, self.ndlen - 1):
                self.ndstride[i + 1] = self.ndstride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        inout self, shape: List[Int], offset: Int = 0, order: String = "C"
    ) raises:
        self.ndoffset = offset
        self.ndlen = shape.__len__()
        self.ndstride = DTypePointer[dtype].alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        if order == "C":
            for i in range(self.ndlen):
                var temp: Int = 1
                for j in range(i + 1, self.ndlen):
                    temp = temp * shape[j]
                self.ndstride[i] = temp
        elif order == "F":
            self.ndstride[0] = 1
            for i in range(0, self.ndlen - 1):
                self.ndstride[i + 1] = self.ndstride[i] * shape[i]
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
        self.ndoffset = offset
        self.ndlen = shape.__len__()
        self.ndstride = DTypePointer[dtype].alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        if order == "C":
            for i in range(self.ndlen):
                var temp: Int = 1
                for j in range(i + 1, self.ndlen):
                    temp = temp * shape[j]
                self.ndstride[i] = temp
        elif order == "F":
            self.ndstride[0] = 1
            for i in range(0, self.ndlen - 1):
                self.ndstride[i + 1] = self.ndstride[i] * shape[i]
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
        self.ndoffset = offset
        self.ndlen = shape.ndlen
        self.ndstride = DTypePointer[dtype].alloc(shape.ndlen)
        memset_zero(self.ndstride, shape.ndlen)
        if order == "C":
            if shape.ndlen == 1:
                self.ndstride[0] = 1
            else:
                for i in range(shape.ndlen):
                    var temp: Int = 1
                    for j in range(i + 1, shape.ndlen):
                        temp = temp * shape[j]
                    self.ndstride[i] = temp
        elif order == "F":
            self.ndstride[0] = 1
            for i in range(0, self.ndlen - 1):
                self.ndstride[i + 1] = self.ndstride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    fn __copy__(inout self, other: Self):
        self.ndoffset = other.ndoffset
        self.ndlen = other.ndlen
        self.ndstride = DTypePointer[dtype].alloc(other.ndlen)
        memcpy(self.ndstride, other.ndstride, other.ndlen)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        if index >= self.ndlen:
            raise Error("Index out of bound")
        if index >= 0:
            return self.ndstride[index].__int__()
        else:
            return self.ndstride[self.ndlen + index].__int__()

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, val: Int) raises:
        if index >= self.ndlen:
            raise Error("Index out of bound")
        if index >= 0:
            self.ndstride[index] = val
        else:
            self.ndstride[self.ndlen + index] = val

    @always_inline("nodebug")
    fn len(self) -> Int:
        return self.ndlen

    @always_inline("nodebug")
    fn __str__(self: Self) -> String:
        var result: String = "Stride: ["
        for i in range(self.ndlen):
            if i == self.ndlen - 1:
                result += self.ndstride[i].__str__()
            else:
                result += self.ndstride[i].__str__() + ", "
        return result + "]"

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Bool:
        for i in range(self.ndlen):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, val: Int) raises -> Bool:
        for i in range(self.ndlen):
            if self[i] == val:
                return True
        return False

    @always_inline("nodebug")
    fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
        # if index >= self.ndlen:
        #     raise Error("Index out of bound")
        return self.ndstride.load[width=width](index)

    @always_inline("nodebug")
    fn store[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[dtype, width]) raises:
        # if index >= self.ndlen:
        #     raise Error("Index out of bound")
        self.ndstride.store[width=width](index, val)

    @always_inline("nodebug")
    fn load_unsafe[width: Int = 1](self, index: Int) -> Int:
        return self.ndstride.load[width=width](index).__int__()

    @always_inline("nodebug")
    fn store_unsafe[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[dtype, width]):
        self.ndstride.store[width=width](index, val)


@value
struct _NDArrayIter[
    is_mutable: Bool, //,
    lifetime: AnyLifetime[is_mutable].type,
    dtype: DType,
    forward: Bool = True,
]:
    """Iterator for NDArray.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        lifetime: The lifetime of the underlying NDArray data.
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var array: NDArray[dtype]
    var length: Int

    fn __init__(
        inout self,
        array: NDArray[dtype],
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.array = array

    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) raises -> NDArray[dtype]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.array.__getitem__(current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.array.__getitem__(current_index)

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index


# ===----------------------------------------------------------------------===#
# NDArray
# ===----------------------------------------------------------------------===#


struct NDArray[dtype: DType = DType.float64](
    Stringable, Representable, CollectionElement, Sized
):
    """The N-dimensional array (NDArray).

    Parameters:
        dtype: Type of item in NDArray. Default type is DType.float64.

    The array can be uniquely defined by the following:
        1. The data buffer of all items.
        2. The shape of the array.
        3. The stride in each dimension
        4. The number of dimensions
        5. The datatype of the elements
        6. The order of the array: Row vs Columns major
    """

    var data: DTypePointer[dtype]
    """Data buffer of the items in the NDArray."""
    var ndim: Int
    """Number of Dimensions."""
    var ndshape: NDArrayShape
    """Size and shape of NDArray."""
    var stride: NDArrayStride
    """Contains offset, strides."""
    var coefficient: NDArrayStride
    """Contains offset, coefficients for slicing."""
    var datatype: DType
    """The datatype of memory."""
    var order: String
    "Memory layout of array C (C order row major) or F (Fortran order col major)."

    alias width: Int = simdwidthof[dtype]()  #
    """Vector size of the data type."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    # default constructor
    @always_inline("nodebug")
    fn __init__(
        inout self,
        *shape: Int,
        fill: Optional[Scalar[dtype]] = None,
        order: String = "C",
    ) raises:
        """
        NDArray initialization for variadic shape with option to fill.

        Args:
            shape: Variadic shape.
            fill: Set all the values to this.
            order: Memory order C or F.

        Example:
            NDArray[DType.float16](VariadicList[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """

        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape.ndsize)
        self.datatype = dtype
        self.order = order
        if fill is not None:
            fill_pointer[dtype](self.data, self.ndshape.ndsize, fill.value()[])

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: List[Int],
        fill: Optional[Scalar[dtype]] = None,
        order: String = "C",
    ) raises:
        """
        NDArray initialization for variadic shape with option to fill.

        Args:
            shape: List of shape.
            fill: Set all the values to this.
            order: Memory order C or F.

        Example:
            NDArray[DType.float16](VariadicList[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """

        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape.ndsize)
        self.datatype = dtype
        self.order = order
        if fill is not None:
            fill_pointer[dtype](self.data, self.ndshape.ndsize, fill.value()[])

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: VariadicList[Int],
        fill: Optional[Scalar[dtype]] = None,
        order: String = "C",
    ) raises:
        """
        NDArray initialization for List of shape with option to fill.

        Args:
            shape: Variadic List of shape.
            fill: Set all the values to this.
            order: Memory order C or F.

        Example:
            NDArray[DType.float16](VariadicList[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """

        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape.ndsize)
        self.datatype = dtype
        self.order = order
        if fill is not None:
            fill_pointer[dtype](self.data, self.ndshape.ndsize, fill.value()[])

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: NDArrayShape,
        fill: Optional[Scalar[dtype]] = None,
        order: String = "C",
    ) raises:
        """
        NDArray initialization for NDArrayShape with option to fill.

        Args:
            shape: Variadic shape.
            fill: Set all the the values to this.
            order: Memory order C or F.

        Example:
            NDArray[DType.float16](VariadicList[Int](3, 2, 4), random=True)
            Returns an array with shape 3 x 2 x 4 and randomly values.
        """

        self.ndim = shape.ndlen
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, order=order)
        self.coefficient = NDArrayStride(shape, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape.ndsize)
        self.datatype = dtype
        self.order = order
        if fill is not None:
            fill_pointer[dtype](self.data, self.ndshape.ndsize, fill.value()[])

    fn __init__(
        inout self,
        data: List[SIMD[dtype, 1]],
        shape: List[Int],
        order: String = "C",
    ) raises:
        """
        NDArray initialization from list of data.

        Args:
            data: List of data.
            shape: List of shape.
            order: Memory order C or F.

        Example:
            `NDArray[DType.int8](List[Int8](1,2,3,4,5,6), shape=List[Int](2,3))`
            Returns an array with shape 3 x 2 with input values.
        """

        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape.ndsize)
        self.datatype = dtype
        self.order = order
        for i in range(self.ndshape.ndsize):
            self.data[i] = data[i]

    @always_inline("nodebug")
    fn __init__(
        inout self,
        *shape: Int,
        min: Scalar[dtype],
        max: Scalar[dtype],
        order: String = "C",
    ) raises:
        """
        NDArray initialization for variadic shape with random values between min and max.

        Args:
            shape: Variadic shape.
            min: Minimum value for the NDArray.
            max: Maximum value for the NDArray.
            order: Memory order C or F.

        Example:
            ```mojo
            import numojo as nm
            fn main() raises:
                var A = nm.NDArray[DType.float16](2, 2, min=0.0, max=10.0)
                print(A)
            ```
            A is an array with shape 2 x 2 and randomly values between 0 and 10.
            The output goes as follows.

            ```console
            [[	6.046875	6.98046875	]
             [	6.6484375	1.736328125	]]
            2-D array  Shape: [2, 2]  DType: float16
            ```
        """
        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.datatype = dtype
        self.order = order
        self.data = DTypePointer[dtype].alloc(self.ndshape.ndsize)
        if dtype.is_floating_point():
            for i in range(self.ndshape.ndsize):
                self.data.store(
                    i,
                    random_float64(
                        min.cast[DType.float64](), max.cast[DType.float64]()
                    ).cast[dtype](),
                )
        elif dtype.is_integral():
            for i in range(self.ndshape.ndsize):
                self.data.store(
                    i, random_si64(int(min), int(max)).cast[dtype]()
                )

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: List[Int],
        min: Scalar[dtype],
        max: Scalar[dtype],
        order: String = "C",
    ) raises:
        """
        NDArray initialization for list shape with random values between min and max.

        Args:
            shape: List of shape.
            min: Minimum value for the NDArray.
            max: Maximum value for the NDArray.
            order: Memory order C or F.

        Example:
            ```mojo
            import numojo as nm
            fn main() raises:
                var A = nm.NDArray[DType.float16](List[Int](2, 2), min=0.0, max=10.0)
                print(A)
            ```
            A is an array with shape 2 x 2 and randomly values between 0 and 10.
            The output goes as follows.

            ```console
            [[	6.046875	6.98046875	]
             [	6.6484375	1.736328125	]]
            2-D array  Shape: [2, 2]  DType: float16
            ```
        """
        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.datatype = dtype
        self.order = order
        self.data = DTypePointer[dtype].alloc(self.ndshape.ndsize)
        if dtype.is_floating_point():
            for i in range(self.ndshape.ndsize):
                self.data.store(
                    i,
                    random_float64(
                        min.cast[DType.float64](), max.cast[DType.float64]()
                    ).cast[dtype](),
                )
        elif dtype.is_integral():
            for i in range(self.ndshape.ndsize):
                self.data.store(
                    i, random_si64(int(min), int(max)).cast[dtype]()
                )

    fn __init__(
        inout self,
        text: String,
        order: String = "C",
    ) raises:
        """
        NDArray initialization from string representation of an ndarray.
        The shape can be inferred from the string representation.
        The literals will be casted to the dtype of the NDArray.

        Note:
        StringLiteral is also allowed as input as it is coerced to String type
        before it is passed into the function.

        Example:
        ```mojo
        import numojo as nm

        fn main() raises:
            var A = nm.NDArray[DType.int8]("[[[1,2],[3,4]],[[5,6],[7,8]]]")
            var B = nm.NDArray[DType.float16]("[[1,2,3,4],[5,6,7,8]]")
            var C = nm.NDArray[DType.float32]("[0.1, -2.3, 41.5, 19.29145, -199]")
            var D = nm.NDArray[DType.int32]("[0.1, -2.3, 41.5, 19.29145, -199]")

            print(A)
            print(B)
            print(C)
            print(D)
        ```

        The output goes as follows. Note that the numbers are automatically
        casted to the dtype of the NDArray.

        ```console
        [[[     1       2       ]
          [     3       4       ]]
         [[     5       6       ]
          [     7       8       ]]]
        3-D array  Shape: [2, 2, 2]  DType: int8

        [[      1.0     2.0     3.0     4.0     ]
         [      5.0     6.0     7.0     8.0     ]]
        2-D array  Shape: [2, 4]  DType: float16

        [       0.10000000149011612     2.2999999523162842      41.5    19.291450500488281      199.0   ]
        1-D array  Shape: [5]  DType: float32

        [       0       2       41      19      199     ]
        1-D array  Shape: [5]  DType: int32
        ```

        Args:
            text: String representation of an ndarray.
            order: Memory order C or F.
        """

        var data = List[Scalar[dtype]]()
        """Inferred data buffer of the array"""
        var shape = List[Int]()
        """Inferred shape of the array"""
        var bytes = text.as_bytes()
        var ndim = 0
        """Inferred number_as_str of dimensions."""
        var level = 0
        """Current level of the array."""
        var number_as_str: String = ""
        for i in range(len(bytes)):
            var b = bytes[i]
            if chr(int(b)) == "[":
                level += 1
                ndim = max(ndim, level)
                if len(shape) < ndim:
                    shape.append(0)
                shape[level - 1] = 0

            if isdigit(b) or chr(int(b)) == ".":
                number_as_str = number_as_str + chr(int(b))
            if (chr(int(b)) == ",") or (chr(int(b)) == "]"):
                if number_as_str != "":
                    var number = atof(number_as_str).cast[dtype]()
                    data.append(number)  # Add the number to the data buffer
                    number_as_str = ""  # Clean the number cache
                    shape[-1] = shape[-1] + 1
            if chr(int(b)) == "]":
                level = level - 1
                if level < 0:
                    raise ("Unmatched left and right brackets!")
                if level > 0:
                    shape[level - 1] = shape[level - 1] + 1
        self.__init__(data=data, shape=shape, order=order)

    # Why do  these last two constructors exist?
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
        """
        Extremely specific NDArray initializer.
        """
        self.ndim = ndim
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(stride=strides, offset=0)
        self.coefficient = NDArrayStride(stride=coefficient, offset=offset)
        self.datatype = dtype
        self.order = order
        self.data = DTypePointer[dtype].alloc(size)
        memset_zero(self.data, size)

    # creating NDArray from numpy array
    # TODO: Make it work for all data types apart from float64
    fn __init__(inout self, data: PythonObject, order: String = "C") raises:
        if dtype != DType.float64:
            raise Error("Only float64 is supported for now")
        var len = int(len(data.shape))
        var shape: List[Int] = List[Int]()
        for i in range(len):
            if int(data.shape[i]) == 1:
                continue
            shape.append(int(data.shape[i]))
        self.ndim = shape.__len__()
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(shape, offset=0, order=order)
        self.coefficient = NDArrayStride(shape, offset=0, order=order)
        self.data = DTypePointer[dtype].alloc(self.ndshape.ndsize)
        memset_zero(self.data, self.ndshape.ndsize)
        self.datatype = dtype
        self.order = order
        for i in range(self.ndshape.ndsize):
            self.data[i] = data.item(PythonObject(i)).to_float64()

        # var array: PythonObject
        # try:
        #     var np = Python.import_module("numpy")
        #     array = np.float32(data.copy())
        # except e:
        #     array = data.copy()
        #     print("Error in to_tensor", e)

        # var pointer = int(array.__array_interface__["data"][0].to_float64())
        # var pointer_d = DTypePointer[self.dtype](address=pointer)
        # memcpy(self.data, pointer_d, self.ndshape.ndsize)

        # _ = array  # to avoid unused variable warning
        # _ = data

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
        """
        Extremely specific NDArray initializer.
        """
        self.ndim = ndim
        self.ndshape = NDArrayShape(shape)
        self.stride = NDArrayStride(strides, offset=0, order=order)
        self.coefficient = NDArrayStride(
            coefficient, offset=offset, order=order
        )
        self.datatype = dtype
        self.order = order
        self.data = data + self.stride.ndoffset

    @always_inline("nodebug")
    fn __copyinit__(inout self, other: Self):
        """
        Copy other into self.
        """
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
        """
        Move other into self.
        """
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
    # Setter dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, val: SIMD[dtype, 1]) raises:
        """
        Set the value of a single index.
        """
        if index >= self.ndshape.ndsize:
            raise Error("Invalid index: index out of bound")
        if index >= 0:
            self.data.store[width=1](index, val)
        else:
            self.data.store[width=1](index + self.ndshape.ndsize, val)

    @always_inline("nodebug")
    fn __setitem__(inout self, *index: Int, val: SIMD[dtype, 1]) raises:
        """
        Set the value at the index list.
        """
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
        """
        Set the value at the index list.
        """
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
        """
        Set the value at the index corisponding to the varaidic list.
        """
        if index.__len__() != self.ndim:
            raise Error("Error: Length of Indices do not match the shape")
        for i in range(index.__len__()):
            if index[i] >= self.ndshape[i]:
                raise Error("Error: Elements of `index` exceed the array shape")
        var idx: Int = _get_index(index, self.coefficient)
        self.data.store[width=1](idx, val)

    # compiler doesn't accept this
    # fn __setitem__(inout self, mask: NDArray[DType.bool], value: Scalar[dtype]) raises:
    #     """
    #     Set the value of the array at the indices where the mask is true.
    #     """
    #     if mask.ndshape != self.ndshape: # this behavious could be removed potentially
    #         raise Error("Mask and array must have the same shape")

    #     for i in range(mask.ndshape.ndsize):
    #         if mask.data.load[width=1](i):
    #             print(value)
    #             self.data.store[width=1](i, value)

    # ===-------------------------------------------------------------------===#
    # Getter dunders
    # ===-------------------------------------------------------------------===#
    fn get_scalar(self, index: Int) raises -> SIMD[dtype, 1]:
        """
        Linearly retreive a value from the underlying Pointer.

        Example:
        ```console
        > Array.get_scalar(15)
        ```
        returns the item of index 15 from the array's data buffer.

        Not that it is different from `item()` as `get_scalar` does not checked
        against C-order or F-order.
        ```console
        > # A is a 3x3 matrix, F-order (column-major)
        > A.get_scalar(3)  # Row 0, Col 1
        > A.item(3)  # Row 1, Col 0
        ```
        """
        if index >= self.ndshape.ndsize:
            raise Error("Invalid index: index out of bound")
        if index >= 0:
            return self.data.load[width=1](index)
        else:
            return self.data.load[width=1](index + self.ndshape.ndsize)

    fn __getitem__(self, idx: Int) raises -> Self:
        """
        Retreive a slice of the array corrisponding to the index at the first dimension.

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
                var size_at_dim: Int = self.ndshape[i]
                slice_list.append(Slice(0, size_at_dim))

        var narr: Self = self.__getitem__(slice_list)

        if self.ndim == 1:
            narr.ndim = 0
            narr.ndshape.ndshape[0] = 0

        return narr

    fn _adjust_slice_(self, inout span: Slice, dim: Int):
        """
        Adjusts the slice values to lie within 0 and dim.
        """
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
                slice_list.append(Slice(0, self.ndshape[i]))

        var narr: Self = self[slice_list]
        return narr

    fn __getitem__(self, owned slices: List[Slice]) raises -> Self:
        """
        Retreive slices of an array from list of slices.

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
            if (
                slices[i].start >= self.ndshape[i]
                or slices[i].end > self.ndshape[i]
            ):
                raise Error("Error: Slice value exceeds the array shape")
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
                count += 1
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
            raise Error("Error: No of slices greater than rank of array")
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
                var size_at_dim: Int = self.ndshape[i]
                slice_list.append(Slice(0, size_at_dim))

        var narr: Self = self.__getitem__(slice_list)

        if count_int == self.ndim:
            narr.ndim = 0
            narr.ndshape.ndshape[0] = 0

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
        > var C = nm.NDArray[nm.i8](3, 3, 3,random=True)
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
        var ndshape = self.ndshape
        ndshape.ndshape.__setitem__(0, len(index))
        ndshape.ndsize = 1
        for i in range(ndshape.ndlen):
            ndshape.ndsize *= int(ndshape.ndshape[i])
        var result = NDArray[dtype](ndshape)
        var size_per_item = ndshape.ndsize // len(index)

        # Fill in the values
        for i in range(len(index)):
            for j in range(size_per_item):
                result.data.store[width=1](
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
        Get items of array corrisponding to a mask.

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
        for i in range(mask.ndshape.ndsize):
            if mask.data.load[width=1](i):
                true.append(i)

        var result = Self(true.__len__())
        for i in range(true.__len__()):
            result.data.store[width=1](i, self.get_scalar(true[i]))

        return result

    fn __setitem__(inout self, mask: NDArray[DType.bool], value: Self) raises:
        """
        Set the value of the array at the indices where the mask is true.

        """
        if (
            mask.ndshape != self.ndshape
        ):  # this behavious could be removed potentially
            raise Error("Mask and array must have the same shape")

        for i in range(mask.ndshape.ndsize):
            if mask.data.load[width=1](i):
                self.data.store[width=1](i, value.data.load[width=1](i))

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
            Int representation of the array

        """
        if (self.size() == 1) or (self.ndim == 0):
            return int(self.get_scalar(0))
        else:
            raise (
                "Only 0-D arrays or length-1 arrays can be converted to scalars"
            )

    fn __pos__(self) raises -> Self:
        """
        Unary positve returens self unless boolean type.
        """
        if self.dtype.is_bool():
            raise Error(
                "ndarray:NDArrray:__pos__: pos does not except bool type arrays"
            )
        return self

    fn __neg__(self) raises -> Self:
        """
        Unary negative returens self unless boolean type.

        For bolean use `__invert__`(~)
        """
        if self.dtype.is_bool():
            raise Error(
                "ndarray:NDArrray:__pos__: pos does not except bool type arrays"
            )
        return self * -1.0

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> NDArray[DType.bool]:
        """
        Itemwise equivelence.
        """
        return math.equal[dtype](self, other)

    @always_inline("nodebug")
    fn __eq__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise equivelence between scalar and Array.
        """
        return math.equal[dtype](self, other)

    @always_inline("nodebug")
    fn __ne__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise nonequivelence.
        """
        return math.not_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __ne__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise nonequivelence between scalar and Array.
        """
        return math.not_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __lt__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than.
        """
        return math.less[dtype](self, other)

    @always_inline("nodebug")
    fn __lt__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than between scalar and Array.
        """
        return math.less[dtype](self, other)

    @always_inline("nodebug")
    fn __le__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to.
        """
        return math.less_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __le__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        return math.less_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __gt__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than.
        """
        return math.greater[dtype](self, other)

    @always_inline("nodebug")
    fn __gt__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than between scalar and Array.
        """
        return math.greater[dtype](self, other)

    @always_inline("nodebug")
    fn __ge__(self, other: SIMD[dtype, 1]) raises -> NDArray[DType.bool]:
        """
        Itemwise greater than or equal to.
        """
        return math.greater_equal[dtype](self, other)

    @always_inline("nodebug")
    fn __ge__(self, other: NDArray[dtype]) raises -> NDArray[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        return math.greater_equal[dtype](self, other)

    fn __add__(inout self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `array + scalar`.
        """
        return math.add[dtype](self, other)

    fn __add__(inout self, other: Self) raises -> Self:
        """
        Enables `array + array`.
        """
        return math.add[dtype](self, other)

    fn __radd__(inout self, rhs: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar + array`.
        """
        return math.add[dtype](self, rhs)

    # TODO make an inplace version of arithmetic functions for the i dunders
    fn __iadd__(inout self, other: SIMD[dtype, 1]) raises:
        """
        Enables `array += scalar`.
        """
        self = _af.math_func_one_array_one_SIMD_in_one_array_out[
            dtype, SIMD.__add__
        ](self, other)

    fn __iadd__(inout self, other: Self) raises:
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

    fn __isub__(inout self, s: SIMD[dtype, 1]) raises:
        """
        Enables `array -= scalar`.
        """
        self = self - s

    fn __isub__(inout self, s: Self) raises:
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

    fn __imul__(inout self, s: SIMD[dtype, 1]) raises:
        """
        Enables `array *= scalar`.
        """
        self = self * s

    fn __imul__(inout self, s: Self) raises:
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
        return math.invert[dtype](self)

    fn __pow__(self, p: Int) -> Self:
        return self._elementwise_pow(p)

    fn __pow__(self, p: Self) raises -> Self:
        if self.ndshape.ndsize != p.ndshape.ndsize:
            raise Error("Both arrays must have same number of elements")

        var result = Self(self.ndshape)

        @parameter
        fn vectorized_pow[simd_width: Int](index: Int) -> None:
            result.data.store[width=simd_width](
                index,
                self.data.load[width=simd_width](index)
                ** p.load[width=simd_width](index),
            )

        vectorize[vectorized_pow, self.width](self.ndshape.ndsize)
        return result

    fn __ipow__(inout self, p: Int):
        self = self.__pow__(p)

    fn _elementwise_pow(self, p: Int) -> Self:
        var new_vec = self

        @parameter
        fn array_scalar_vectorize[simd_width: Int](index: Int) -> None:
            new_vec.data.store[width=simd_width](
                index, pow(self.data.load[width=simd_width](index), p)
            )

        vectorize[array_scalar_vectorize, self.width](self.ndshape.ndsize)
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

    fn __itruediv__(inout self, s: SIMD[dtype, 1]) raises:
        """
        Enables `array /= scalar`.
        """
        self = self.__truediv__(s)

    fn __itruediv__(inout self, other: Self) raises:
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

    fn __ifloordiv__(inout self, s: SIMD[dtype, 1]) raises:
        """
        Enables `array //= scalar`.
        """
        self = self.__floordiv__(s)

    fn __ifloordiv__(inout self, other: Self) raises:
        """
        Enables `array //= array`.
        """
        self = self.__floordiv__(other)

    fn __rfloordiv__(self, s: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar // array`.
        """
        return math.floor_div[dtype](s, self)

    fn __mod__(inout self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `array % scalar`.
        """
        return math.mod[dtype](self, other)

    fn __mod__(inout self, other: NDArray[dtype]) raises -> Self:
        """
        Enables `array % array`.
        """
        return math.mod[dtype](self, other)

    fn __imod__(inout self, other: SIMD[dtype, 1]) raises:
        """
        Enables `array %= scalar`.
        """
        self = math.mod[dtype](self, other)

    fn __imod__(inout self, other: NDArray[dtype]) raises:
        """
        Enables `array %= array`.
        """
        self = math.mod[dtype](self, other)

    fn __rmod__(inout self, other: SIMD[dtype, 1]) raises -> Self:
        """
        Enables `scalar % array`.
        """
        return math.mod[dtype](other, self)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __str__(self) -> String:
        """
        Enables str(array)
        """
        try:
            return (
                self._array_to_string(0, 0)
                + "\n"
                + str(self.ndim)
                + "-D array  "
                + self.ndshape.__str__()
                + "  DType: "
                + self.dtype.__str__()
            )
        except e:
            print("Cannot convert array to string", e)
            return ""

    fn __repr__(self) -> String:
        """
        Compute the "official" string representation of NDArray.
        An example is:
        ```mojo
        fn main() raises:
            var A = NDArray[DType.int8](List[Scalar[DType.int8]](14,97,-59,-4,112,), shape=List[Int](5,))
            print(repr(A))
        ```
        It prints what can be used to construct the array itself:
        ```console
        NDArray[DType.int8](List[Scalar[DType.int8]](14,97,-59,-4,112,), shape=List[Int](5,))
        ```
        """
        try:
            var result: String = str("NDArray[DType.") + str(self.dtype) + str(
                "](List[Scalar[DType."
            ) + str(self.dtype) + str("]](")
            if self.size() > 6:
                for i in range(6):
                    result = result + str(self.load[width=1](i)) + str(",")
                result = result + " ... "
            else:
                for i in self:
                    result = result + str(i) + str(",")
            result = result + str("), shape=List[Int](")
            for i in range(self.ndshape.ndlen):
                result = result + str(self.ndshape.ndshape[i]) + ","
            result = result + str("))")
            return result
        except e:
            print("Cannot convert array to string", e)
            return ""

    # Should len be size or number of dimensions instead of the first dimension shape?
    fn __len__(self) -> Int:
        return int(self.ndshape.ndshape[0])

    fn __iter__(self) raises -> _NDArrayIter[__lifetime_of(self), dtype]:
        """Iterate over elements of the NDArray, returning copied value.

        Returns:
            An iterator of NDArray elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _NDArrayIter[__lifetime_of(self), dtype](
            array=self,
            length=self.ndshape[0],
        )

    fn __reversed__(
        self,
    ) raises -> _NDArrayIter[__lifetime_of(self), dtype, forward=False]:
        """Iterate backwards over elements of the NDArray, returning
        copied value.

        Returns:
            A reversed iterator of NDArray elements.
        """

        return _NDArrayIter[__lifetime_of(self), dtype, forward=False](
            array=self,
            length=self.ndshape[0],
        )

    fn _array_to_string(self, dimension: Int, offset: Int) raises -> String:
        if self.ndim == 0:
            return str(self.item(0))
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
        if self.ndshape.ndsize != other.ndshape.ndsize:
            raise Error("The lengths of two vectors do not match.")

        var sum = Scalar[dtype](0)
        for i in range(self.ndshape.ndsize):
            sum = sum + self.get_scalar(i) * other.get_scalar(i)
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
            buffer.__setitem__(i, self.data.load[width=1](i + id * width))
        return buffer

    fn col(self, id: Int) raises -> Self:
        """Get the ith column of the matrix."""

        if self.ndim > 2:
            raise Error("Only support 2-D array (matrix).")

        var width = self.ndshape[1]
        var height = self.ndshape[0]
        var buffer = Self(height)
        for i in range(height):
            buffer.__setitem__(i, self.data.load[width=1](id + i * width))
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
        if self.ndshape.ndshape[1] != other.ndshape.ndshape[0]:
            raise Error(
                "Second dimension of A does not match first dimension of B."
            )

        var new_matrix = Self(self.ndshape[0], other.ndshape[1])
        for row in range(self.ndshape[0]):
            for col in range(other.ndshape[1]):
                new_matrix.__setitem__(
                    col + row * other.ndshape[1],
                    self.row(row).vdot(other.col(col)),
                )
        return new_matrix

    fn size(self) -> Int:
        """
        Function to retreive size.
        """
        return self.ndshape.ndsize

    fn num_elements(self) -> Int:
        """
        Function to retreive size (compatability).
        """
        return self.ndshape.ndsize

    # should this return the List[Int] shape and self.ndshape be used instead of making it a no input function call?
    # * We are fixed dtype for this array shape for the linalg solve module.
    fn shape(self) -> NDArrayShape[i32]:
        """
        Get the shape as an NDArray Shape.

        To get a list of shape call this then list
        """
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

    # ===-------------------------------------------------------------------===#
    # Operations along an axis
    # ===-------------------------------------------------------------------===#
    # TODO: implement for arbitrary axis1, and axis2
    fn T(inout self) raises:
        """
        Transpose the array.
        """
        if self.ndim != 2:
            raise Error("Only 2-D arrays can be transposed currently.")
        var rows = self.ndshape[0]
        var cols = self.ndshape[1]

        var transposed = NDArray[dtype](cols, rows)
        for i in range(rows):
            for j in range(cols):
                # the setitem is not working due to the symmetry issue of getter and setter
                transposed.__setitem__(
                    VariadicList[Int](j, i), val=self.item(i, j)
                )
        self = transposed

    fn all(self) raises -> Bool:
        """
        If all true return true.
        """
        # make this a compile time check
        # Respnse to above compile time errors are way harder to read at the moment.
        if not (self.dtype.is_bool() or self.dtype.is_integral()):
            raise Error("Array elements must be Boolean or Integer.")
        # We might need to figure out how we want to handle truthyness before can do this
        var result: Bool = True

        @parameter
        fn vectorized_all[simd_width: Int](idx: Int) -> None:
            result = result and allb(
                (self.data + idx).simd_strided_load[width=simd_width](1)
            )

        vectorize[vectorized_all, self.width](self.ndshape.ndsize)
        return result

    fn any(self) raises -> Bool:
        """
        True if any true.
        """
        # make this a compile time check
        if not (self.dtype.is_bool() or self.dtype.is_integral()):
            raise Error("Array elements must be Boolean or Integer.")
        var result: Bool = False

        @parameter
        fn vectorized_any[simd_width: Int](idx: Int) -> None:
            result = result or anyb(
                (self.data + idx).simd_strided_load[width=simd_width](1)
            )

        vectorize[vectorized_any, self.width](self.ndshape.ndsize)
        return result

    fn argmax(self) -> Int:
        """
        Get location in pointer of max value.
        """
        var result: Int = 0
        var max_val: SIMD[dtype, 1] = self.load[width=1](0)
        for i in range(1, self.ndshape.ndsize):
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
        for i in range(1, self.ndshape.ndsize):
            var temp: SIMD[dtype, 1] = self.load[width=1](i)
            if temp < min_val:
                min_val = temp
                result = i
        return result

    fn argsort(self) raises -> NDArray[DType.index]:
        """
        Sort the NDArray and return the sorted indices.

        See `numojo.core.sort.argsort()`.

        Returns:
            The indices of the sorted NDArray.
        """

        return sort.argsort(self)

    fn astype[type: DType](self) raises -> NDArray[type]:
        """
        Convert type of array.
        """
        # I wonder if we can do this operation inplace instead of allocating memory.
        alias nelts = simdwidthof[dtype]()
        var narr: NDArray[type] = NDArray[type](self.ndshape, order=self.order)
        # narr.datatype = type

        @parameter
        if type == DType.bool:

            @parameter
            fn vectorized_astype[width: Int](idx: Int) -> None:
                (narr.unsafe_ptr() + idx).simd_strided_store[width](
                    self.load[width](idx).cast[type](), 1
                )

            vectorize[vectorized_astype, self.width](self.ndshape.ndsize)
        else:

            @parameter
            if self.dtype == DType.bool:

                @parameter
                fn vectorized_astypenb_from_b[width: Int](idx: Int) -> None:
                    narr.store[width](
                        idx,
                        (self.data + idx)
                        .simd_strided_load[width](1)
                        .cast[type](),
                    )

                vectorize[vectorized_astypenb_from_b, self.width](
                    self.ndshape.ndsize
                )
            else:

                @parameter
                fn vectorized_astypenb[width: Int](idx: Int) -> None:
                    narr.store[width](idx, self.load[width](idx).cast[type]())

                vectorize[vectorized_astypenb, self.width](self.ndshape.ndsize)

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
        """
        Fill all items of array with value.
        """

        @parameter
        fn vectorized_fill[simd_width: Int](index: Int) -> None:
            self.data.store[width=simd_width](index, val)

        vectorize[vectorized_fill, self.width](self.ndshape.ndsize)

        return self

    fn flatten(inout self) raises:
        """
        Convert shape of array to one dimensional.
        """
        self.ndshape = NDArrayShape(
            self.ndshape.ndsize, size=self.ndshape.ndsize
        )
        self.stride = NDArrayStride(shape=self.ndshape, offset=0)

        # var res: NDArray[dtype] = NDArray[dtype](self.ndshape.ndsize)
        # alias width: Int = simdwidthof[dtype]()

        # @parameter
        # fn vectorized_flatten[simd_width: Int](index: Int) -> None:
        #     res.data.store[width=simd_width](
        #         index, self.data.load[width=simd_width](index)
        #     )

        # vectorize[vectorized_flatten, simd_width](self.ndshape.ndsize)
        # self = res^

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
        c stride Stride: [3, 1]
        14
        c stride Stride: [3, 1]
        -4
        c stride Stride: [3, 1]
        -48
        c stride Stride: [3, 1]
        97
        c stride Stride: [3, 1]
        112
        c stride Stride: [3, 1]
        -40
        c stride Stride: [3, 1]
        -59
        c stride Stride: [3, 1]
        -94
        c stride Stride: [3, 1]
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
            if index[0] < self.size():
                if (
                    self.order == "F"
                ):  # column-major should be converted to row-major
                    # The following code can be taken out as a function that
                    # convert any index to coordinates according to the order
                    var c_stride = NDArrayStride(shape=self.ndshape)
                    var c_coordinates = List[Int]()
                    var idx: Int = index[0]
                    for i in range(c_stride.ndlen):
                        var coordinate = idx // c_stride[i]
                        idx = idx - c_stride[i] * coordinate
                        c_coordinates.append(coordinate)
                    return self.data.load[width=1](
                        _get_index(c_coordinates, self.stride)
                    )

                return self.data.load[width=1](index[0])
            else:
                raise Error("Error: Elements of `index` exceed the array size")

        # If more than one index is given
        if index.__len__() != self.ndim:
            raise Error("Error: Length of Indices do not match the shape")
        for i in range(index.__len__()):
            if index[i] >= self.ndshape[i]:
                raise Error("Error: Elements of `index` exceed the array shape")
        return self.data.load[width=1](_get_index(index, self.stride))

    fn max(self, axis: Int = 0) raises -> Self:
        """
        Max on axis.
        """
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

        var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(result_shape))
        slices[axis] = Slice(0, 1)
        result = self[slices]
        for i in range(1, axis_size):
            slices[axis] = Slice(i, i + 1)
            var arr_slice = self[slices]
            var mask1 = greater(arr_slice, result)
            var mask2 = less(arr_slice, result)
            # Wherever result is less than the new slice it is set to zero
            # Wherever arr_slice is greater than the old result it is added to fill those zeros
            result = add(
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

        var result: NDArray[dtype] = NDArray[dtype](NDArrayShape(result_shape))
        slices[axis] = Slice(0, 1)
        result = self[slices]
        for i in range(1, axis_size):
            slices[axis] = Slice(i, i + 1)
            var arr_slice = self[slices]
            var mask1 = less(arr_slice, result)
            var mask2 = greater(arr_slice, result)
            # Wherever result is greater than the new slice it is set to zero
            # Wherever arr_slice is less than the old result it is added to fill those zeros
            result = add(
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
        return tround[dtype](self)

    fn sort(inout self) raises:
        """
        Sort the array inplace using quickstort.
        """
        sort.quick_sort_inplace[dtype](self, 0, self.size() - 1)

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
        for i in range(self.ndshape.ndsize):
            result.append(self.data[i])
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
        return trace[dtype](self, offset, axis1, axis2)

    # Technically it only changes the ArrayDescriptor and not the fundamental data
    fn reshape(inout self, *shape: Int, order: String = "C") raises:
        """
        Reshapes the NDArray to given Shape.

        Args:
            shape: Variadic list of shape.
            order: Order of the array - Row major `C` or Column major `F`.
        """
        var s: VariadicList[Int] = shape
        reshape[dtype](self, s, order=order)

    fn unsafe_ptr(self) -> DTypePointer[dtype, 0]:
        """
        Retreive pointer without taking ownership.
        """
        return self.data

    fn to_numpy(self) raises -> PythonObject:
        """
        Convert to a numpy array.
        """
        return to_numpy(self)
