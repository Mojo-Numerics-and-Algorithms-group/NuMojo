"""
Implements NDArrayShape type.

`NDArrayShape` is a series of `DType.int32` on the heap.
"""

from memory import memset_zero, memcpy
from utils import Variant
from builtin.type_aliases import AnyLifetime

alias Shape = NDArrayShape
alias shape = NDArrayShape


@register_passable("trivial")
struct NDArrayShape[dtype: DType = DType.int32](Stringable, Formattable):
    """Implements the NDArrayShape."""

    # Fields
    var ndsize: Int
    """Total no of elements in the corresponding array."""
    var shape: UnsafePointer[Scalar[dtype]]
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
        self.shape = UnsafePointer[Scalar[dtype]]().alloc(len(shape))
        memset_zero(self.shape, len(shape))
        for i in range(len(shape)):
            self.shape[i] = shape[i]
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
        self.shape = UnsafePointer[Scalar[dtype]]().alloc(len(shape))
        memset_zero(self.shape, len(shape))
        var count: Int = 1
        for i in range(len(shape)):
            self.shape[i] = shape[i]
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
        self.shape = UnsafePointer[Scalar[dtype]]().alloc(len(shape))
        memset_zero(self.shape, len(shape))
        for i in range(len(shape)):
            self.shape[i] = shape[i]
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
        self.shape = UnsafePointer[Scalar[dtype]]().alloc(len(shape))
        memset_zero(self.shape, len(shape))
        var count: Int = 1
        for i in range(len(shape)):
            self.shape[i] = shape[i]
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
        self.shape = UnsafePointer[Scalar[dtype]]().alloc(len(shape))
        memset_zero(self.shape, len(shape))
        for i in range(len(shape)):
            self.shape[i] = shape[i]
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
        self.shape = UnsafePointer[Scalar[dtype]]().alloc(len(shape))
        memset_zero(self.shape, len(shape))
        var count: Int = 1
        for i in range(len(shape)):
            self.shape[i] = shape[i]
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
        self.shape = UnsafePointer[Scalar[dtype]]().alloc(shape.ndlen)
        memset_zero(self.shape, shape.ndlen)
        for i in range(shape.ndlen):
            self.shape[i] = shape[i]

    fn __copy__(inout self, other: Self):
        """
        Copy from other into self.
        """
        self.ndsize = other.ndsize
        self.ndlen = other.ndlen
        self.shape = UnsafePointer[Scalar[dtype]]().alloc(other.ndlen)
        memcpy(self.shape, other.shape, other.ndlen)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        """
        Get shape at specified index.
        """
        if index >= self.ndlen:
            raise Error("Index out of bound")
        if index >= 0:
            return self.shape[index].__int__()
        else:
            return self.shape[self.ndlen + index].__int__()

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, val: Int) raises:
        """
        Set shape at specified index.
        """
        if index >= self.ndlen:
            raise Error("Index out of bound")
        if index >= 0:
            self.shape[index] = val
        else:
            self.shape[self.ndlen + index] = val

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
    fn __str__(self) -> String:
        """
        Return a string of the shape of the array described by arrayshape.
        """
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        var result: String = "Shape: ["
        for i in range(self.ndlen):
            if i == self.ndlen - 1:
                result += self.shape[i].__str__()
            else:
                result += self.shape[i].__str__() + ", "
        result = result + "]"
        writer.write(result)

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
        return self.shape.load[width=width](index)

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
        self.shape.store[width=width](index, val)

    @always_inline("nodebug")
    fn load_int(self, index: Int) -> Int:
        """
        SIMD load dimensional information.
        """
        return self.shape.load[width=1](index).__int__()

    @always_inline("nodebug")
    fn store_int(inout self, index: Int, val: Int):
        """
        SIMD store dimensional information.
        """
        self.shape.store[width=1](index, val)
