"""
Implements NDArrayShape type.

`NDArrayShape` is a series of `Int` on the heap.
"""

from memory import UnsafePointer, memcpy

alias Shape = NDArrayShape


@register_passable("trivial")
struct NDArrayShape(Stringable, Writable):
    """Implements the NDArrayShape."""

    # Fields
    var size: Int
    """Total number of elements of corresponding array."""
    var _buf: UnsafePointer[Int]
    """Data buffer."""
    var ndim: Int
    """Number of dimensions of array."""

    @always_inline("nodebug")
    fn __init__(mut self, *shape: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions.

        Args:
            shape: Variable number of integers representing the shape dimensions.
        """
        self.size = 1
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])
            self.size *= shape[i]

    @always_inline("nodebug")
    fn __init__(mut self, *shape: Int, size: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions and a specified size.

        Args:
            shape: Variable number of integers representing the shape dimensions.
            size: The total number of elements in the array.
        """
        self.size = size
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        var count: Int = 1
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])
            count *= shape[i]
        if count != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(mut self, shape: List[Int]):
        """
        Initializes the NDArrayShape with a list of shape dimensions.

        Args:
            shape: A list of integers representing the shape dimensions.
        """
        self.size = 1
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])
            self.size *= shape[i]

    @always_inline("nodebug")
    fn __init__(mut self, shape: List[Int], size: Int) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions and a specified size.

        Args:
            shape: A list of integers representing the shape dimensions.
            size: The specified size of the NDArrayShape.
        """
        self.size = (
            size  # maybe I should add a check here to make sure it matches
        )
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        var count: Int = 1
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])
            count *= shape[i]
        if count != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(mut self, shape: VariadicList[Int]):
        """
        Initializes the NDArrayShape with a list of shape dimensions.

        Args:
            shape: A list of integers representing the shape dimensions.
        """
        self.size = 1
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])
            self.size *= shape[i]

    @always_inline("nodebug")
    fn __init__(mut self, shape: VariadicList[Int], size: Int) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions and a specified size.

        Args:
            shape: A list of integers representing the shape dimensions.
            size: The specified size of the NDArrayShape.
        """
        self.size = (
            size  # maybe I should add a check here to make sure it matches
        )
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        var count: Int = 1
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])
            count *= shape[i]
        if count != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(mut self, shape: NDArrayShape) raises:
        """
        Initializes the NDArrayShape with another NDArrayShape.

        Args:
            shape: Another NDArrayShape to initialize from.
        """
        self.size = shape.size
        self.ndim = shape.ndim
        self._buf = UnsafePointer[Int]().alloc(shape.ndim)
        memcpy(self._buf, shape._buf, shape.ndim)

    fn __copy__(mut self, other: Self):
        """
        Copy from other into self.
        """
        self.size = other.size
        self.ndim = other.ndim
        self._buf = UnsafePointer[Int]().alloc(other.ndim)
        memcpy(self._buf, other._buf, other.ndim)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        """
        Get shape at specified index.
        """
        if index >= self.ndim:
            raise Error("Index out of bound")
        if index >= 0:
            return self._buf[index].__int__()
        else:
            return self._buf[self.ndim + index].__int__()

    @always_inline("nodebug")
    fn __setitem__(mut self, index: Int, val: Int) raises:
        """
        Set shape at specified index.
        """
        if index >= self.ndim:
            raise Error("Index out of bound")
        if index >= 0:
            self._buf[index] = val
        else:
            self._buf[self.ndim + index] = val

    @always_inline("nodebug")
    fn len(self) -> Int:
        """
        Get number of dimensions of the array described by arrayshape.
        """
        return self.ndim

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """
        Return a string of the shape of the array described by arrayshape.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        var result: String = "Shape: ["
        for i in range(self.ndim):
            if i == self.ndim - 1:
                result += self._buf[i].__str__()
            else:
                result += self._buf[i].__str__() + ", "
        result = result + "]"
        writer.write(result)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Bool:
        """
        Check if two arrayshapes have identical dimensions.
        """
        for i in range(self.ndim):
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
        for i in range(self.ndim):
            if self[i] == val:
                return True
        return False

    # # can be used for vectorized index calculation
    # @always_inline("nodebug")
    # fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
    #     """
    #     SIMD load dimensional information.
    #     """
    #     # if index >= self.ndim:
    #     # raise Error("Index out of bound")
    #     return self._buf.load[width=width](index)

    # # can be used for vectorized index retrieval
    # @always_inline("nodebug")
    # fn store[
    #     width: Int = 1
    # ](mut self, index: Int, val: SIMD[dtype, width]) raises:
    #     """
    #     SIMD store dimensional information.
    #     """
    #     # if index >= self.ndim:
    #     #     raise Error("Index out of bound")
    #     self._buf.store(index, val)
