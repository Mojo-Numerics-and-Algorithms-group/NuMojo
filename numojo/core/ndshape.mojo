"""
Implements NDArrayShape type.

`NDArrayShape` is a series of `Int` on the heap.
"""

from memory import UnsafePointer, memcpy, memcmp

alias Shape = NDArrayShape


@register_passable
struct NDArrayShape(Stringable, Writable):
    """Implements the NDArrayShape."""

    # Fields
    var _buf: UnsafePointer[Int]
    """Data buffer."""
    var ndim: Int
    """Number of dimensions of array."""

    @always_inline("nodebug")
    fn __init__(out self, shape: Int):
        """
        Initializes the NDArrayShape with one dimension.

        Args:
            shape: Size of the array.
        """
        self.ndim = 1
        self._buf = UnsafePointer[Int]().alloc(shape)
        self._buf.init_pointee_copy(shape)

    @always_inline("nodebug")
    fn __init__(out self, *shape: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions.

        Args:
            shape: Variable number of integers representing the shape dimensions.
        """
        if len(shape) == 0:
            raise Error("Cannot create NDArray: shape cannot be empty")
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])

    @always_inline("nodebug")
    fn __init__(out self, *shape: Int, size: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions and a specified size.

        Args:
            shape: Variable number of integers representing the shape dimensions.
            size: The total number of elements in the array.
        """
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])
        if self.size_of_array() != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(out self, shape: List[Int]):
        """
        Initializes the NDArrayShape with a list of shape dimensions.

        Args:
            shape: A list of integers representing the shape dimensions.
        """
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])

    @always_inline("nodebug")
    fn __init__(out self, shape: List[Int], size: Int) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions and a specified size.

        Args:
            shape: A list of integers representing the shape dimensions.
            size: The specified size of the NDArrayShape.
        """

        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])
        if self.size_of_array() != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(out self, shape: VariadicList[Int]):
        """
        Initializes the NDArrayShape with a list of shape dimensions.

        Args:
            shape: A list of integers representing the shape dimensions.
        """
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])

    @always_inline("nodebug")
    fn __init__(out self, shape: VariadicList[Int], size: Int) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions and a specified size.

        Args:
            shape: A list of integers representing the shape dimensions.
            size: The specified size of the NDArrayShape.
        """

        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])
        if self.size_of_array() != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(out self, shape: NDArrayShape) raises:
        """
        Initializes the NDArrayShape with another NDArrayShape.

        Args:
            shape: Another NDArrayShape to initialize from.
        """
        self.ndim = shape.ndim
        self._buf = UnsafePointer[Int]().alloc(shape.ndim)
        memcpy(self._buf, shape._buf, shape.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(shape[i])

    @always_inline("nodebug")
    fn __init__(
        out self,
        ndim: Int,
        initialized: Bool,
    ) raises:
        """
        Construct NDArrayShape with number of dimensions.

        This method is useful when you want to create a shape with given ndim
        without knowing the shape values.

        Args:
            ndim: Number of dimensions.
            initialized: Whether the shape is initialized.
                If yes, the values will be set to 1.
                If no, the values will be uninitialized.
        """
        if ndim < 0:
            raise Error("Number of dimensions must be non-negative.")
        self.ndim = ndim
        self._buf = UnsafePointer[Int]().alloc(ndim)
        if initialized:
            for i in range(ndim):
                (self._buf + i).init_pointee_copy(1)

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """
        Initializes the NDArrayShape from another NDArrayShape.

        Args:
            other: Another NDArrayShape to initialize from.
        """
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
    fn __len__(self) -> Int:
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
        Check if two shapes are identical.
        """
        if self.ndim != other.ndim:
            return False
        if memcmp(self._buf, other._buf, self.ndim) != 0:
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

    # ===-------------------------------------------------------------------===#
    # Other methods
    # ===-------------------------------------------------------------------===#

    fn size_of_array(self) -> Int:
        """
        Returns the total number of elements in the array.
        """
        var size = 1
        for i in range(self.ndim):
            size *= self._buf[i]
        return size

    @staticmethod
    fn join(*shapes: Self) raises -> Self:
        """
        Join multiple shapes into a single shape.

        Args:
            shapes: Variable number of NDArrayShape objects.

        Returns:
            A new NDArrayShape object.
        """
        var total_dims = 0
        for shape in shapes:
            total_dims += shape[].ndim

        var new_shape = Self(ndim=total_dims, initialized=False)

        var index = 0
        for shape in shapes:
            for i in range(shape[].ndim):
                (new_shape._buf + index).init_pointee_copy(shape[][i])
                index += 1

        return new_shape

    # ===-------------------------------------------------------------------===#
    # Other private methods
    # ===-------------------------------------------------------------------===#

    fn _flip(self) raises -> Self:
        """
        Returns a new shape by flipping the items.

        UNSAFE! No boundary check!

        Example:
        ```mojo
        import numojo as nm
        var A = nm.random.randn(2, 3, 4)
        print(A.shape)          # Shape: [2, 3, 4]
        print(A.shape._flip())  # Shape: [4, 3, 2]
        ```
        """

        var shape = NDArrayShape(self)
        for i in range(shape.ndim):
            shape._buf[i] = self._buf[self.ndim - 1 - i]
        return shape

    fn _move_axis_to_end(self, owned axis: Int) raises -> Self:
        """
        Returns a new shape by moving the value of axis to the end.

        UNSAFE! No boundary check!

        Example:
        ```mojo
        import numojo as nm
        var A = nm.random.randn(2, 3, 4)
        print(A.shape._move_axis_to_end(0))  # Shape: [3, 4, 2]
        print(A.shape._move_axis_to_end(1))  # Shape: [2, 4, 3]
        ```
        """

        if axis < 0:
            axis += self.ndim

        var shape = NDArrayShape(self)

        if axis == self.ndim - 1:
            return shape

        var value = shape[axis]
        for i in range(axis, shape.ndim - 1):
            shape._buf[i] = shape._buf[i + 1]
        shape._buf[shape.ndim - 1] = value
        return shape

    fn _pop(self, axis: Int) raises -> Self:
        """
        drop information of certain axis.
        """
        var res = Self(ndim=self.ndim - 1, initialized=False)
        memcpy(dest=res._buf, src=self._buf, count=axis)
        memcpy(
            dest=res._buf + axis,
            src=self._buf + axis + 1,
            count=self.ndim - axis - 1,
        )
        return res

    # # can be used for vectorized index calculation
    # @always_inline("nodebug")
    # fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
    #     """
    #     SIMD load dimensional information.
    #     """
    #     if index >= self.ndim:
    #         raise Error("Index out of bound")
    #     return self._buf.load[width=width](index)

    # # can be used for vectorized index retrieval
    # @always_inline("nodebug")
    # fn store[
    #     width: Int = 1
    # ](out self, index: Int, val: SIMD[dtype, width]) raises:
    #     """
    #     SIMD store dimensional information.
    #     """
    #     # if index >= self.ndim:
    #     #     raise Error("Index out of bound")
    #     self._buf.ptr.store(index, val)
