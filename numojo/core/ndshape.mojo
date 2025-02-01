# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Implements NDArrayShape type.
"""

from memory import UnsafePointer, memcpy, memcmp

alias Shape = NDArrayShape
"""An alias of the NDArrayShape."""


@register_passable
struct NDArrayShape(Stringable, Writable):
    """
    Presents the shape of `NDArray` type.

    The data buffer of the NDArrayShape is a series of `Int`.
    The number of elements in the shape must be positive, since the number of
    dimensions of the array must be larger than 0. The number of dimension is
    checkout upon creation of the shape.
    """

    # Fields
    var _buf: UnsafePointer[Int]
    """Data buffer."""
    var ndim: Int
    """Number of dimensions of array. It must be larger than 0."""

    @always_inline("nodebug")
    fn __init__(out self, shape: Int) raises:
        """
        Initializes the NDArrayShape with one dimension.

        Args:
            shape: Size of the array.
        """

        if shape < 1:
            raise Error(String("Items of shape must be positive."))

        self.ndim = 1
        self._buf = UnsafePointer[Int]().alloc(shape)
        self._buf.init_pointee_copy(shape)

    @always_inline("nodebug")
    fn __init__(out self, *shape: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions.

        Raises:
           Error: If the number of dimensions is not positive.

        Args:
            shape: Variable number of integers representing the shape dimensions.
        """
        if len(shape) <= 0:
            raise Error("Number of dimensions of array must be positive.")
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            if shape[i] < 1:
                raise Error(String("Items of shape must be positive."))
            (self._buf + i).init_pointee_copy(shape[i])

    @always_inline("nodebug")
    fn __init__(out self, *shape: Int, size: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions and a specified size.

        Raises:
            Error: If the number of dimensions is not positive.
            Error: If items of shape is not positive.
            Error: If the size is not a multiple of the product of all shape dimensions.

        Args:
            shape: Variable number of integers representing the shape dimensions.
            size: The total number of elements in the array.
        """
        if len(shape) <= 0:
            raise Error("Number of dimensions of array must be positive.")
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            if shape[i] < 1:
                raise Error(String("Items of shape must be positive."))
            (self._buf + i).init_pointee_copy(shape[i])
        if self.size_of_array() != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(out self, shape: List[Int]) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions.

        Raises:
            Error: If the number of dimensions is not positive.
            Error: If the items of the list are not positive.

        Args:
            shape: A list of integers representing the shape dimensions.
        """
        if len(shape) <= 0:
            raise Error("Number of dimensions of array must be positive.")
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            if shape[i] < 1:
                raise Error("Items of shape must be positive.")
            (self._buf + i).init_pointee_copy(shape[i])

    @always_inline("nodebug")
    fn __init__(out self, shape: List[Int], size: Int) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions and a specified size.

        Raises:
            Error: If the number of dimensions is not positive.
            Error: If the items of the list are not positive.
            Error: If the size of the array does not match the specified size.

        Args:
            shape: A list of integers representing the shape dimensions.
            size: The specified size of the NDArrayShape.
        """

        if len(shape) <= 0:
            raise Error("Number of dimensions of array must be positive.")

        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            if shape[i] < 1:
                raise Error("Items of shape must be positive.")
            (self._buf + i).init_pointee_copy(shape[i])
        if self.size_of_array() != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(out self, shape: VariadicList[Int]) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions.

        Raises:
            Error: If the number of dimensions is not positive.
            Error: If the items of the shape are not positive.

        Args:
            shape: A list of integers representing the shape dimensions.
        """

        if len(shape) <= 0:
            raise Error("Number of dimensions of array must be positive.")

        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            if shape[i] < 1:
                raise Error("Items of shape must be positive.")
            (self._buf + i).init_pointee_copy(shape[i])

    @always_inline("nodebug")
    fn __init__(out self, shape: VariadicList[Int], size: Int) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions and a specified size.

        Raises:
            Error: If the number of dimensions is not positive.
            Error: If the items of the shape are not positive.
            Error: If the size of the array does not match the specified size.

        Args:
            shape: A list of integers representing the shape dimensions.
            size: The specified size of the NDArrayShape.
        """

        if len(shape) <= 0:
            raise Error("Number of dimensions of array must be positive.")

        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            if shape[i] < 1:
                raise Error("Items of shape must be positive.")
            (self._buf + i).init_pointee_copy(shape[i])

        if self.size_of_array() != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(out self, shape: NDArrayShape) raises:
        """
        Initializes the NDArrayShape from another NDArrayShape.
        A deep copy of the data buffer is conducted.

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

        Raises:
           Error: If the number of dimensions is not positive.

        Args:
            ndim: Number of dimensions.
            initialized: Whether the shape is initialized.
                If yes, the values will be set to 1.
                If no, the values will be uninitialized.
        """
        if ndim <= 0:
            raise Error("Number of dimensions must be positive.")

        self.ndim = ndim
        self._buf = UnsafePointer[Int]().alloc(ndim)
        if initialized:
            for i in range(ndim):
                (self._buf + i).init_pointee_copy(1)

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """
        Initializes the NDArrayShape from another NDArrayShape.
        A deep copy of the data buffer is conducted.

        Args:
            other: Another NDArrayShape to initialize from.
        """
        self.ndim = other.ndim
        self._buf = UnsafePointer[Int]().alloc(other.ndim)
        memcpy(self._buf, other._buf, other.ndim)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        """
        Gets shape at specified index.

        raises:
           Error: Index out of bound.

        Args:
          index: Index to get the shape.

        Returns:
           Shape value at the given index.
        """

        var normalized_index: Int = index
        if normalized_index < 0:
            normalized_index += self.ndim
        if (normalized_index >= self.ndim) or (normalized_index < 0):
            raise Error(
                String("Index {} out of bound [{}, {})").format(
                    -self.ndim, self.ndim
                )
            )

        return self._buf[normalized_index]

    @always_inline("nodebug")
    fn __setitem__(mut self, index: Int, val: Int) raises:
        """
        Sets shape at specified index.

        raises:
           Error: Index out of bound.
           Error: Value is not positive.

        Args:
          index: Index to get the shape.
          val: Value to set at the given index.
        """

        var normalized_index: Int = index
        if normalized_index < 0:
            normalized_index += self.ndim
        if (normalized_index >= self.ndim) or (normalized_index < 0):
            raise Error(
                String("Index {} out of bound [{}, {})").format(
                    -self.ndim, self.ndim
                )
            )

        if val <= 0:
            raise Error(String("Value to be set is not positive."))

        self._buf[index] = val

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """
        Gets number of dimensions of the array.

        Returns:
          Number of dimensions of the array.
        """
        return self.ndim

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """
        Returns a string of the shape of the array.

        Returns:
            String representation of the shape of the array.
        """
        return "numojo.Shape" + str(self)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """
        Returns a string of the shape of the array.

        Returns:
            String representation of the shape of the array.
        """
        var result: String = "("
        for i in range(self.ndim):
            result += str(self._buf[i])
            if i < self.ndim - 1:
                result += ","
        result += ")"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Shape: " + str(self) + "  " + "ndim: " + str(self.ndim))

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """
        Checks if two shapes have identical dimensions and values.

        Args:
            other: The shape to compare with.

        Returns:
            True if both shapes have identical dimensions and values.
        """
        if self.ndim != other.ndim:
            return False
        if memcmp(self._buf, other._buf, self.ndim) != 0:
            return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> Bool:
        """
        Checks if two shapes have identical dimensions and values.

        Returns:
           True if both shapes do not have identical dimensions or values.
        """
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, val: Int) raises -> Bool:
        """
        Checks if the given value is present in the array.

        Returns:
          True if the given value is present in the array.
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

        Returns:
          The total number of elements in the corresponding array.
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
        ***UNSAFE!*** No boundary check!

        Example:
        ```mojo
        import numojo as nm
        var A = nm.random.randn(2, 3, 4)
        print(A.shape)          # Shape: [2, 3, 4]
        print(A.shape._flip())  # Shape: [4, 3, 2]
        ```

        Returns:
            A new shape with the items flipped.
        """

        var shape = NDArrayShape(self)
        for i in range(shape.ndim):
            shape._buf[i] = self._buf[self.ndim - 1 - i]
        return shape

    fn _move_axis_to_end(self, owned axis: Int) raises -> Self:
        """
        Returns a new shape by moving the value of axis to the end.
        ***UNSAFE!*** No boundary check!

        Args:
            axis: The axis (index) to drop. It should be in `[-ndim, ndim)`.

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
        Drops the item at the given axis (index).
        ***UNSAFE!*** No boundary check!

        Args:
            axis: The axis (index) to drop. It should be in `[0, ndim)`.

        Returns:
            A new shape with the item at the given axis (index) dropped.
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
