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
struct NDArrayShape(
    ImplicitlyCopyable, Sized, Stringable & Representable, Writable
):
    """
    Presents the shape of `NDArray` type.

    The data buffer of the NDArrayShape is a series of `Int` on memory.
    The number of elements in the shape must be positive.
    The elements of the shape must be positive.
    The number of dimension and values of elements are checked upon
    creation of the shape.
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

        Raises:
            Error: If the shape is not positive.

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
            raise Error(
                String(
                    "\nError in `NDArrayShape.__init__()`: Number of dimensions"
                    " of array must be positive. However, it is {}."
                ).format(len(shape))
            )

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
            raise Error(
                String(
                    "\nError in `NDArrayShape.__init__()`: Number of dimensions"
                    " of array must be positive. However, it is {}."
                ).format(len(shape))
            )
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
            raise Error(
                String(
                    "\nError in `NDArrayShape.__init__()`: Number of dimensions"
                    " of array must be positive. However, it is {}."
                ).format(len(shape))
            )
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            if shape[i] < 0:
                raise Error("Items of shape must be non negative.")
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
            raise Error(
                String(
                    "\nError in `NDArrayShape.__init__()`: Number of dimensions"
                    " of array must be positive. However, it is {}."
                ).format(len(shape))
            )

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
            raise Error(
                String(
                    "\nError in `NDArrayShape.__init__()`: Number of dimensions"
                    " of array must be positive. However, it is {}."
                ).format(len(shape))
            )

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
            raise Error(
                String(
                    "\nError in `NDArrayShape.__init__()`: Number of dimensions"
                    " of array must be positive. However, it is {}."
                ).format(len(shape))
            )

        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            if shape[i] < 1:
                raise Error("Items of shape must be positive.")
            (self._buf + i).init_pointee_copy(shape[i])

        if self.size_of_array() != size:
            raise Error("Cannot create NDArray: shape and size mismatch")

    @always_inline("nodebug")
    fn __init__(out self, shape: NDArrayShape):
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
            (self._buf + i).init_pointee_copy(shape._buf[i])

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
        `ndim == 0` is allowed in this method for 0darray (numojo scalar).

        Raises:
           Error: If the number of dimensions is negative.

        Args:
            ndim: Number of dimensions.
            initialized: Whether the shape is initialized.
                If yes, the values will be set to 1.
                If no, the values will be uninitialized.
        """
        if ndim < 0:
            raise Error(
                "Error in `numojo.NDArrayShape.__init__(out self, ndim:"
                " Int, initialized: Bool,)`. \n"
                "Number of dimensions must be non-negative."
            )

        if ndim == 0:
            # This is a 0darray (numojo scalar)
            self.ndim = ndim
            self._buf = UnsafePointer[Int]()

        else:
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

        var normalized_idx: Int = index
        if normalized_idx < 0:
            normalized_idx += self.ndim
        if (normalized_idx >= self.ndim) or (normalized_idx < 0):
            raise Error(
                String("Index {} out of bound [{}, {})").format(
                    -self.ndim, self.ndim
                )
            )

        return self._buf[normalized_idx]

    # TODO: Check the negative steps result
    @always_inline("nodebug")
    fn _compute_slice_params(
        self, slice_index: Slice
    ) raises -> (Int, Int, Int):
        var n = self.ndim
        if n == 0:
            return (0, 1, 0)

        var step = slice_index.step.or_else(1)
        if step == 0:
            raise Error("Slice step cannot be zero.")

        var start: Int
        var stop: Int
        if step > 0:
            start = slice_index.start.or_else(0)
            stop = slice_index.end.or_else(n)
        else:
            start = slice_index.start.or_else(n - 1)
            stop = slice_index.end.or_else(-1)

        if start < 0:
            start += n
        if stop < 0:
            stop += n

        if step > 0:
            if start < 0:
                start = 0
            if start > n:
                start = n
            if stop < 0:
                stop = 0
            if stop > n:
                stop = n
        else:
            if start >= n:
                start = n - 1
            if start < -1:
                start = -1
            if stop >= n:
                stop = n - 1
            if stop < -1:
                stop = -1

        var length: Int = 0
        if step > 0:
            if start < stop:
                length = Int((stop - start + step - 1) / step)
        else:
            if start > stop:
                var neg_step = -step
                length = Int((start - stop + neg_step - 1) / neg_step)

        return (start, step, length)

    @always_inline("nodebug")
    fn __getitem__(self, slice_index: Slice) raises -> NDArrayShape:
        """
        Return a sliced view of the dimension tuple as a new NDArrayShape.
        Delegates normalization & validation to _compute_slice_params.
        """
        var updated_slice: Tuple[Int, Int, Int] = self._compute_slice_params(
            slice_index
        )
        var start = updated_slice[0]
        var step = updated_slice[1]
        var length = updated_slice[2]

        if length <= 0:
            return NDArrayShape(ndim=0, initialized=False)

        var result = NDArrayShape(ndim=length, initialized=False)
        var idx = start
        for i in range(length):
            (result._buf + i).init_pointee_copy(self._buf[idx])
            idx += step
        return result^

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

        var normalized_idx: Int = index
        if normalized_idx < 0:
            normalized_idx += self.ndim
        if (normalized_idx >= self.ndim) or (normalized_idx < 0):
            raise Error(
                String("Index {} out of bound [{}, {})").format(
                    -self.ndim, self.ndim
                )
            )

        if val <= 0:
            raise Error(
                String(
                    "\nError in `NDArrayShape.__setitem__`: "
                    "Value to be set is not positive."
                )
            )

        self._buf[normalized_idx] = val

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """
        Gets number of elements in the shape.
        It equals the number of dimensions of the array.

        Returns:
          Number of elements in the shape.
        """
        return self.ndim

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """
        Returns a string of the shape of the array.

        Returns:
            String representation of the shape of the array.
        """
        return "numojo.Shape" + String(self)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """
        Returns a string of the shape of the array.

        Returns:
            String representation of the shape of the array.
        """
        var result: String = "("
        for i in range(self.ndim):
            result += String(self._buf[i])
            if i < self.ndim - 1:
                result += ","
        result += ")"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Shape: " + String(self) + "  " + "ndim: " + String(self.ndim)
        )

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

    @always_inline("nodebug")
    fn copy(read self) raises -> Self:
        """
        Returns a deep copy of the shape.
        """

        var res = Self(ndim=self.ndim, initialized=False)
        memcpy(res._buf, self._buf, self.ndim)
        return res

    fn join(self, *shapes: Self) raises -> Self:
        """
        Join multiple shapes into a single shape.

        Args:
            shapes: Variable number of NDArrayShape objects.

        Returns:
            A new NDArrayShape object.
        """
        var total_dims = self.ndim
        for i in range(len(shapes)):
            total_dims += shapes[i].ndim

        var new_shape = Self(ndim=total_dims, initialized=False)

        var index = 0
        for i in range(self.ndim):
            (new_shape._buf + index).init_pointee_copy(self[i])
            index += 1

        for i in range(len(shapes)):
            for j in range(shapes[i].ndim):
                (new_shape._buf + index).init_pointee_copy(shapes[i][j])
                index += 1

        return new_shape

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

    fn swapaxes(self, axis1: Int, axis2: Int) raises -> Self:
        """
        Returns a new shape with the given axes swapped.

        Args:
            axis1: The first axis to swap.
            axis2: The second axis to swap.

        Returns:
            A new shape with the given axes swapped.
        """
        var res = self
        res[axis1] = self[axis2]
        res[axis2] = self[axis1]
        return res

    # ===-------------------------------------------------------------------===#
    # Other private methods
    # ===-------------------------------------------------------------------===#

    fn _extend(self, *shapes: Int) raises -> Self:
        """
        Extend the shape by sizes of extended dimensions.
        ***UNSAFE!*** No boundary check!

        Args:
            shapes: Sizes of extended dimensions.

        Returns:
            A new NDArrayShape object.
        """
        var total_dims = self.ndim + len(shapes)

        var new_shape = Self(ndim=total_dims, initialized=False)

        var offset = 0
        for i in range(self.ndim):
            (new_shape._buf + offset).init_pointee_copy(self[i])
            offset += 1
        for shape in shapes:
            (new_shape._buf + offset).init_pointee_copy(shape)
            offset += 1

        return new_shape

    fn _flip(self) -> Self:
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

    fn _move_axis_to_end(self, var axis: Int) -> Self:
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

        var value = shape._buf[axis]
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
