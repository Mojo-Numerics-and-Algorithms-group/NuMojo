# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Implements NDArrayStrides type.
"""

from memory import UnsafePointer, memcmp, memcpy

alias Strides = NDArrayStrides
"""An alias of the NDArrayStrides."""


@register_passable
struct NDArrayStrides(Sized, Stringable, Writable):
    """
    Presents the strides of `NDArray` type.

    The data buffer of the NDArrayStrides is a series of `Int` on memory.
    The number of elements in the strides must be positive.
    The number of dimension is checked upon creation of the strides.
    """

    # Fields
    var _buf: UnsafePointer[Int]
    """Data buffer."""
    var ndim: Int
    """Number of dimensions of array. It must be larger than 0."""

    @always_inline("nodebug")
    fn __init__(out self, *strides: Int) raises:
        """
        Initializes the NDArrayStrides from strides.

        Raises:
           Error: If the number of dimensions is not positive.

        Args:
            strides: Strides of the array.
        """
        if len(strides) <= 0:
            raise Error(
                String(
                    "\nError in `NDArrayShape.__init__()`: Number of dimensions"
                    " of array must be positive. However, it is {}."
                ).format(len(strides))
            )

        self.ndim = len(strides)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(strides[i])

    @always_inline("nodebug")
    fn __init__(out self, strides: List[Int]) raises:
        """
        Initializes the NDArrayStrides from a list of strides.

        Raises:
           Error: If the number of dimensions is not positive.

        Args:
            strides: Strides of the array.
        """
        if len(strides) <= 0:
            raise Error(
                String(
                    "\nError in `NDArrayShape.__init__()`: Number of dimensions"
                    " of array must be positive. However, it is {}."
                ).format(len(strides))
            )

        self.ndim = len(strides)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(strides[i])

    @always_inline("nodebug")
    fn __init__(out self, strides: VariadicList[Int]) raises:
        """
        Initializes the NDArrayStrides from a variadic list of strides.

        Raises:
           Error: If the number of dimensions is not positive.

        Args:
            strides: Strides of the array.
        """
        if len(strides) <= 0:
            raise Error(
                String(
                    "\nError in `NDArrayShape.__init__()`: Number of dimensions"
                    " of array must be positive. However, it is {}."
                ).format(len(strides))
            )

        self.ndim = len(strides)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(strides[i])

    @always_inline("nodebug")
    fn __init__(out self, strides: NDArrayStrides):
        """
        Initializes the NDArrayStrides from another strides.
        A deep-copy of the elements is conducted.

        Args:
            strides: Strides of the array.
        """

        self.ndim = strides.ndim
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        memcpy(self._buf, strides._buf, strides.ndim)

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: NDArrayShape,
        order: String = "C",
    ) raises:
        """
        Initializes the NDArrayStrides from a shape and an order.

        Raises:
            ValueError: If the order argument is not `C` or `F`.

        Args:
            shape: Shape of the array.
            order: Order of the memory layout
                (row-major "C" or column-major "F").
                Default is "C".
        """

        self.ndim = shape.ndim
        self._buf = UnsafePointer[Int]().alloc(shape.ndim)

        if order == "C":
            var temp = 1
            for i in range(self.ndim - 1, -1, -1):
                (self._buf + i).init_pointee_copy(temp)
                temp *= shape[i]
        elif order == "F":
            var temp = 1
            for i in range(0, self.ndim):
                (self._buf + i).init_pointee_copy(temp)
                temp *= shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(out self, *shape: Int, order: String) raises:
        """
        Overloads the function `__init__(shape: NDArrayShape, order: String)`.
        Initializes the NDArrayStrides from a given shapes and an order.

        Raises:
            ValueError: If the order argument is not `C` or `F`.

        Args:
            shape: Shape of the array.
            order: Order of the memory layout
                (row-major "C" or column-major "F").
                Default is "C".
        """
        self = Self(shape=NDArrayShape(shape), order=order)

    @always_inline("nodebug")
    fn __init__(out self, shape: List[Int], order: String = "C") raises:
        """
        Overloads the function `__init__(shape: NDArrayShape, order: String)`.
        Initializes the NDArrayStrides from a given shapes and an order.

        Raises:
            ValueError: If the order argument is not `C` or `F`.

        Args:
            shape: Shape of the array.
            order: Order of the memory layout
                (row-major "C" or column-major "F").
                Default is "C".
        """
        self = Self(shape=NDArrayShape(shape), order=order)

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: VariadicList[Int],
        order: String = "C",
    ) raises:
        """
        Overloads the function `__init__(shape: NDArrayShape, order: String)`.
        Initializes the NDArrayStrides from a given shapes and an order.

        Raises:
            ValueError: If the order argument is not `C` or `F`.

        Args:
            shape: Shape of the array.
            order: Order of the memory layout
                (row-major "C" or column-major "F").
                Default is "C".
        """
        self = Self(shape=NDArrayShape(shape), order=order)

    @always_inline("nodebug")
    fn __init__(
        out self,
        ndim: Int,
        initialized: Bool,
    ) raises:
        """
        Construct NDArrayStrides with number of dimensions.
        This method is useful when you want to create a strides with given ndim
        without knowing the strides values.
        `ndim == 0` is allowed in this method for 0darray (numojo scalar).

        Raises:
           Error: If the number of dimensions is negative.

        Args:
            ndim: Number of dimensions.
            initialized: Whether the strides is initialized.
                If yes, the values will be set to 0.
                If no, the values will be uninitialized.
        """
        if ndim < 0:
            raise Error(
                "Error in `numojo.NDArrayStrides.__init__(out self, ndim:"
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
                    (self._buf + i).init_pointee_copy(0)

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """
        Initializes the NDArrayStrides from another strides.
        A deep-copy of the elements is conducted.

        Args:
            other: Strides of the array.
        """
        self.ndim = other.ndim
        self._buf = UnsafePointer[Int]().alloc(other.ndim)
        memcpy(self._buf, other._buf, other.ndim)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        """
        Gets stride at specified index.

        Raises:
           Error: Index out of bound.

        Args:
          index: Index to get the stride.

        Returns:
           Stride value at the given index.
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
        Sets stride at specified index.

        Raises:
           Error: Index out of bound.
           Error: Value is not positive.

        Args:
          index: Index to get the stride.
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

        self._buf[normalized_index] = val

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """
        Gets number of elements in the strides.
        It equals to the number of dimensions of the array.

        Returns:
          Number of elements in the strides.
        """
        return self.ndim

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """
        Returns a string of the strides of the array.

        Returns:
            String representation of the strides of the array.
        """
        return "numojo.Strides" + String(self)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """
        Returns a string of the strides of the array.

        Returns:
            String representation of the strides of the array.
        """
        var result: String = "("
        for i in range(self.ndim):
            result += String(self._buf[i])
            if i < self.ndim - 1:
                result += ","
        result = result + ")"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Strides: " + String(self) + "  " + "ndim: " + String(self.ndim)
        )

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """
        Checks if two strides have identical dimensions and values.

        Args:
            other: The strides to compare with.

        Returns:
            True if both strides have identical dimensions and values.
        """
        if self.ndim != other.ndim:
            return False
        if memcmp(self._buf, other._buf, self.ndim) != 0:
            return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """
        Checks if two strides have identical dimensions and values.

        Returns:
           True if both strides do not have identical dimensions or values.
        """
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, val: Int) -> Bool:
        """
        Checks if the given value is present in the strides.

        Returns:
          True if the given value is present in the strides.
        """
        for i in range(self.ndim):
            if self._buf[i] == val:
                return True
        return False

    # ===-------------------------------------------------------------------===#
    # Other methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn copy(read self) raises -> Self:
        """
        Returns a deep copy of the strides.
        """

        var res = Self(ndim=self.ndim, initialized=False)
        memcpy(res._buf, self._buf, self.ndim)
        return res

    fn swapaxes(self, axis1: Int, axis2: Int) raises -> Self:
        """
        Returns a new strides with the given axes swapped.

        Args:
            axis1: The first axis to swap.
            axis2: The second axis to swap.

        Returns:
            A new strides with the given axes swapped.
        """
        var res = self
        res[axis1] = self[axis2]
        res[axis2] = self[axis1]
        return res

    # ===-------------------------------------------------------------------===#
    # Other private methods
    # ===-------------------------------------------------------------------===#

    fn _flip(self) -> Self:
        """
        Returns a new strides by flipping the items.
        ***UNSAFE!*** No boundary check!

        Returns:
            A new strides with the items flipped.

        Example:
        ```mojo
        import numojo as nm
        var A = nm.random.randn(2, 3, 4)
        print(A.strides)          # Stride: [12, 4, 1]
        print(A.strides._flip())  # Stride: [1, 4, 12]
        ```
        """

        var strides = NDArrayStrides(self)
        for i in range(strides.ndim):
            strides._buf[i] = self._buf[self.ndim - 1 - i]
        return strides

    fn _move_axis_to_end(self, owned axis: Int) -> Self:
        """
        Returns a new strides by moving the value of axis to the end.
        ***UNSAFE!*** No boundary check!

        Example:
        ```mojo
        import numojo as nm
        var A = nm.random.randn(2, 3, 4)
        print(A.strides)                       # Stride: [12, 4, 1]
        print(A.strides._move_axis_to_end(0))  # Stride: [4, 1, 12]
        print(A.strides._move_axis_to_end(1))  # Stride: [12, 1, 4]
        ```
        """

        if axis < 0:
            axis += self.ndim

        var strides = NDArrayStrides(self)

        if axis == self.ndim - 1:
            return strides

        var value = strides._buf[axis]
        for i in range(axis, strides.ndim - 1):
            strides._buf[i] = strides._buf[i + 1]
        strides._buf[strides.ndim - 1] = value
        return strides

    fn _pop(self, axis: Int) raises -> Self:
        """
        Drops information of certain axis.
        ***UNSAFE!*** No boundary check!

        Args:
            axis: The axis (index) to drop. It should be in `[0, ndim)`.

        Returns:
            A new stride with the item at the given axis (index) dropped.
        """
        var res = Self(ndim=self.ndim - 1, initialized=False)
        memcpy(dest=res._buf, src=self._buf, count=axis)
        memcpy(
            dest=res._buf + axis,
            src=self._buf.offset(axis + 1),
            count=self.ndim - axis - 1,
        )
        return res


# @always_inline("nodebug")
# fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
#     # if index >= self.ndim:
#     #     raise Error("Index out of bound")
#     return self._buf.ptr.load[width=width](index)

# @always_inline("nodebug")
# fn store[
#     width: Int = 1
# ](mut self, index: Int, val: SIMD[dtype, width]) raises:
#     # if index >= self.ndim:
#     #     raise Error("Index out of bound")
#     self._buf.ptr.store(index, val)

# @always_inline("nodebug")
# fn load_unsafe[width: Int = 1](self, index: Int) -> Int:
#     return self._buf.ptr.load[width=width](index).__int__()

# @always_inline("nodebug")
# fn store_unsafe[
#     width: Int = 1
# ](mut self, index: Int, val: SIMD[dtype, width]):
#     self._buf.ptr.store(index, val)
