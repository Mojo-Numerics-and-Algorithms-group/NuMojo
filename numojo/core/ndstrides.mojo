# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Implements NDArrayStrides type.
"""

from memory import memcmp, memcpy
from memory import LegacyUnsafePointer as UnsafePointer

from numojo.core.error import IndexError, ValueError

comptime strides = NDArrayStrides
comptime Strides = NDArrayStrides
"""An comptime of the NDArrayStrides."""


@register_passable
struct NDArrayStrides(
    ImplicitlyCopyable, Movable, Representable, Sized, Stringable, Writable
):
    """
    Presents the strides of `NDArray` type.

    The data buffer of the NDArrayStrides is a series of `Int` on memory.
    The number of elements in the strides must be positive.
    The number of dimension is checked upon creation of the strides.
    """

    # Aliases
    comptime _type: DType = DType.int

    # Fields
    var _buf: UnsafePointer[Scalar[Self._type]]
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
                ValueError(
                    message=String(
                        "Number of dimensions must be positive, got {}."
                    ).format(len(strides)),
                    suggestion="Provide at least one stride value.",
                    location="NDArrayStrides.__init__(*strides: Int)",
                )
            )

        self.ndim = len(strides)
        self._buf = UnsafePointer[Scalar[Self._type]]().alloc(self.ndim)
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
                ValueError(
                    message=String(
                        "Number of dimensions must be positive, got {}."
                    ).format(len(strides)),
                    suggestion="Provide a non-empty list of strides.",
                    location="NDArrayStrides.__init__(strides: List[Int])",
                )
            )

        self.ndim = len(strides)
        self._buf = UnsafePointer[Scalar[Self._type]]().alloc(self.ndim)
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
                ValueError(
                    message=String(
                        "Number of dimensions must be positive, got {}."
                    ).format(len(strides)),
                    suggestion="Provide a non-empty variadic list of strides.",
                    location=(
                        "NDArrayStrides.__init__(strides: VariadicList[Int])"
                    ),
                )
            )

        self.ndim = len(strides)
        self._buf = UnsafePointer[Scalar[Self._type]]().alloc(self.ndim)
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
        self._buf = UnsafePointer[Scalar[Self._type]]().alloc(self.ndim)
        memcpy(dest=self._buf, src=strides._buf, count=strides.ndim)

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
        self._buf = UnsafePointer[Scalar[Self._type]]().alloc(shape.ndim)

        if order == "C":
            var temp = 1
            for i in range(self.ndim - 1, -1, -1):
                (self._buf + i).init_pointee_copy(temp)
                temp *= shape[i]
        elif order == "F":
            var temp = 1
            # Should we check for temp overflow here? Maybe no?
            for i in range(0, self.ndim):
                (self._buf + i).init_pointee_copy(temp)
                temp *= shape[i]
        else:
            raise Error(
                ValueError(
                    message=String(
                        "Invalid order '{}'; expected 'C' or 'F'."
                    ).format(order),
                    suggestion=(
                        "Use 'C' for row-major or 'F' for column-major layout."
                    ),
                    location="NDArrayStrides.__init__(shape, order)",
                )
            )

    @always_inline("nodebug")
    fn __init__(out self, *shape: Int, order: String) raises:
        """
        Overloads the function `__init__(shape: NDArrayStrides, order: String)`.
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
        Overloads the function `__init__(shape: NDArrayStrides, order: String)`.
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
        Overloads the function `__init__(shape: NDArrayStrides, order: String)`.
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
                ValueError(
                    message=String(
                        "Number of dimensions must be non-negative, got {}."
                    ).format(ndim),
                    suggestion="Provide ndim >= 0.",
                    location="NDArrayStrides.__init__(ndim, initialized)",
                )
            )

        if ndim == 0:
            # This is a 0darray (numojo scalar)
            self.ndim = ndim
            self._buf = UnsafePointer[Scalar[Self._type]]()

        else:
            self.ndim = ndim
            self._buf = UnsafePointer[Scalar[Self._type]]().alloc(ndim)
            if initialized:
                for i in range(ndim):
                    (self._buf + i).init_pointee_copy(0)

    @staticmethod
    fn row_major(shape: NDArrayShape) raises -> NDArrayStrides:
        """
        Create row-major (C-style) strides from a shape.

        Row-major means the last dimension has stride 1 and strides increase
        going backwards through dimensions.

        Args:
            shape: The shape of the array.

        Returns:
            A new NDArrayStrides object with row-major memory layout.

        Example:
        ```mojo
        import numojo as nm
        var shape = nm.Shape(2, 3, 4)
        var strides = nm.Strides.row_major(shape)
        print(strides)  # Strides: (12, 4, 1)
        ```
        """
        return NDArrayStrides(shape=shape, order="C")

    @staticmethod
    fn col_major(shape: NDArrayShape) raises -> NDArrayStrides:
        """
        Create column-major (Fortran-style) strides from a shape.

        Column-major means the first dimension has stride 1 and strides increase
        going forward through dimensions.

        Args:
            shape: The shape of the array.

        Returns:
            A new NDArrayStrides object with column-major memory layout.

        Example:
        ```mojo
        import numojo as nm
        var shape = nm.Shape(2, 3, 4)
        var strides = nm.Strides.col_major(shape)
        print(strides)  # Strides: (1, 2, 6)
        ```
        """
        return NDArrayStrides(shape=shape, order="F")

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """
        Initializes the NDArrayStrides from another strides.
        A deep-copy of the elements is conducted.

        Args:
            other: Strides of the array.
        """
        self.ndim = other.ndim
        self._buf = UnsafePointer[Scalar[Self._type]]().alloc(other.ndim)
        memcpy(dest=self._buf, src=other._buf, count=other.ndim)

    fn __del__(deinit self):
        """
        Destructor for NDArrayStrides.
        Frees the allocated memory for the data buffer.
        """
        if self.ndim > 0:
            self._buf.free()

    fn normalize_index(self, index: Int) -> Int:
        """
        Normalizes the given index to be within the valid range.

        Args:
            index: The index to normalize.

        Returns:
            The normalized index.
        """
        var normalized_idx: Int = index
        if normalized_idx < 0:
            normalized_idx += self.ndim
        return normalized_idx

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
        if index >= self.ndim or index < -self.ndim:
            raise Error(
                IndexError(
                    message=String("Index {} out of range [{}, {}).").format(
                        index, -self.ndim, self.ndim
                    ),
                    suggestion="Use indices in [-ndim, ndim).",
                    location="NDArrayStrides.__getitem__",
                )
            )
        var normalized_idx: Int = self.normalize_index(index)
        return Int(self._buf[normalized_idx])

    @always_inline("nodebug")
    fn _compute_slice_params(
        self, slice_index: Slice
    ) raises -> Tuple[Int, Int, Int]:
        """
        Compute normalized slice parameters (start, step, length).

        Args:
            slice_index: The slice to compute parameters for.

        Returns:
            A tuple of (start, step, length).

        Raises:
            Error: If the slice step is zero.
        """
        var n: Int = self.ndim
        if n == 0:
            return (0, 1, 0)

        var step = slice_index.step.or_else(1)
        if step == 0:
            raise Error(
                ValueError(
                    message="Slice step cannot be zero.",
                    suggestion="Use a non-zero step value.",
                    location="NDArrayStrides._compute_slice_params",
                )
            )

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
    fn __getitem__(self, slice_index: Slice) raises -> NDArrayStrides:
        """
        Return a sliced view of the strides as a new NDArrayStrides.
        Delegates normalization & validation to _compute_slice_params.

        Args:
            slice_index: The slice to extract.

        Returns:
            A new NDArrayStrides containing the sliced values.

        Example:
        ```mojo
        import numojo as nm
        var strides = nm.Strides(12, 4, 1)
        print(strides[1:])  # Strides: (4, 1)
        ```
        """
        var updated_slice: Tuple[Int, Int, Int] = self._compute_slice_params(
            slice_index
        )
        var start = updated_slice[0]
        var step = updated_slice[1]
        var length = updated_slice[2]

        if length <= 0:
            var empty_result = NDArrayStrides(ndim=0, initialized=False)
            return empty_result

        var result = NDArrayStrides(ndim=length, initialized=False)
        var idx = start
        for i in range(length):
            (result._buf + i).init_pointee_copy(self._buf[idx])
            idx += step
        return result^

    @always_inline("nodebug")
    fn __setitem__(mut self, index: Int, val: Scalar[Self._type]) raises:
        """
        Sets stride at specified index.

        Raises:
           Error: Index out of bound.
           Error: Value is not positive.

        Args:
          index: Index to get the stride.
          val: Value to set at the given index.
        """
        if index >= self.ndim or index < -self.ndim:
            raise Error(
                IndexError(
                    message=String("Index {} out of range [{}, {}).").format(
                        index, -self.ndim, self.ndim
                    ),
                    suggestion="Use indices in [-ndim, ndim).",
                    location=(
                        "NDArrayStrides.__setitem__(index: Int, val:"
                        " Scalar[DType.int])"
                    ),
                )
            )
        var normalized_idx: Int = self.normalize_index(index)
        self._buf[normalized_idx] = val

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

    fn __iter__(self) raises -> _StrideIter:
        """
        Iterate over elements of the NDArrayStrides, returning copied values.

        Returns:
            An iterator of NDArrayStrides elements.

        Example:
        ```mojo
        import numojo as nm
        var strides = nm.Strides(12, 4, 1)
        for stride in strides:
            print(stride)  # Prints: 12, 4, 1
        ```
        """
        return _StrideIter(
            strides=self,
            length=self.ndim,
        )

    # ===-------------------------------------------------------------------===#
    # Other methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn copy(read self) raises -> Self:
        """
        Returns a deep copy of the strides.
        """

        var res = Self(ndim=self.ndim, initialized=False)
        memcpy(dest=res._buf, src=self._buf, count=self.ndim)
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
        var norm_axis1: Int = self.normalize_index(axis1)
        var norm_axis2: Int = self.normalize_index(axis2)

        if norm_axis1 < 0 or norm_axis1 >= self.ndim:
            raise Error(
                IndexError(
                    message=String("axis1 {} out of range [0, {}).").format(
                        norm_axis1, self.ndim
                    ),
                    suggestion="Provide axis1 in [-ndim, ndim).",
                    location="NDArrayStrides.swapaxes",
                )
            )
        if norm_axis2 < 0 or norm_axis2 >= self.ndim:
            raise Error(
                IndexError(
                    message=String("axis2 {} out of range [0, {}).").format(
                        norm_axis2, self.ndim
                    ),
                    suggestion="Provide axis2 in [-ndim, ndim).",
                    location="NDArrayStrides.swapaxes",
                )
            )

        var res = Self(ndim=self.ndim, initialized=False)
        memcpy(dest=res._buf, src=self._buf, count=self.ndim)
        res[axis1] = self[axis2]
        res[axis2] = self[axis1]
        return res^

    fn join(self, *strides: Self) raises -> Self:
        """
        Join multiple strides into a single strides.

        Args:
            strides: Variable number of NDArrayStrides objects.

        Returns:
            A new NDArrayStrides object with all values concatenated.

        Example:
        ```mojo
        import numojo as nm
        var s1 = nm.Strides(12, 4)
        var s2 = nm.Strides(1)
        var joined = s1.join(s2)
        print(joined)  # Strides: (12, 4, 1)
        ```
        """
        var total_dims: Int = self.ndim
        for i in range(len(strides)):
            total_dims += strides[i].ndim

        var new_strides: Self = Self(ndim=total_dims, initialized=False)

        var index: Int = 0
        for i in range(self.ndim):
            (new_strides._buf + index).init_pointee_copy(self[i])
            index += 1

        for i in range(len(strides)):
            for j in range(strides[i].ndim):
                (new_strides._buf + index).init_pointee_copy(strides[i][j])
                index += 1

        return new_strides

    # ===-------------------------------------------------------------------===#
    # Other private methods
    # ===-------------------------------------------------------------------===#

    fn _extend(self, *values: Int) raises -> Self:
        """
        Extend the strides by additional values.
        ***UNSAFE!*** No boundary check!

        Args:
            values: Additional stride values to append.

        Returns:
            A new NDArrayStrides object with the extended values.

        Example:
        ```mojo
        import numojo as nm
        var strides = nm.Strides(12, 4)
        var extended = strides._extend(1)
        print(extended)  # Strides: (12, 4, 1)
        ```
        """
        var total_dims: Int = self.ndim + len(values)
        var new_strides: Self = Self(ndim=total_dims, initialized=False)

        var offset: UInt = 0
        for i in range(self.ndim):
            (new_strides._buf + offset).init_pointee_copy(self[i])
            offset += 1
        for value in values:
            (new_strides._buf + offset).init_pointee_copy(value)
            offset += 1

        return new_strides^

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

    fn _move_axis_to_end(self, var axis: Int) -> Self:
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

    fn load[width: Int = 1](self, idx: Int) raises -> SIMD[Self._type, width]:
        """
        Load a SIMD vector from the Strides at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to load from.

        Returns:
            A SIMD vector containing the loaded values.

        Raises:
            Error: If the load exceeds the bounds of the Strides.
        """
        if idx < 0 or idx + width > self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Load operation out of bounds: idx={} width={} ndim={}"
                    ).format(idx, width, self.ndim),
                    suggestion=(
                        "Ensure that idx and width are within valid range."
                    ),
                    location="Strides.load",
                )
            )

        return self._buf.load[width=width](idx)

    fn store[
        width: Int = 1
    ](self, idx: Int, value: SIMD[Self._type, width]) raises:
        """
        Store a SIMD vector into the Strides at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to store to.
            value: The SIMD vector to store.

        Raises:
            Error: If the store exceeds the bounds of the Strides.
        """
        if idx < 0 or idx + width > self.ndim:
            raise Error(
                IndexError(
                    message=String(
                        "Store operation out of bounds: idx={} width={} ndim={}"
                    ).format(idx, width, self.ndim),
                    suggestion=(
                        "Ensure that idx and width are within valid range."
                    ),
                    location="Strides.store",
                )
            )

        self._buf.store[width=width](idx, value)

    fn unsafe_load[width: Int = 1](self, idx: Int) -> SIMD[Self._type, width]:
        """
        Unsafely load a SIMD vector from the Strides at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to load from.

        Returns:
            A SIMD vector containing the loaded values.
        """
        return self._buf.load[width=width](idx)

    fn unsafe_store[
        width: Int = 1
    ](self, idx: Int, value: SIMD[Self._type, width]):
        """
        Unsafely store a SIMD vector into the Strides at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to store to.
            value: The SIMD vector to store.
        """
        self._buf.store[width=width](idx, value)


struct _StrideIter[
    forward: Bool = True,
](ImplicitlyCopyable, Movable):
    """Iterator for NDArrayStrides.

    Parameters:
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var strides: NDArrayStrides
    var length: Int

    fn __init__(
        out self,
        strides: NDArrayStrides,
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.strides = strides

    fn __iter__(self) -> Self:
        return self

    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index > 0

    fn __next__(mut self) raises -> Scalar[DType.int]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.strides.__getitem__(current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.strides.__getitem__(current_index)

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index
