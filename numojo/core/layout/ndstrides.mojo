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
from memory import UnsafePointer
from numojo.core.indexing.index_buffer import IndexBuffer
from numojo.core.layout.ndshape import NDArrayShape

from numojo.core.error import NumojoError


@register_passable
struct NDArrayStrides(
    Equatable,
    ImplicitlyCopyable,
    Movable,
    Representable,
    Sized,
    Stringable,
    Writable,
):
    """
    Presents the strides of `NDArray` type.

    The data buffer of the NDArrayStrides is a series of `Int` on memory.
    The number of elements in the strides must be positive.
    The number of dimension is checked upon creation of the strides.
    """

    # ===----------------------------------------------------------------------=== #
    # Aliases
    # ===----------------------------------------------------------------------=== #

    comptime element_type: DType = DType.int
    """The data type of the NDArrayStrides elements."""

    # ===----------------------------------------------------------------------=== #
    # Fields
    # ===----------------------------------------------------------------------=== #

    var _buf: IndexBuffer
    """Data buffer."""
    var ndim: Int
    """Number of dimensions of array. It must be larger than 0."""

    # ===----------------------------------------------------------------------=== #
    # Lifecycle Methods
    # ===----------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __init__(out self):
        """
        Initializes an empty NDArrayStrides.
        """
        self.ndim = 0
        self._buf = IndexBuffer()

    @always_inline("nodebug")
    fn __init__(out self, buf: IndexBuffer):
        """
        Initializes the NDArrayStrides from an IndexBuffer.

        Args:
            buf: The IndexBuffer to initialize from.
        """
        self.ndim = buf.ndim
        self._buf = buf

    @always_inline("nodebug")
    fn __init__(out self, *strides: Int) raises:
        """
        Initializes the NDArrayStrides from strides.

        Args:
            strides: Strides of the array.

        Raises:
           Error: If the number of dimensions is not positive.
        """
        self.ndim = len(strides)
        if self.ndim <= 0:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Number of dimensions must be positive, got {}. Provide"
                        " at least one stride value.".format(self.ndim)
                    ),
                    location="NDArrayStrides.__init__(*strides: Int)",
                )
            )

        self._buf = IndexBuffer(size=self.ndim)
        for i in range(self.ndim):
            self._buf.init_value(i, strides[i])

    @always_inline("nodebug")
    fn __init__(out self, strides: List[Int]) raises:
        """
        Initializes the NDArrayStrides from a list of strides.

        Args:
            strides: Strides of the array.

        Raises:
           Error: If the number of dimensions is not positive.
        """
        self.ndim = len(strides)
        if self.ndim <= 0:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Number of dimensions must be positive, got {}. Provide"
                        " a non-empty list of strides.".format(self.ndim)
                    ),
                    location="NDArrayStrides.__init__(strides: List[Int])",
                )
            )

        self._buf = IndexBuffer(size=self.ndim)
        for i in range(self.ndim):
            self._buf.init_value(i, strides[i])

    @always_inline("nodebug")
    fn __init__(out self, strides: VariadicList[Int]) raises:
        """
        Initializes the NDArrayStrides from a variadic list of strides.

        Args:
            strides: Strides of the array.

        Raises:
           Error: If the number of dimensions is not positive.
        """
        self.ndim = len(strides)
        if self.ndim <= 0:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Number of dimensions must be positive, got {}. Provide"
                        " a non-empty variadic list of strides.".format(
                            self.ndim
                        )
                    ),
                    location=(
                        "NDArrayStrides.__init__(strides: VariadicList[Int])"
                    ),
                )
            )

        self._buf = IndexBuffer(size=self.ndim)
        for i in range(self.ndim):
            self._buf.init_value(i, strides[i])

    @always_inline("nodebug")
    fn __init__(out self, strides: NDArrayStrides):
        """
        Initializes the NDArrayStrides from another strides.
        A deep-copy of the elements is conducted.

        Args:
            strides: Strides of the array.
        """
        self.ndim = strides.ndim
        self._buf = IndexBuffer(size=self.ndim)
        memcpy(
            dest=self._buf.ptr,
            src=strides._buf.ptr,
            count=strides.ndim,
        )

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: NDArrayShape,
        order: String = "C",
    ) raises:
        """
        Initializes the NDArrayStrides from a shape and an order.

        Args:
            shape: Shape of the array.
            order: Order of the memory layout
                (row-major "C" or column-major "F").
                Default is "C".

        Raises:
            ValueError: If the order argument is not `C` or `F`.
        """
        self.ndim = shape.ndim
        self._buf = IndexBuffer(size=shape.ndim)

        if order == "C":
            var temp = 1
            for i in range(self.ndim - 1, -1, -1):
                self._buf.store(i, temp)
                temp *= Int(shape[i])
        elif order == "F":
            var temp = 1
            for i in range(0, self.ndim):
                self._buf.store(i, temp)
                temp *= Int(shape[i])
        else:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Invalid order '{}'; expected 'C' or 'F'. Use 'C' for"
                        " row-major or 'F' for column-major layout.".format(
                            order
                        )
                    ),
                    location="NDArrayStrides.__init__(shape, order)",
                )
            )

    @always_inline("nodebug")
    fn __init__(out self, *shape: Int, order: String) raises:
        """
        Overloads the function `__init__(shape: NDArrayStrides, order: String)`.
        Initializes the NDArrayStrides from a given shapes and an order.

        Args:
            shape: Shape of the array.
            order: Order of the memory layout
                (row-major "C" or column-major "F").

        Raises:
            ValueError: If the order argument is not `C` or `F`.
        """
        self = Self(shape=NDArrayShape(shape), order=order)

    @always_inline("nodebug")
    fn __init__(out self, shape: List[Int], order: String = "C") raises:
        """
        Overloads the function `__init__(shape: NDArrayStrides, order: String)`.
        Initializes the NDArrayStrides from a given shapes and an order.

        Args:
            shape: Shape of the array.
            order: Order of the memory layout
                (row-major "C" or column-major "F").

        Raises:
            ValueError: If the order argument is not `C` or `F`.
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

        Args:
            shape: Shape of the array.
            order: Order of the memory layout
                (row-major "C" or column-major "F").

        Raises:
            ValueError: If the order argument is not `C` or `F`.
        """
        self = Self(shape=NDArrayShape(shape), order=order)

    @always_inline("nodebug")
    fn __init__(
        out self,
        *,
        ndim: Int,
        initialized: Bool,
    ) raises:
        """
        Construct NDArrayStrides with number of dimensions.
        This method is useful when you want to create a strides with given ndim
        without knowing the strides values.
        `ndim == 0` is allowed in this method for 0darray (numojo scalar).

        Args:
            ndim: Number of dimensions.
            initialized: Whether the strides is initialized.
                If yes, the values will be set to 0.
                If no, the values will be uninitialized.

        Raises:
           Error: If the number of dimensions is negative.
        """
        if ndim < 0:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Number of dimensions must be non-negative, got {}."
                        " Provide ndim >= 0.".format(ndim)
                    ),
                    location="NDArrayStrides.__init__(ndim, initialized)",
                )
            )

        if ndim == 0:
            # This is a 0darray (numojo scalar)
            self.ndim = ndim
            self._buf = IndexBuffer(size=1)
            self._buf.init_value(0, 0)
        else:
            self.ndim = ndim
            self._buf = IndexBuffer(size=ndim)
            if initialized:
                for i in range(ndim):
                    self._buf.init_value(i, 0)

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """
        Initializes the NDArrayStrides from another strides.
        A deep-copy of the elements is conducted.

        Args:
            other: Strides of the array.
        """
        self.ndim = other.ndim
        if other.ndim == 0:
            self._buf = IndexBuffer(size=1)
            self._buf.init_value(0, 0)
        else:
            self._buf = IndexBuffer(size=other.ndim)
            memcpy(
                dest=self._buf.ptr,
                src=other._buf.ptr,
                count=other.ndim,
            )

    # ===----------------------------------------------------------------------=== #
    # Element Access Methods
    # ===----------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        """
        Gets stride at specified index.

        Args:
          index: Index to get the stride.

        Returns:
           Stride value at the given index.

        Raises:
           Error: Index out of bound.
        """
        return Int(self._buf[index])

    @always_inline("nodebug")
    fn __getitem__(self, slice_index: Slice) raises -> NDArrayStrides:
        """
        Return a sliced view of the strides as a new NDArrayStrides.

        Args:
            slice_index: Slice object defining the sub-buffer.

        Returns:
            A new NDArrayStrides representing the sliced strides.
        """
        return Self(self._buf[slice_index])

    @always_inline("nodebug")
    fn __setitem__(mut self, index: Int, val: Scalar[Self.element_type]) raises:
        """
        Sets stride at specified index.

        Args:
          index: Index to set the stride.
          val: Value to set at the given index.

        Raises:
           Error: Index out of bound.
        """
        self._buf[index] = val

    fn load[
        width: Int = 1
    ](self, idx: Int) raises -> SIMD[Self.element_type, width]:
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
                NumojoError(
                    category="index",
                    message=(
                        "Load operation out of bounds: idx={} width={} ndim={}."
                        " Ensure that idx and width are within valid range."
                        .format(idx, width, self.ndim)
                    ),
                    location="Strides.load",
                )
            )
        return self._buf.load[width=width](idx)

    fn store[
        width: Int = 1
    ](self, idx: Int, value: SIMD[Self.element_type, width]) raises:
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
                NumojoError(
                    category="index",
                    message=(
                        "Store operation out of bounds: idx={} width={}"
                        " ndim={}. Ensure that idx and width are within valid"
                        " range.".format(idx, width, self.ndim)
                    ),
                    location="Strides.store",
                )
            )
        self._buf.store[width=width](idx, value)

    fn unsafe_load[
        width: Int = 1
    ](self, idx: Int) -> SIMD[Self.element_type, width]:
        """
        Unsafely load a SIMD vector from the Strides at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to load from.

        Returns:
            A SIMD vector containing the loaded values.
        """
        return self._buf.unsafe_load[width=width](idx)

    fn unsafe_store[
        width: Int = 1
    ](self, idx: Int, value: SIMD[Self.element_type, width]):
        """
        Unsafely store a SIMD vector into the Strides at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to store to.
            value: The SIMD vector to store.
        """
        self._buf.unsafe_store[width=width](idx, value)

    # ===----------------------------------------------------------------------=== #
    # Transformation Methods
    # ===----------------------------------------------------------------------=== #

    fn permute(self, axes: List[Int]) raises -> Self:
        """
        Return new strides with axes reordered.

        Args:
            axes: New axis order. Must contain each axis exactly once.

        Returns:
            A new NDArrayStrides with axes permuted.

        Raises:
            Error: If axes length doesn't match ndim or contains invalid/duplicate axes.
        """
        if len(axes) != self.ndim:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "axes length {} does not match ndim {}. Provide a"
                        " permutation of all axes."
                    ).format(len(axes), self.ndim),
                    location="NDArrayStrides.permute",
                )
            )

        var normalized = IndexBuffer(size=self.ndim)
        for i in range(self.ndim):
            var axis = axes[i]
            if axis < 0:
                axis += self.ndim
            if axis < 0 or axis >= self.ndim:
                raise Error(
                    NumojoError(
                        category="index",
                        message=(
                            "axis {} out of range [0, {}). Provide axes in"
                            " [-ndim, ndim)."
                        ).format(axis, self.ndim),
                        location="NDArrayStrides.permute",
                    )
                )
            normalized.init_value(i, axis)

        for i in range(self.ndim):
            for j in range(i + 1, self.ndim):
                if normalized[i] == normalized[j]:
                    raise Error(
                        NumojoError(
                            category="index",
                            message=(
                                "axes must be a permutation; duplicate axis"
                                " {} found."
                            ).format(normalized[i]),
                            location="NDArrayStrides.permute",
                        )
                    )

        var result = NDArrayStrides(ndim=self.ndim, initialized=False)
        for i in range(self.ndim):
            result._buf.init_value(i, self._buf[Int(normalized[i])])
        return result^

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
        var val1 = res[axis1]
        var val2 = res[axis2]
        res[axis1] = val2
        res[axis2] = val1
        return res

    fn join(self, *strides: Self) -> Self:
        """
        Join multiple strides into a single strides.

        Args:
            strides: Variable number of NDArrayStrides objects.

        Returns:
            A new NDArrayStrides object with all values concatenated.
        """
        var bufs = List[IndexBuffer]()
        for i in range(len(strides)):
            bufs.append(strides[i]._buf)
        return Self(self._buf.join(bufs))

    fn extend(self, *values: Int) -> Self:
        """
        Extend the shape by sizes of extended dimensions.

        Args:
            values: Sizes of extended dimensions.

        Returns:
            A new NDArrayShape object.
        """
        var new_vals = List[Int]()
        for i in range(self.ndim):
            new_vals.append(values[i])
        return Self(self._buf.extend(new_vals))

    fn flip(mut self):
        """
        Flip the items in-place.
        """
        self._buf.flip()

    fn flipped(self) -> Self:
        """
        Returns a new strides by flipping the items.

        Returns:
            A new strides with the items flipped.
        """
        return Self(self._buf.flipped())

    fn move_axis_to_end(self, axis: Int) -> Self:
        """
        Returns a new strides by moving the value of axis to the end.

        Args:
            axis: The axis (index) to move.

        Returns:
            A new strides with the axis moved to the end.
        """
        return Self(self._buf.move_axis_to_end(axis))

    fn pop(self, axis: Int) raises -> Self:
        """
        Drops information of certain axis.

        Args:
            axis: The axis (index) to drop.

        Returns:
            A new stride with the item at the given axis (index) dropped.
        """
        return Self(self._buf.pop(axis))

    # ===----------------------------------------------------------------------=== #
    # Properties
    # ===----------------------------------------------------------------------=== #

    fn is_contiguous(self, shape: NDArrayShape) raises -> Bool:
        """
        Check if strides represent a contiguous layout for the shape.

        Args:
            shape: The shape of the array.

        Returns:
            True if the strides are contiguous.
        """
        if shape.ndim != self.ndim:
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "shape rank {} does not match strides rank {}. Provide"
                        " matching shape and strides."
                    ).format(shape.ndim, self.ndim),
                    location="NDArrayStrides.is_contiguous",
                )
            )
        if self.ndim == 0:
            return True
        var expected = 1
        for i in range(self.ndim - 1, -1, -1):
            if shape[i] > 1 and self[i] != expected:
                return False
            expected *= Int(shape[i])
        return True

    # ===----------------------------------------------------------------------=== #
    # Traits
    # ===----------------------------------------------------------------------=== #

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
            result += String(self._buf.unsafe_load(i))
            if i < self.ndim - 1:
                result += ", "
        result = result + ")"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        """
        Writes the strides representation to a writer.
        """
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
        return self._buf == other._buf

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """
        Checks if two strides have identical dimensions and values.

        Args:
            other: The strides to compare with.

        Returns:
           True if both strides do not have identical dimensions or values.
        """
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, val: Int) -> Bool:
        """
        Checks if the given value is present in the strides.

        Args:
            val: The value to search for.

        Returns:
          True if the given value is present in the strides.
        """
        return val in self._buf

    @always_inline("nodebug")
    fn __contains__(self, val: Scalar[Self.element_type]) -> Bool:
        """
        Check if the NDArrayStrides contains the given value.

        Args:
            val: Value to check for.

        Returns:
            True if the value is in the Item, False otherwise.
        """
        return val in self._buf

    # ===----------------------------------------------------------------------=== #
    # Static Methods
    # ===----------------------------------------------------------------------=== #

    @staticmethod
    fn row_major(shape: NDArrayShape) raises -> NDArrayStrides:
        """
        Create row-major (C-style) strides from a shape.

        Args:
            shape: The shape of the array.

        Returns:
            A new NDArrayStrides object with row-major memory layout.
        """
        return NDArrayStrides(shape=shape, order="C")

    @staticmethod
    fn col_major(shape: NDArrayShape) raises -> NDArrayStrides:
        """
        Create column-major (Fortran-style) strides from a shape.

        Args:
            shape: The shape of the array.

        Returns:
            A new NDArrayStrides object with column-major memory layout.
        """
        return NDArrayStrides(shape=shape, order="F")

    @staticmethod
    fn default(shape: NDArrayShape) raises -> NDArrayStrides:
        """
        Create default (row-major) strides from a shape.

        Args:
            shape: The shape of the array.

        Returns:
            A new NDArrayStrides object with default memory layout.
        """
        return NDArrayStrides(shape=shape, order="C")

    # ===----------------------------------------------------------------------=== #
    # Utility Methods
    # ===----------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn tolist(self) -> List[Int]:
        """
        Convert the strides to a list of integers.

        Returns:
            A list containing the stride values.
        """
        var res = List[Int]()
        for i in range(self.ndim):
            res.append(Int(self._buf.unsafe_load(i)))
        return res^

    @always_inline("nodebug")
    fn normalize_index(self, index: Int) -> Int:
        """
        Normalizes the given index to be within the valid range [0, ndim).

        Args:
            index: The index to normalize.

        Returns:
            The normalized index.
        """
        var normalized_idx: Int = index
        if normalized_idx < 0:
            normalized_idx += self.ndim
        return normalized_idx

    # ===----------------------------------------------------------------------=== #
    # Iterators
    # ===----------------------------------------------------------------------=== #
    fn __iter__(ref self) -> _StrideIter[origin_of(self), True]:
        """
        Iterate over elements of the NDArrayStrides, returning copied values.

        Returns:
            An iterator of NDArrayStrides elements.
        """
        return _StrideIter[origin_of(self), True](
            strides=Pointer(to=self),
            length=self.ndim,
        )

    fn __reversed__(ref self) -> _StrideIter[origin_of(self), False]:
        """
        Iterate over elements of the NDArrayStrides, returning copied values.

        Returns:
            An iterator of NDArrayStrides elements.
        """
        return _StrideIter[origin_of(self), False](
            strides=Pointer(to=self),
            length=self.ndim,
        )


struct _StrideIter[
    origin: ImmutOrigin = ImmutExternalOrigin,
    forward: Bool = True,
](ImplicitlyCopyable, Movable):
    """Iterator for NDArrayStrides.

    Parameters:
        origin: The origin of the iterator.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var strides: Pointer[NDArrayStrides, Self.origin]
    var length: Int

    fn __init__(
        out self,
        strides: Pointer[NDArrayStrides, Self.origin],
        length: Int,
    ):
        self.index = 0 if Self.forward else length - 1
        self.length = length
        self.strides = strides

    fn __iter__(self) -> Self:
        return self

    fn __has_next__(self) -> Bool:
        @parameter
        if Self.forward:
            return self.index < self.length
        else:
            return self.index >= 0

    fn __next__(mut self) raises -> Scalar[DType.int]:
        @parameter
        if Self.forward:
            var current_index = self.index
            self.index += 1
            return self.strides[].__getitem__(current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.strides[].__getitem__(current_index)

    fn __len__(self) -> Int:
        @parameter
        if Self.forward:
            return self.length - self.index
        else:
            return self.index + 1
