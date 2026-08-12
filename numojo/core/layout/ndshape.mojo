# ===----------------------------------------------------------------------=== #
# NuMojo: NDArrayShape type
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""NDArrayShape (numojo.core.layout.ndshape)
--------------------------------------------
Implements NDArrayShape type representing the shape of an NDArray.
The shape is stored as a contiguous buffer of integers, with the number of dimensions (ndim) tracked separately.
The NDArrayShape provides methods for element access, shape transformations (e.g., permute, reverse),
and properties like size and rank.
"""

from std.memory import unsafe_memcpy, unsafe_memcmp, UnsafePointer

from numojo.core.indexing.index_buffer import IndexBuffer
from numojo.core.layout.ndstrides import NDArrayStrides
from numojo.core.error import NumojoError


struct NDArrayShape(
    Equatable,
    ImplicitlyCopyable,
    Movable,
    RegisterPassable,
    Sized,
    Writable,
):
    """
    Presents the shape of `NDArray` type.

    The data buffer of the NDArrayShape is a series of `Int` on memory.
    The number of elements in the shape must be positive.
    The elements of the shape must be non-negative.
    The number of dimension and values of elements are checked upon
    creation of the shape.

    Example:
    ```mojo
    import numojo as nm
    var shape1 = nm.Shape(2, 3, 4)
    print(shape1)  # Shape: (2, 3, 4)
    var shape2 = nm.Shape([5, 6, 7])
    print(shape2)  # Shape: (5, 6, 7)
    ```
    """

    # ===----------------------------------------------------------------------=== #
    # Aliases
    # ===----------------------------------------------------------------------=== #

    comptime element_type: DType = DType.int
    """The data type of the NDArrayShape elements."""

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
    def __init__(out self):
        """
        Initializes an empty NDArrayShape.
        """
        self.ndim = 0
        self._buf = IndexBuffer()

    @always_inline("nodebug")
    def __init__(out self, var buf: IndexBuffer):
        """
        Initializes the NDArrayShape from an IndexBuffer.

        Args:
            buf: The IndexBuffer to initialize from.
        """
        self.ndim = buf.ndim
        self._buf = buf^

    @always_inline("nodebug")
    def __init__(out self, *shape: Int) raises:
        """
        Initializes the NDArrayShape with variable shape dimensions.

        Args:
            shape: Variable number of integers representing the shape dimensions.

        Raises:
           Error: If any shape dimension is negative.
        """
        self.ndim = len(shape)
        self._buf = IndexBuffer(size=len(shape))
        for i in range(len(shape)):
            if shape[i] < 0:
                raise Error(
                    NumojoError(
                        category="shape",
                        message=(
                            "Shape dimension at index {} must be non-negative,"
                            " got {}. Use non-negative integers for all shape"
                            " dimensions."
                        ).format(i, shape[i]),
                        location="NDArrayShape.__init__(*shape: Int)",
                    )
                )
            self._buf.init_value(i, Scalar[DType.int](shape[i]))

    @always_inline("nodebug")
    def __init__(out self, shape: List[Int]) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions.

        Args:
            shape: A list of integers representing the shape dimensions.

        Raises:
            Error: If the number of dimensions is not positive.
            Error: If any shape dimension is negative.
        """
        self.ndim = len(shape)
        if self.ndim <= 0:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Number of dimensions must be positive, got {}. Provide"
                        " at least one shape dimension."
                    ).format(self.ndim),
                    location="NDArrayShape.__init__(shape: List[Int])",
                )
            )
        self._buf = IndexBuffer(size=self.ndim)
        for i in range(self.ndim):
            if shape[i] < 0:
                raise Error(
                    NumojoError(
                        category="shape",
                        message=(
                            "Shape dimension at index {} must be non-negative,"
                            " got {}. Use non-negative integers for all shape"
                            " dimensions."
                        ).format(i, shape[i]),
                        location="NDArrayShape.__init__(shape: List[Int])",
                    )
                )
            self._buf.init_value(i, Scalar[DType.int](shape[i]))

    @always_inline("nodebug")
    def __init__(out self, shape: VariadicList[Int, _]) raises:
        """
        Initializes the NDArrayShape with a list of shape dimensions.

        Args:
            shape: A variadic list of integers representing the shape dimensions.

        Raises:
            Error: If the number of dimensions is not positive.
            Error: If any shape dimension is negative.
        """
        self.ndim = len(shape)
        if self.ndim <= 0:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Number of dimensions must be positive, got {}. Provide"
                        " at least one shape dimension."
                    ).format(self.ndim),
                    location="NDArrayShape.__init__(shape: VariadicList[Int])",
                ),
            )

        self._buf = IndexBuffer(size=self.ndim)
        for i in range(self.ndim):
            if shape[i] < 0:
                raise Error(
                    NumojoError(
                        category="shape",
                        message=(
                            "Shape dimension at index {} must be non-negative,"
                            " got {}. Use non-negative integers for all shape"
                            " dimensions."
                        ).format(i, shape[i]),
                        location=(
                            "NDArrayShape.__init__(shape: VariadicList[Int])"
                        ),
                    )
                )
            self._buf.init_value(i, Scalar[DType.int](shape[i]))

    @always_inline("nodebug")
    def __init__(out self, shape: NDArrayShape):
        """
        Initializes the NDArrayShape from another NDArrayShape.
        A deep copy of the data buffer is conducted.

        Args:
            shape: Another NDArrayShape to initialize from.
        """
        self.ndim = shape.ndim
        self._buf = IndexBuffer(size=self.ndim)
        unsafe_memcpy(dest=self._buf.ptr, src=shape._buf.ptr, count=shape.ndim)

    @always_inline("nodebug")
    def __init__(
        out self,
        *,
        ndim: Int,
        initialized: Bool,
    ) raises:
        """
        Construct NDArrayShape with number of dimensions.
        This method is useful when you want to create a shape with given ndim
        without knowing the shape values.
        `ndim == 0` is allowed in this method for 0darray (numojo scalar).

        Args:
            ndim: Number of dimensions.
            initialized: Whether the shape is initialized.
                If yes, the values will be set to 1.
                If no, the values will be uninitialized.

        Raises:
           Error: If the number of dimensions is negative.

        Note:
            After creating the shape with uninitialized values,
            you must set the values before using it! Otherwise, it may lead to undefined behavior.
        """
        if ndim < 0:
            raise Error(
                NumojoError(
                    category="value",
                    message=(
                        "Number of dimensions must be non-negative, got {}."
                        " Provide ndim >= 0.".format(ndim)
                    ),
                    location="NDArrayShape.__init__(ndim, initialized)",
                )
            )

        if ndim == 0:
            # This denotes a 0darray (numojo scalar)
            self.ndim = ndim
            self._buf = IndexBuffer(
                size=1
            )  # allocate 1 element to avoid null pointer
            self._buf.init_value(0, 0)
        else:
            self.ndim = ndim
            self._buf = IndexBuffer(size=ndim)
            if initialized:
                for i in range(ndim):
                    self._buf.init_value(i, 1)

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        """
        Initializes the NDArrayShape from ancopy NDArrayShape.
        A deep copy of the data buffer is conducted.

        Args:
            copy: Ancopy NDArrayShape to initialize from.
        """
        self.ndim = copy.ndim
        if copy.ndim == 0:
            self._buf = IndexBuffer(size=1)
            self._buf.init_value(0, 0)
        else:
            self._buf = IndexBuffer(size=copy.ndim)
            unsafe_memcpy(
                dest=self._buf.ptr,
                src=copy._buf.ptr,
                count=copy.ndim,
            )

    # ===----------------------------------------------------------------------=== #
    # Element Access Methods
    # ===----------------------------------------------------------------------=== #


    @always_inline("nodebug")
    def __getitem__(
        self, index: Scalar[Self.element_type]
    ) raises -> Scalar[Self.element_type]:
        """
        Gets shape dimension at specified index.

        Args:
          index: Index to get the shape.

        Returns:
           Shape value at the given index.
        """
        return self._buf[index]

    @always_inline("nodebug")
    def __getitem__(self, slice_index: Slice) raises -> NDArrayShape:
        """
        Return a sliced view of the dimension tuple as a new NDArrayShape.

        Args:
            slice_index: Slice object defining the sub-buffer.

        Returns:
            A new NDArrayShape representing the sliced dimensions.
        """
        return Self(self._buf[slice_index])

    @always_inline("nodebug")
    def __setitem__(
        mut self,
        index: Scalar[Self.element_type],
        val: Scalar[Self.element_type],
    ) raises:
        """
        Sets shape at specified index.

        Args:
          index: Index to set the shape.
          val: Value to set at the given index.

        Raises:
           Error: Index out of bound.
        """
        self._buf[index] = val


    def load[
        width: Int = 1
    ](self, idx: Int) raises -> SIMD[Self.element_type, width]:
        """
        Load a SIMD vector from the Shape at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to load from.

        Returns:
            A SIMD vector containing the loaded values.

        Raises:
            Error: If the load exceeds the bounds of the Shape.
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
                    location="Shape.load",
                )
            )
        return self._buf.ptr.unsafe_load[width=width](idx)

    def store[
        width: Int = 1
    ](self, idx: Int, value: SIMD[Self.element_type, width]) raises:
        """
        Store a SIMD vector into the Shape at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to store to.
            value: The SIMD vector to store.

        Raises:
            Error: If the store exceeds the bounds of the Shape.
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
                    location="Shape.store",
                )
            )
        self._buf.ptr.unsafe_store[width=width](idx, value)

    def unsafe_load[
        width: Int = 1
    ](self, idx: Int) -> SIMD[Self.element_type, width]:
        """
        Unsafely load a SIMD vector from the Shape at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to load from.

        Returns:
            A SIMD vector containing the loaded values.
        """
        return self._buf.unsafe_load[width=width](idx)

    def unsafe_store[
        width: Int = 1
    ](self, idx: Int, value: SIMD[Self.element_type, width]):
        """
        Unsafely store a SIMD vector into the Shape at the specified index.

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

    def row_major(self) raises -> NDArrayStrides:
        """
        Create row-major (C-style) strides from a shape.

        Row-major means the last dimension has stride 1 and strides increase
        going backwards through dimensions.

        Returns:
            A new NDArrayStrides object with row-major memory layout.
        """
        return NDArrayStrides(shape=self, order="C")

    def col_major(self) raises -> NDArrayStrides:
        """
        Create column-major (Fortran-style) strides from a shape.

        Column-major means the first dimension has stride 1 and strides increase
        going forward through dimensions.

        Returns:
            A new NDArrayStrides object with column-major memory layout.
        """
        return NDArrayStrides(shape=self, order="F")

    def reverse(self) -> Self:
        """
        Return a new shape with dimensions reversed.

        Returns:
            A new NDArrayShape with reversed dimensions.
        """
        return Self(self._buf.flipped())

    def permute(self, axes: List[Int]) raises -> Self:
        """
        Return a new shape with axes reordered.

        Args:
            axes: New axis order. Must contain each axis exactly once.

        Returns:
            A new NDArrayShape with axes permuted.

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
                    location="NDArrayShape.permute",
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
                        location="NDArrayShape.permute",
                    )
                )
            normalized.init_value(i, Scalar[DType.int](axis))

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
                            location="NDArrayShape.permute",
                        )
                    )

        var result = NDArrayShape(ndim=self.ndim, initialized=False)
        for i in range(self.ndim):
            result._buf.init_value(
                i, Scalar[DType.int](self._buf[normalized[i]])
            )
        return result^

    def broadcast(self, other: Self) raises -> Self:
        """
        Compute the broadcast result shape of `self` and `other`,
        following NumPy broadcasting rules.

        Shapes are aligned from the trailing dimension. Two dimensions are
        compatible when they are equal, or when one of them is 1. Missing
        leading dimensions are treated as size 1.

        Args:
            other: The other shape to broadcast against.

        Returns:
            The broadcast result shape.

        Raises:
            Error: If the shapes are not broadcast-compatible.
        """
        var ndim = max(self.ndim, other.ndim)
        var result = NDArrayShape(ndim=ndim, initialized=False)

        for i in range(ndim):
            var a_dim = self.ndim - 1 - i
            var b_dim = other.ndim - 1 - i
            var a_size = self[a_dim] if a_dim >= 0 else 1
            var b_size = other[b_dim] if b_dim >= 0 else 1

            if a_size == b_size:
                result[ndim - 1 - i] = a_size
            elif a_size == 1:
                result[ndim - 1 - i] = b_size
            elif b_size == 1:
                result[ndim - 1 - i] = a_size
            else:
                raise Error(
                    NumojoError(
                        category="shape",
                        message=(
                            "Shapes {} and {} are not broadcast-compatible."
                        ).format(self, other),
                        location="NDArrayShape.broadcast",
                    )
                )

        return result^

    def join(self, *shapes: Self) -> Self:
        """
        Join multiple shapes into a single shape.

        Args:
            shapes: Variable number of NDArrayShape objects.

        Returns:
            A new NDArrayShape object.
        """
        var bufs = List[IndexBuffer]()
        for i in range(len(shapes)):
            bufs.append(shapes[i]._buf)
        return Self(self._buf.join(bufs))

    def swapaxes(self, axis1: Int, axis2: Int) raises -> Self:
        """
        Returns a new shape with the given axes swapped.

        Args:
            axis1: The first axis to swap.
            axis2: The second axis to swap.

        Returns:
            A new shape with the given axes swapped.
        """
        var res = self
        var val1 = res[axis1]
        var val2 = res[axis2]
        res[axis1] = val2
        res[axis2] = val1
        return res

    def extend(self, *values: Int) -> Self:
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

    def flip(mut self):
        """
        Flip the items in-place.
        """
        self._buf.flip()

    def flipped(self) -> Self:
        """
        Returns a new shape by flipping the items.

        Returns:
            A new shape with the items flipped.
        """
        return Self(self._buf.flipped())

    def move_axis_to_end(self, axis: Int) -> Self:
        """
        Returns a new shape by moving the value of axis to the end.

        Args:
            axis: The axis (index) to move.

        Returns:
            A new shape with the axis moved to the end.
        """
        return Self(self._buf.move_axis_to_end(axis))

    def pop(self, axis: Int) raises -> Self:
        """
        Drops the item at the given axis (index).

        Args:
            axis: The axis (index) to drop.

        Returns:
            A new shape with the item at the given axis (index) dropped.
        """
        var new = self._buf.pop(axis)
        return Self(new^)

    # ===----------------------------------------------------------------------=== #
    # Properties
    # ===----------------------------------------------------------------------=== #
    @always_inline("nodebug")
    def rank(self) -> Int:
        """
        Returns the number of dimensions of the shape.

        Returns:
            The rank (ndim) of the shape.
        """
        return self.ndim

    def size(self) -> Int:
        """
        Returns the total number of elements in the array.

        Returns:
          The total number of elements in the corresponding array.
        """
        return Int(self._buf.product())

    def sum(self) -> Scalar[Self.element_type]:
        """
        Compute the sum of all elements in NDArrayShape.

        Returns:
            Sum of all elements in the NDArrayShape.
        """
        return self._buf.sum()

    def product(self) -> Scalar[Self.element_type]:
        """
        Compute the product of all elements in the IndexBuffer.

        Returns:
            Product of all elements in the IndexBuffer.
        """
        return self._buf.product()

    # ===----------------------------------------------------------------------=== #
    # Traits
    # ===----------------------------------------------------------------------=== #
    @always_inline("nodebug")
    def __len__(self) -> Int:
        """
        Gets number of elements in the shape.
        It equals the number of dimensions of the array.

        Returns:
          Number of elements in the shape.
        """
        return self.ndim

    @always_inline("nodebug")
    def __repr__(self) -> String:
        """
        Returns a string of the shape of the array.

        Returns:
            String representation of the shape of the array.
        """
        return "numojo.Shape" + self.__str__()

    def write_repr_to[W: Writer](self, mut writer: W):
        """Write the string representation to a writer.

        Parameters:
            W: The writer type.
        """
        # TODO: Deprecate `__repr__` and move its body directly into this method.
        writer.write(self.__repr__())

    @always_inline("nodebug")
    def __str__(self) -> String:
        """
        Returns a string of the shape of the array.

        Returns:
            String representation of the shape of the array.
        """
        var result: String = "("
        for i in range(self.ndim):
            result += String(self._buf.ptr[unsafe_offset=i])
            if i < self.ndim - 1:
                result += ", "
        result += ")"
        return result

    def write_to[W: Writer](self, mut writer: W):
        """
        Writes the shape representation to a writer.
        """
        writer.write(
            "Shape: " + self.__str__() + "  " + "ndim: " + String(self.ndim)
        )

    @always_inline("nodebug")
    def __eq__(self, other: Self) -> Bool:
        """
        Checks if two shapes have identical dimensions and values.

        Args:
            other: The shape to compare with.

        Returns:
            True if both shapes have identical dimensions and values.
        """
        return self._buf == other._buf

    @always_inline("nodebug")
    def __ne__(self, other: Self) -> Bool:
        """
        Checks if two shapes have identical dimensions and values.

        Args:
            other: The shape to compare with.

        Returns:
           True if both shapes do not have identical dimensions or values.
        """
        return not self.__eq__(other)


    @always_inline("nodebug")
    def __contains__(self, val: Scalar[Self.element_type]) -> Bool:
        """
        Check if the NDArrayShape contains the given value.

        Args:
            val: Value to check for.

        Returns:
            True if the value is in the NDArrayShape, False otherwise.
        """
        return val in self._buf

    # ===----------------------------------------------------------------------=== #
    # Utility Methods
    # ===----------------------------------------------------------------------=== #
    @always_inline("nodebug")
    def tolist(self) -> List[Int]:
        """
        Convert the shape to a list of integers.

        Returns:
            A list containing the shape dimensions.
        """
        var res = List[Int]()
        for i in range(self.ndim):
            res.append(Int(self._buf.unsafe_load(i)))
        return res^

    @always_inline("nodebug")
    def normalize_index(self, index: Int) -> Int:
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
    def __iter__(ref self) -> _ShapeIter[origin_of(self), True]:
        """
        Iterate over elements of the NDArrayShape, returning copied values.

        Returns:
            An iterator of NDArrayShape elements.
        """
        return _ShapeIter[origin_of(self), True](
            shape=Pointer(to=self),
            length=self.ndim,
        )

    def __reversed__(ref self) -> _ShapeIter[origin_of(self), False]:
        """
        Iterate over elements of the NDArrayShape in reverse order, returning copied values.

        Returns:
            An iterator of NDArrayShape elements in reverse order.
        """
        return _ShapeIter[origin_of(self), False](
            shape=Pointer(to=self),
            length=self.ndim,
        )


# ===----------------------------------------------------------------------=== #
# NDArrayShape Iterator
# ===----------------------------------------------------------------------=== #
struct _ShapeIter[
    origin: ImmOrigin = ImmUntrackedOrigin,
    forward: Bool = True,
](ImplicitlyCopyable, Movable):
    """Iterator for NDArrayShape.

    Parameters:
        origin: The origin of the iterator.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var shape: Pointer[NDArrayShape, Self.origin]
    var length: Int

    def __init__(
        out self,
        shape: Pointer[NDArrayShape, Self.origin],
        length: Int,
    ):
        self.index = 0 if Self.forward else length - 1
        self.length = length
        self.shape = shape

    def __iter__(self) -> Self:
        return self

    def __has_next__(self) -> Bool:
        comptime if Self.forward:
            return self.index < self.length
        else:
            return self.index >= 0

    def __next__(mut self) raises -> Scalar[DType.int]:
        comptime if Self.forward:
            var current_index = self.index
            self.index += 1
            return Scalar[DType.int](self.shape[].__getitem__(current_index))
        else:
            var current_index = self.index
            self.index -= 1
            return Scalar[DType.int](self.shape[].__getitem__(current_index))

    def __len__(self) -> Int:
        comptime if Self.forward:
            return self.length - self.index
        else:
            return self.index + 1
