# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Implements Item type.

`Item` is a series of `Int` on the heap.
"""

from builtin.int import index as convert_to_int
from memory import memcpy, memset_zero
from memory import UnsafePointer
from numojo.core.indexing.index_buffer import IndexBuffer
from memory import memcmp
from os import abort
from sys import simd_width_of
from utils import Variant

from numojo.core.error import NumojoError
from numojo.core.traits.indexer_collection_element import (
    IndexerCollectionElement,
)


@register_passable
struct Item(
    Equatable,
    ImplicitlyCopyable,
    Movable,
    Representable,
    Sized,
    Stringable,
    Writable,
):
    """
    Represents a multi-dimensional index for array access.

    The `Item` struct is used to specify the coordinates of an element within an N-dimensional array.
    For example, `arr[Item(1, 2, 3)]` retrieves the element at position (1, 2, 3) in a 3D array.

    Each `Item` instance holds a sequence of integer indices, one for each dimension of the array.
    This allows for precise and flexible indexing into arrays of arbitrary dimensionality.

    Example:
        ```mojo
        from numojo.prelude import *
        import numojo as nm
        var arr = nm.arange[f32](0, 27).reshape(Shape(3, 3, 3))
        var value = arr[Item(1, 2, 3)]  # Accesses arr[1, 2, 3]
        ```
    """

    # ===----------------------------------------------------------------------=== #
    # Aliases
    # ===----------------------------------------------------------------------=== #

    comptime element_type: DType = DType.int
    """The data type of the Item elements."""

    # ===----------------------------------------------------------------------=== #
    # Fields
    # ===----------------------------------------------------------------------=== #

    var _buf: IndexBuffer
    """Data buffer."""
    var ndim: Int
    """Number of dimensions (length of the index tuple)."""

    # ===----------------------------------------------------------------------=== #
    # Lifecycle Methods
    # ===----------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __init__(out self):
        """
        Initializes an empty Item.
        """
        self.ndim = 0
        self._buf = IndexBuffer()

    @always_inline("nodebug")
    fn __init__(out self, buf: IndexBuffer):
        """
        Initializes the Item from an IndexBuffer.

        Args:
            buf: The IndexBuffer to initialize from.
        """
        self.ndim = buf.ndim
        self._buf = buf

    @always_inline("nodebug")
    fn __init__[T: Indexer](out self, *args: T):
        """Construct the Item with variable arguments.

        Parameters:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            args: Initial values.
        """
        self.ndim = len(args)
        self._buf = IndexBuffer(size=self.ndim)
        for i in range(self.ndim):
            self._buf.init_value(i, convert_to_int(args[i]))

    @always_inline("nodebug")
    fn __init__[T: IndexerCollectionElement](out self, args: List[T]):
        """Construct the Item from a list.

        Parameters:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            args: Initial values.
        """
        self.ndim = len(args)
        self._buf = IndexBuffer(size=self.ndim)
        for i in range(self.ndim):
            self._buf.init_value(i, convert_to_int(args[i]))

    @always_inline("nodebug")
    fn __init__(out self, args: List[Int]):
        """Construct the Item from a list.

        Args:
            args: Initial values.
        """
        self.ndim = len(args)
        self._buf = IndexBuffer(size=self.ndim)
        for i in range(self.ndim):
            self._buf.init_value(i, args[i])

    @always_inline("nodebug")
    fn __init__(out self, args: VariadicList[Int]):
        """Construct the Item from a variadic list.

        Args:
            args: Initial values.
        """
        self.ndim = len(args)
        self._buf = IndexBuffer(size=self.ndim)
        for i in range(self.ndim):
            self._buf.init_value(i, args[i])

    @always_inline("nodebug")
    fn __init__(out self, *, ndim: Int):
        """Construct the Item with given length and initialize to zero.

        Args:
            ndim: The length of the Item.
        """
        self.ndim = ndim
        self._buf = IndexBuffer(size=ndim)
        memset_zero(self._buf.ptr, ndim)

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """Copy construct the Item.

        Args:
            other: The Item to copy.
        """
        self.ndim = other.ndim
        self._buf = IndexBuffer(size=self.ndim)
        memcpy(dest=self._buf.ptr, src=other._buf.ptr, count=self.ndim)

    # ===----------------------------------------------------------------------=== #
    # Element Access Methods
    # ===----------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) raises -> Int:
        """Gets the value at the specified index.

        Args:
            idx: The index of the value to get.

        Returns:
            The value at the specified index.

        Raises:
            Error: If index is out of range.
        """
        return Int(self._buf[idx])

    @always_inline("nodebug")
    fn __getitem__(self, slice_index: Slice) raises -> Self:
        """
        Return a sliced view of the item as a new Item.

        Args:
            slice_index: Slice object defining the sub-buffer.

        Returns:
            A new Item representing the sliced values.
        """
        return Self(self._buf[slice_index])

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, val: Int) raises:
        """Set the value at the specified index.

        Args:
            idx: The index of the value to set.
            val: The value to set.

        Raises:
            Error: If index is out of range.
        """
        self._buf[idx] = val

    fn load[
        width: Int = 1
    ](self, idx: Int) raises -> SIMD[Self.element_type, width]:
        """
        Load a SIMD vector from the Item at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to load from.

        Returns:
            A SIMD vector containing the loaded values.

        Raises:
            Error: If the load exceeds the bounds of the Item.
        """
        if idx < 0 or idx + width > self.ndim:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Load operation out of bounds: idx={} width={} ndim={}."
                        " Ensure that idx and width are within valid range."
                    ).format(idx, width, self.ndim),
                    location="Item.load",
                )
            )
        return self._buf.load[width=width](idx)

    fn store[
        width: Int = 1
    ](self, idx: Int, value: SIMD[Self.element_type, width]) raises:
        """
        Store a SIMD vector into the Item at the specified index.

        Parameters:
            width: The width of the SIMD vector.

        Args:
            idx: The starting index to store to.
            value: The SIMD vector to store.

        Raises:
            Error: If the store exceeds the bounds of the Item.
        """
        if idx < 0 or idx + width > self.ndim:
            raise Error(
                NumojoError(
                    category="index",
                    message=String(
                        "Store operation out of bounds: idx={} width={}"
                        " ndim={}. Ensure that idx and width are within valid"
                        " range."
                    ).format(idx, width, self.ndim),
                    location="Item.store",
                )
            )
        self._buf.store[width=width](idx, value)

    fn unsafe_load[
        width: Int = 1
    ](self, idx: Int) -> SIMD[Self.element_type, width]:
        """
        Unsafely load a SIMD vector from the Item at the specified index.

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
        Unsafely store a SIMD vector into the Item at the specified index.

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

    fn swapaxes(self, axis1: Int, axis2: Int) raises -> Self:
        """
        Returns a new item with the given axes swapped.

        Args:
            axis1: The first axis to swap.
            axis2: The second axis to swap.

        Returns:
            A new item with the given axes swapped.
        """
        var res = self
        var val1 = res[axis1]
        var val2 = res[axis2]
        res[axis1] = val2
        res[axis2] = val1
        return res

    fn join(self, *others: Self) -> Self:
        """
        Join multiple items into a single item.

        Args:
            others: Variable number of Item objects.

        Returns:
            A new Item object with all values concatenated.
        """
        var bufs = List[IndexBuffer]()
        for i in range(len(others)):
            bufs.append(others[i]._buf)
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
        Returns a new item by flipping the items.

        Returns:
            A new item with the items flipped.
        """
        return Self(self._buf.flipped())

    fn move_axis_to_end(self, axis: Int) -> Self:
        """
        Returns a new item by moving the value of axis to the end.

        Args:
            axis: The axis (index) to move.

        Returns:
            A new item with the specified axis moved to the end.
        """
        return Self(self._buf.move_axis_to_end(axis))

    fn pop(self, axis: Int) raises -> Self:
        """
        Drops information of certain axis.

        Args:
            axis: The axis (index) to drop.

        Returns:
            A new item with the item at the given axis (index) dropped.
        """
        return Self(self._buf.pop(axis))

    # ===----------------------------------------------------------------------=== #
    # Properties
    # ===----------------------------------------------------------------------=== #
    @always_inline("nodebug")
    fn rank(self) -> Int:
        """
        Returns the number of dimensions of the Item.

        Returns:
            The rank (ndim) of the Item.
        """
        return self.ndim

    fn sum(self) -> Scalar[Self.element_type]:
        """
        Compute the sum of all elements in Item.

        Returns:
            Sum of all elements in the Item.
        """
        return self._buf.sum()

    fn product(self) -> Scalar[Self.element_type]:
        """
        Compute the product of all elements in the Item.

        Returns:
            Product of all elements in the Item.
        """
        return self._buf.product()

    # ===----------------------------------------------------------------------=== #
    # Traits
    # ===----------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the length of the Item.

        Returns:
            The length of the Item.
        """
        return self.ndim

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """
        Returns a string representation of the Item.

        Returns:
            String representation of the Item.
        """
        return "numojo.Item" + self.__str__()

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """
        Returns a string representation of the Item.

        Returns:
            String representation of the Item.
        """
        var result: String = "("
        for i in range(self.ndim):
            result += String(self._buf.unsafe_load(i))
            if i < self.ndim - 1:
                result += ", "
        result += ")"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        """
        Writes the Item representation to a writer.
        """
        writer.write("Coordinates: " + self.__str__() + "  ")

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """
        Checks if two items have identical dimensions and values.

        Args:
            other: The item to compare with.

        Returns:
            True if both items have identical dimensions and values.
        """
        return self._buf == other._buf

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """
        Checks if two items have different dimensions or values.

        Args:
            other: The item to compare with.

        Returns:
            True if both items do not have identical dimensions or values.
        """
        return not self.__eq__(other)

    fn __contains__(self, val: Scalar[Self.element_type]) -> Bool:
        """
        Check if the Item contains the given value.

        Args:
            val: Value to check for.

        Returns:
            True if the value is in the Item, False otherwise.
        """
        return val in self._buf

    @always_inline("nodebug")
    fn __contains__(self, val: Int) -> Bool:
        """
        Checks if the given value is present in the item.

        Args:
            val: The value to search for.

        Returns:
            True if the given value is present in the item.
        """
        return val in self._buf

    # ===----------------------------------------------------------------------=== #
    # Utility Methods
    # ===----------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn tolist(self) -> List[Int]:
        """
        Convert the Item to a list of integers.

        Returns:
            A list containing the Item values.
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
        var norm_idx: Int = index
        if norm_idx < 0:
            norm_idx += self.ndim
        return norm_idx

    # ===----------------------------------------------------------------------=== #
    # Iterators
    # ===----------------------------------------------------------------------=== #
    fn __iter__(ref self) -> _ItemIter[origin_of(self), True]:
        """Iterate over elements of the Item.

        Returns:
            An iterator of Item elements.
        """
        return _ItemIter[origin_of(self), True](
            item=Pointer(to=self),
            length=self.ndim,
        )

    fn __reversed__(ref self) -> _ItemIter[origin_of(self), False]:
        """Iterate over elements of the Item in reverse.

        Returns:
            An iterator of Item elements in reverse.
        """
        return _ItemIter[origin_of(self), False](
            item=Pointer(to=self),
            length=self.ndim,
        )


struct _ItemIter[
    origin: ImmutOrigin = ImmutExternalOrigin,
    forward: Bool = True,
](ImplicitlyCopyable, Movable):
    """Iterator for Item.

    Parameters:
        origin: The origin of the Item being iterated.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var item: Pointer[Item, Self.origin]
    var length: Int

    fn __init__(
        out self,
        item: Pointer[Item, Self.origin],
        length: Int,
    ):
        self.index = 0 if Self.forward else length - 1
        self.length = length
        self.item = item

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
            return self.item[].__getitem__(current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.item[].__getitem__(current_index)

    fn __len__(self) -> Int:
        @parameter
        if Self.forward:
            return self.length - self.index
        else:
            return self.index + 1
