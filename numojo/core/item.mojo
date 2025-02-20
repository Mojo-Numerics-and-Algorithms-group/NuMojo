"""
Implements Item type.

`Item` is a series of `Int` on the heap.
"""

from builtin.type_aliases import Origin
from memory import UnsafePointer, memset_zero, memcpy
from sys import simdwidthof
from utils import Variant

from numojo.core.traits.indexer_collection_element import (
    IndexerCollectionElement,
)

alias item = Item


@register_passable
struct Item(CollectionElement):
    var _buf: UnsafePointer[Int]
    var ndim: Int

    @always_inline("nodebug")
    fn __init__[T: Indexer](out self, *args: T):
        """Construct the tuple.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            args: Initial values.
        """
        self._buf = UnsafePointer[Int]().alloc(args.__len__())
        self.ndim = args.__len__()
        for i in range(args.__len__()):
            self._buf[i] = Int(args[i])

    @always_inline("nodebug")
    fn __init__[T: IndexerCollectionElement](out self, args: List[T]) raises:
        """Construct the tuple.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            args: Initial values.
        """
        self.ndim = len(args)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(Int(args[i]))

    @always_inline("nodebug")
    fn __init__(out self, args: VariadicList[Int]) raises:
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self.ndim = len(args)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(Int(args[i]))

    @always_inline("nodebug")
    fn __init__(
        out self,
        ndim: Int,
        initialized: Bool,
    ) raises:
        """
        Construct Item with number of dimensions.

        This method is useful when you want to create a Item with given ndim
        without knowing the Item values.

        Args:
            ndim: Number of dimensions.
            initialized: Whether the shape is initialized.
                If yes, the values will be set to 0.
                If no, the values will be uninitialized.
        """
        if ndim < 0:
            raise Error("Number of dimensions must be non-negative.")
        self.ndim = ndim
        self._buf = UnsafePointer[Int]().alloc(ndim)
        if initialized:
            for i in range(ndim):
                (self._buf + i).init_pointee_copy(0)

    @always_inline("nodebug")
    fn __copyinit__(mut self, other: Self):
        """Copy construct the tuple.

        Args:
            other: The tuple to copy.
        """
        self.ndim = other.ndim
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        memcpy(self._buf, other._buf, self.ndim)

    @always_inline("nodebug")
    fn __del__(owned self):
        self._buf.free()

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the length of the tuple.

        Returns:
            The length of the tuple.
        """
        return self.ndim

    @always_inline("nodebug")
    fn __getitem__[T: Indexer](self, idx: T) raises -> Int:
        """Get the value at the specified index.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            idx: The index of the value to get.

        Returns:
            The value at the specified index.
        """

        var normalized_idx: Int = Int(idx)
        if normalized_idx < 0:
            normalized_idx = Int(idx) + self.ndim

        if normalized_idx < 0 or normalized_idx >= self.ndim:
            raise Error(
                String(
                    "Error in `numojo.Item.__getitem__()`: \n"
                    "Index ({}) out of range [{}, {})\n"
                ).format(Int(idx), -self.ndim, self.ndim - 1)
            )

        return self._buf[normalized_idx]

    @always_inline("nodebug")
    fn __setitem__[T: Indexer, U: Indexer](self, idx: T, val: U) raises:
        """Set the value at the specified index.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.
            U: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            idx: The index of the value to set.
            val: The value to set.
        """

        var normalized_idx: Int = Int(idx)
        if normalized_idx < 0:
            normalized_idx = Int(idx) + self.ndim

        if normalized_idx < 0 or normalized_idx >= self.ndim:
            raise Error(
                String(
                    "Error in `numojo.Item.__getitem__()`: \n"
                    "Index ({}) out of range [{}, {})\n"
                ).format(Int(idx), -self.ndim, self.ndim - 1)
            )

        self._buf[normalized_idx] = Int(val)

    fn __iter__(self) raises -> _ItemIter:
        """Iterate over elements of the NDArray, returning copied value.

        Returns:
            An iterator of NDArray elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _ItemIter(
            item=self,
            length=self.ndim,
        )

    fn __repr__(self) -> String:
        var result: String = "numojo.Item" + String(self)
        return result

    fn __str__(self) -> String:
        var result: String = "("
        for i in range(self.ndim):
            result += String(self._buf[i])
            if i < self.ndim - 1:
                result += ","
        result += ")"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Item at index: "
            + String(self)
            + "  "
            + "Length: "
            + String(self.ndim)
        )

    # ===-------------------------------------------------------------------===#
    # Other methods
    # ===-------------------------------------------------------------------===#

    fn offset(self, strides: NDArrayStrides) -> Int:
        """
        Calculates the offset of the item according to strides.

        Args:
            strides: The strides of the array.

        Returns:
            The offset of the item.

        Examples:

        ```mojo
        from numojo.prelude import *
        var item = Item(1, 2, 3)
        var strides = nm.Strides(4, 3, 2)
        print(item.offset(strides))
        # This prints `16`.
        ```
        """

        var offset: Int = 0
        for i in range(self.ndim):
            offset += self._buf[i] * strides._buf[i]
        return offset


@value
struct _ItemIter[
    forward: Bool = True,
]:
    """Iterator for Item.

    Parameters:
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var item: Item
    var length: Int

    fn __init__(
        mut self,
        item: Item,
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.item = item

    fn __iter__(self) -> Self:
        return self

    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index > 0

    fn __next__(mut self) raises -> Scalar[DType.index]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.item.__getitem__(current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.item.__getitem__(current_index)

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index
