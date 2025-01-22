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
    var len: Int

    @always_inline("nodebug")
    fn __init__[T: Indexer](out self, *args: T):
        """Construct the tuple.

        Parameter:
            T: Type of values. It can be converted to `Int` with `index()`.

        Args:
            args: Initial values.
        """
        self._buf = UnsafePointer[Int]().alloc(args.__len__())
        self.len = args.__len__()
        for i in range(args.__len__()):
            self._buf[i] = index(args[i])

    @always_inline("nodebug")
    fn __init__[T: IndexerCollectionElement](out self, args: List[T]) raises:
        """Construct the tuple.

        Parameter:
            T: Type of values. It can be converted to `Int` with `index()`.

        Args:
            args: Initial values.
        """
        self.len = len(args)
        self._buf = UnsafePointer[Int]().alloc(self.len)
        for i in range(self.len):
            (self._buf + i).init_pointee_copy(index(args[i]))

    @always_inline("nodebug")
    fn __init__(out self, args: VariadicList[Int]) raises:
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self.len = len(args)
        self._buf = UnsafePointer[Int]().alloc(self.len)
        for i in range(self.len):
            (self._buf + i).init_pointee_copy(index(args[i]))

    @always_inline("nodebug")
    fn __copyinit__(mut self, other: Self):
        """Copy construct the tuple.

        Args:
            other: The tuple to copy.
        """
        self.len = other.len
        self._buf = UnsafePointer[Int]().alloc(self.len)
        memcpy(self._buf, other._buf, self.len)

    @always_inline("nodebug")
    fn __del__(owned self):
        self._buf.free()

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the length of the tuple.

        Returns:
            The length of the tuple.
        """
        return self.len

    @always_inline("nodebug")
    fn __getitem__[T: Indexer](self, idx: T) raises -> Int:
        """Get the value at the specified index.

        Parameter:
            T: Type of values. It can be converted to `Int` with `index()`.

        Args:
            idx: The index of the value to get.

        Returns:
            The value at the specified index.
        """

        var normalized_idx: Int = index(idx)
        if normalized_idx < 0:
            normalized_idx = idx + self.len

        if normalized_idx < 0 or normalized_idx >= self.len:
            raise Error(
                String("Index ({}) out of range [{}, {})").format(
                    index(idx), -self.len, self.len - 1
                )
            )

        return self._buf[normalized_idx]

    @always_inline("nodebug")
    fn __setitem__[T: Indexer, U: Indexer](self, idx: T, val: U) raises:
        """Set the value at the specified index.

        Parameter:
            T: Type of values. It can be converted to `Int` with `index()`.
            U: Type of values. It can be converted to `Int` with `index()`.

        Args:
            idx: The index of the value to set.
            val: The value to set.
        """

        var normalized_idx: Int = index(idx)
        if normalized_idx < 0:
            normalized_idx = idx + self.len

        if normalized_idx < 0 or normalized_idx >= self.len:
            raise Error(
                String("Index ({}) out of range [{}, {})").format(
                    index(idx), -self.len, self.len - 1
                )
            )

        self._buf[normalized_idx] = index(val)

    fn __iter__(self) raises -> _ItemIter:
        """Iterate over elements of the NDArray, returning copied value.

        Returns:
            An iterator of NDArray elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _ItemIter(
            item=self,
            length=self.len,
        )

    fn __repr__(self) -> String:
        var result: String = "numojo.Item" + str(self)
        return result

    fn __str__(self) -> String:
        var result: String = "("
        for i in range(self.len):
            result += str(self._buf[i])
            if i < self.len - 1:
                result += ","
        result += ")"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Item at index: " + str(self) + "  " + "Length: " + str(self.len)
        )


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
