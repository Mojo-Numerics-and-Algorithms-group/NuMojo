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
            T: The type of the initial values. It can be converted to `Int`
                with `index()` function.

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
        for i in range(other.len):
            self._buf[i] = other._buf[i]

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the length of the tuple.

        Returns:
            The length of the tuple.
        """
        return self.len

    @always_inline("nodebug")
    fn __getitem__[T: Indexer](self, idx: T) -> Int:
        """Get the value at the specified index.

        Args:
            idx: The index of the value to get.

        Returns:
            The value at the specified index.
        """
        return self._buf[index(idx)]

    @always_inline("nodebug")
    fn __setitem__[T: Indexer, U: Indexer](self, idx: T, val: U):
        """Set the value at the specified index.

        Args:
            idx: The index of the value to set.
            val: The value to set.
        """
        self._buf[index(idx)] = index(val)

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

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Item: " + self.str() + "  " + "Length: " + str(self.len))

    fn str(self) -> String:
        var result: String = "("
        for i in range(self.len):
            result += str(self._buf[i])
            if i < self.len - 1:
                result += ","
        result += ")"
        return result


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
