"""
Implements Item type.

`Item` is a series of `Int` on the heap.
"""

from builtin.type_aliases import Origin
from builtin.int import index as index_int
from memory import UnsafePointer, memset_zero, memcpy
from memory import memcmp
from os import abort
from sys import simd_width_of
from utils import Variant

from numojo.core.error import IndexError, ValueError
from numojo.core.traits.indexer_collection_element import (
    IndexerCollectionElement,
)

# simple alias for users. Use `Item` internally.
alias item = Item


@register_passable
struct Item(
    ImplicitlyCopyable, Movable, Representable, Sized, Stringable, Writable
):
    """
    Specifies the indices of an item of an array.
    """

    # Aliases
    alias _type: DType = DType.int

    # Fields
    var _buf: UnsafePointer[Scalar[Self._type]]
    var ndim: Int

    @always_inline("nodebug")
    fn __init__[T: Indexer](out self, *args: T):
        """Construct the tuple.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            args: Initial values.
        """
        self._buf = UnsafePointer[Scalar[Self._type]]().alloc(args.__len__())
        self.ndim = args.__len__()
        for i in range(args.__len__()):
            self._buf[i] = index(args[i])

    @always_inline("nodebug")
    fn __init__[T: IndexerCollectionElement](out self, args: List[T]) raises:
        """Construct the tuple.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            args: Initial values.
        """
        self.ndim = len(args)
        self._buf = UnsafePointer[Scalar[Self._type]]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(index(args[i]))

    @always_inline("nodebug")
    fn __init__(out self, args: VariadicList[Int]) raises:
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self.ndim = len(args)
        self._buf = UnsafePointer[Scalar[Self._type]]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(Int(args[i]))

    fn __init__(
        out self,
        *,
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

        Raises:
            Error: If the number of dimensions is negative.
        """
        if ndim < 0:
            raise Error(
                IndexError(
                    message=String(
                        "Invalid ndim: got {}; must be >= 0."
                    ).format(ndim),
                    suggestion=String(
                        "Pass a non-negative dimension count when constructing"
                        " Item."
                    ),
                    location=String("Item.__init__(ndim: Int)"),
                )
            )

        if ndim == 0:
            self.ndim = 0
            self._buf = UnsafePointer[Scalar[Self._type]]()
        else:
            self.ndim = ndim
            self._buf = UnsafePointer[Scalar[Self._type]]().alloc(ndim)
            if initialized:
                for i in range(ndim):
                    (self._buf + i).init_pointee_copy(0)

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """Copy construct the tuple.

        Args:
            other: The tuple to copy.
        """
        self.ndim = other.ndim
        self._buf = UnsafePointer[Scalar[Self._type]]().alloc(self.ndim)
        memcpy(self._buf, other._buf, self.ndim)

    @always_inline("nodebug")
    fn __del__(deinit self):
        if self.ndim > 0:
            self._buf.free()

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the length of the tuple.

        Returns:
            The length of the tuple.
        """
        return self.ndim

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
    fn __getitem__[T: Indexer](self, idx: T) raises -> Scalar[Self._type]:
        """Gets the value at the specified index.

        Parameter:
            T: Type of values. It can be converted to `Int` with `Int()`.

        Args:
            idx: The index of the value to get.

        Returns:
            The value at the specified index.
        """
        var index: Int = index_int(idx)
        if index >= self.ndim or index < -self.ndim:
            raise Error(
                IndexError(
                    message=String("Index {} out of range [{} , {}).").format(
                        index_int(idx), -self.ndim, self.ndim
                    ),
                    suggestion=String(
                        "Use indices in [-ndim, ndim) (negative indices wrap)."
                    ),
                    location=String("Item.__getitem__"),
                )
            )
        var normalized_idx: Int = self.normalize_index(index_int(idx))
        return self._buf[normalized_idx]

    @always_inline("nodebug")
    fn __getitem__(self, slice_index: Slice) raises -> Self:
        """
        Return a sliced view of the item as a new Item.
        Delegates normalization & validation to _compute_slice_params.

        Args:
            slice_index: The slice to extract.

        Returns:
            A new Item containing the sliced values.

        Example:
        ```mojo
        from numojo.prelude import *
        var item = Item(10, 20, 30, 40, 50)
        print(item[1:4])  # Item: (20, 30, 40)
        print(item[::2])  # Item: (10, 30, 50)
        ```
        """
        var updated_slice: Tuple[Int, Int, Int] = self._compute_slice_params(
            slice_index
        )
        var start = updated_slice[0]
        var step = updated_slice[1]
        var length = updated_slice[2]

        if length <= 0:
            var empty_result = Self(ndim=0, initialized=False)
            return empty_result

        var result = Self(ndim=length, initialized=False)
        var idx = start
        for i in range(length):
            (result._buf + i).init_pointee_copy(self._buf[idx])
            idx += step
        return result^

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

        var normalized_idx: Int = index_int(idx)
        if normalized_idx < 0:
            normalized_idx = index_int(idx) + self.ndim

        if normalized_idx < 0 or normalized_idx >= self.ndim:
            raise Error(
                IndexError(
                    message=String("Index {} out of range [{} , {}).").format(
                        index_int(idx), -self.ndim, self.ndim
                    ),
                    suggestion=String(
                        "Use indices in [-ndim, ndim) (negative indices wrap)."
                    ),
                    location=String("Item.__setitem__"),
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

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """
        Checks if two items have identical dimensions and values.

        Args:
            other: The item to compare with.

        Returns:
            True if both items have identical dimensions and values.
        """
        if self.ndim != other.ndim:
            return False
        if memcmp(self._buf, other._buf, self.ndim) != 0:
            return False
        return True

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

    @always_inline("nodebug")
    fn __contains__(self, val: Int) -> Bool:
        """
        Checks if the given value is present in the item.

        Args:
            val: The value to search for.

        Returns:
            True if the given value is present in the item.
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
        Returns a deep copy of the item.

        Returns:
            A new Item with the same values.
        """
        var res = Self(ndim=self.ndim, initialized=False)
        memcpy(res._buf, self._buf, self.ndim)
        return res^

    fn swapaxes(self, axis1: Int, axis2: Int) raises -> Self:
        """
        Returns a new item with the given axes swapped.

        Args:
            axis1: The first axis to swap.
            axis2: The second axis to swap.

        Returns:
            A new item with the given axes swapped.
        """
        var res: Self = Self(ndim=self.ndim, initialized=False)
        memcpy(dest=res._buf, src=self._buf, count=self.ndim)
        res[axis1] = self[axis2]
        res[axis2] = self[axis1]
        return res

    fn join(self, *others: Self) raises -> Self:
        """
        Join multiple items into a single item.

        Args:
            others: Variable number of Item objects.

        Returns:
            A new Item object with all values concatenated.

        Examples:
        ```mojo
        from numojo.prelude import *
        var item1 = Item(1, 2)
        var item2 = Item(3, 4)
        var item3 = Item(5)
        var joined = item1.join(item2, item3)
        print(joined)  # Item at index: (1,2,3,4,5)
        ```
        """
        var total_dims: Int = self.ndim
        for i in range(len(others)):
            total_dims += others[i].ndim

        var new_item: Self = Self(ndim=total_dims, initialized=False)

        var index: UInt = 0
        for i in range(self.ndim):
            (new_item._buf + index).init_pointee_copy(self[i])
            index += 1

        for i in range(len(others)):
            for j in range(others[i].ndim):
                (new_item._buf + index).init_pointee_copy(others[i][j])
                index += 1

        return new_item^

    # ===-------------------------------------------------------------------===#
    # Other private methods
    # ===-------------------------------------------------------------------===#

    fn _flip(self) raises -> Self:
        """
        Returns a new item by flipping the items.
        ***UNSAFE!*** No boundary check!

        Returns:
            A new item with the items flipped.

        Example:
        ```mojo
        from numojo.prelude import *
        var item = Item(1, 2, 3)
        print(item)          # Item: (1, 2, 3)
        print(item._flip())  # Item: (3, 2, 1)
        ```
        """
        var result: Self = Self(ndim=self.ndim, initialized=False)
        memcpy(dest=result._buf, src=self._buf, count=self.ndim)
        for i in range(result.ndim):
            result._buf[i] = self._buf[self.ndim - 1 - i]
        return result^

    fn _move_axis_to_end(self, var axis: Int) raises -> Self:
        """
        Returns a new item by moving the value of axis to the end.
        ***UNSAFE!*** No boundary check!

        Args:
            axis: The axis (index) to move. It should be in `[-ndim, ndim)`.

        Returns:
            A new item with the specified axis moved to the end.

        Example:
        ```mojo
        from numojo.prelude import *
        var item = Item(10, 20, 30)
        print(item._move_axis_to_end(0))  # Item: (20, 30, 10)
        print(item._move_axis_to_end(1))  # Item: (10, 30, 20)
        ```
        """
        if axis < 0:
            axis += self.ndim

        var result: Self = Self(ndim=self.ndim, initialized=False)
        memcpy(dest=result._buf, src=self._buf, count=self.ndim)

        if axis == self.ndim - 1:
            return result^

        var value: Scalar[Self._type] = result._buf[axis]
        for i in range(axis, result.ndim - 1):
            result._buf[i] = result._buf[i + 1]
        result._buf[result.ndim - 1] = value
        return result^

    fn _pop(self, axis: Int) raises -> Self:
        """
        Drops information of certain axis.
        ***UNSAFE!*** No boundary check!

        Args:
            axis: The axis (index) to drop. It should be in `[0, ndim)`.

        Returns:
            A new item with the item at the given axis (index) dropped.
        """
        var res: Self = Self(ndim=self.ndim - 1, initialized=False)
        memcpy(dest=res._buf, src=self._buf, count=axis)
        memcpy(
            dest=res._buf + axis,
            src=self._buf.offset(axis + 1),
            count=self.ndim - axis - 1,
        )
        return res^

    fn _extend(self, *values: Int) raises -> Self:
        """
        Extend the item by additional values.
        ***UNSAFE!*** No boundary check!

        Args:
            values: Additional values to append.

        Returns:
            A new Item object with the extended values.

        Example:
        ```mojo
        from numojo.prelude import *
        var item = Item(1, 2, 3)
        var extended = item._extend(4, 5)
        print(extended)  # Item: (1, 2, 3, 4, 5)
        ```
        """
        var total_dims: Int = self.ndim + len(values)
        var new_item: Self = Self(ndim=total_dims, initialized=False)

        var offset: UInt = 0
        for i in range(self.ndim):
            (new_item._buf + offset).init_pointee_copy(self[i])
            offset += 1
        for value in values:
            (new_item._buf + offset).init_pointee_copy(value)
            offset += 1

        return new_item^

    fn _compute_slice_params(
        self, slice_index: Slice
    ) raises -> (Int, Int, Int):
        """
        Compute normalized slice parameters (start, step, length).

        Args:
            slice_index: The slice to compute parameters for.

        Returns:
            A tuple of (start, step, length).

        Raises:
            Error: If the slice step is zero.
        """
        var n = self.ndim
        if n == 0:
            return (0, 1, 0)

        var step = slice_index.step.or_else(1)
        if step == 0:
            raise Error(
                ValueError(
                    message="Slice step cannot be zero.",
                    suggestion="Use a non-zero step value.",
                    location="Item._compute_slice_params",
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


struct _ItemIter[
    forward: Bool = True,
](ImplicitlyCopyable, Movable):
    """Iterator for Item.

    Parameters:
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var item: Item
    var length: Int

    fn __init__(
        out self,
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
