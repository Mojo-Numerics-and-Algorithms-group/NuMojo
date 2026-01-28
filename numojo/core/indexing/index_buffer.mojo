# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Shared integer buffer backend for shape/strides/item.

This type owns a contiguous heap buffer of Ints and provides
small helpers for pointer access and SIMD load/store.
"""
from memory import UnsafePointer, memcpy, memset_zero
from sys import simd_width_of
from algorithm.functional import vectorize

from numojo.core.error import NumojoError
from numojo.core.indexing.slicing import InternalSlice


@register_passable
struct IndexBuffer(
    Equatable, ImplicitlyCopyable, Movable, Sized, Stringable, Writable
):
    """
    Shared integer buffer backend for shape/strides/item.
    """

    comptime element_type: DType = DType.int
    """Element type of the buffer."""
    comptime simd_width: Int = simd_width_of[DType.int]()
    """SIMD width for the element type."""
    comptime _origin: MutOrigin = MutExternalOrigin
    """Mutability origin of the buffer."""

    var ptr: UnsafePointer[Scalar[Self.element_type], Self._origin]
    """Pointer to the buffer."""
    var ndim: Int
    """Number of elements in the buffer."""

    # ===----------------------------------------------------------------------=== #
    # Lifecycle Methods
    # ===----------------------------------------------------------------------=== #
    # TODO: add `abort()` for cases where we have ndim < 0 since they are invalid.
    fn __init__(out self, *, size: Int):
        """
        Initialize an IndexBuffer of given size.

        Args:
            size: Number of elements in the buffer.
        """
        self.ndim = size
        if size <= 0:
            self.ptr = UnsafePointer[Scalar[Self.element_type], Self._origin]()
        else:
            self.ptr = alloc[Scalar[DType.int]](size)
            memset_zero(self.ptr, size)

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.element_type], Self._origin],
        size: Int,
    ):
        """
        Initialize an IndexBuffer with an existing pointer and size.

        Args:
            ptr: UnsafePointer to the buffer.
            size: Number of elements in the buffer.
        """
        self.ptr = ptr
        self.ndim = size

    fn __init__(out self):
        """
        Initialize an empty IndexBuffer.
        """
        self.ndim = 0
        self.ptr = UnsafePointer[Scalar[Self.element_type], Self._origin]()

    fn __init__(out self, *values: Int):
        """
        Initialize an IndexBuffer with given Int values.

        Args:
            values: Variadic list of integer values.
        """
        self.ndim = len(values)
        if self.ndim <= 0:
            self.ptr = UnsafePointer[Scalar[Self.element_type], Self._origin]()
            return
        self.ptr = alloc[Scalar[Self.element_type]](self.ndim)
        for i in range(self.ndim):
            (self.ptr + i).init_pointee_copy(values[i])

    # NOTE: In future this will be equivalent to Int.
    fn __init__(out self, *values: Scalar[Self.element_type]):
        """
        Initialize an IndexBuffer with given values.

        Args:
            values: Variadic list of integer values.
        """
        self.ndim = len(values)
        if self.ndim <= 0:
            self.ptr = UnsafePointer[Scalar[Self.element_type], Self._origin]()
            return
        self.ptr = alloc[Scalar[Self.element_type]](self.ndim)
        for i in range(self.ndim):
            (self.ptr + i).init_pointee_copy(values[i])

    fn __init__(out self, values: List[Scalar[Self.element_type]]):
        """
        Initialize an IndexBuffer with a list of values.

        Args:
            values: List of integer values.
        """
        self.ndim = len(values)
        if self.ndim <= 0:
            self.ptr = UnsafePointer[Scalar[Self.element_type], Self._origin]()
            return
        self.ptr = alloc[Scalar[Self.element_type]](self.ndim)
        for i in range(self.ndim):
            (self.ptr + i).init_pointee_copy(values[i])

    fn __init__(out self, values: List[Int]):
        """
        Initialize an IndexBuffer with a list of Int values.

        Args:
            values: List of integer values.
        """
        self.ndim = len(values)
        if self.ndim <= 0:
            self.ptr = UnsafePointer[Scalar[Self.element_type], Self._origin]()
            return
        self.ptr = alloc[Scalar[Self.element_type]](self.ndim)
        for i in range(self.ndim):
            (self.ptr + i).init_pointee_copy(values[i])

    fn __init__(out self, values: VariadicList[Scalar[Self.element_type]]):
        """
        Initialize an IndexBuffer with a range of values.

        Args:
            values: Range of integer values.
        """
        self.ndim = len(values)
        if self.ndim <= 0:
            self.ptr = UnsafePointer[Scalar[Self.element_type], Self._origin]()
            return
        self.ptr = alloc[Scalar[Self.element_type]](self.ndim)
        for i in range(self.ndim):
            (self.ptr + i).init_pointee_copy(values[i])

    fn __copyinit__(out self, other: Self):
        """
        Copy-initialize an IndexBuffer from another IndexBuffer.

        Args:
            other: The other IndexBuffer to copy from.
        """
        self.ndim = other.ndim
        if other.ndim <= 0:
            self.ptr = UnsafePointer[Scalar[Self.element_type], Self._origin]()
            return
        self.ptr = alloc[Scalar[Self.element_type]](other.ndim)
        memcpy(dest=self.ptr, src=other.ptr, count=other.ndim)

    fn __del__(deinit self):
        """
        Deinitialize the IndexBuffer and free resources.
        """
        if self.ndim > 0 and self.ptr:
            self.ptr.free()

    # ===----------------------------------------------------------------------=== #
    # Element Access Methods
    # ===----------------------------------------------------------------------=== #
    fn get_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self.element_type], Self._origin]:
        """
        Get the underlying pointer of the buffer.

        Returns:
            UnsafePointer to the buffer.

        Notes:
            The returned pointer is a reference to the internal pointer.
        """
        return self.ptr

    fn offset(
        self, offset: Int
    ) -> UnsafePointer[Scalar[Self.element_type], Self._origin]:
        """
        Get a pointer offset by the given amount.

        Args:
            offset: Offset amount.

        Returns:
            UnsafePointer offset by the given amount.
        """
        return self.ptr + offset

    fn __getitem__(self, idx: Int) raises -> Scalar[Self.element_type]:
        """
        Get the element at the given index.

        Args:
            idx: Index of the element.

        Returns:
            Element at the given index.
        """
        var index = idx if idx >= 0 else self.ndim + idx
        if index < 0 or index >= self.ndim:
            raise Error(
                NumojoError(
                    category="index",
                    message="index out of bounds",
                    location="IndexBuffer.__getitem__(idx: Int)",
                )
            )
        return self.ptr[index]

    fn __getitem__(self, slice: Slice) raises -> Self:
        """
        Get a sub-buffer using a slice.

        Args:
            slice: Slice object defining the sub-buffer.

        Returns:
            Sub-buffer defined by the slice.
        """
        var step = slice.step.or_else(1)
        if step == 0:
            raise Error(
                NumojoError(
                    category="value",
                    message="slice step cannot be zero",
                    location="IndexBuffer.__getitem__(slice: Slice)",
                )
            )

        var start, end, step_value, length = InternalSlice.get_slice_info(
            slice, self.ndim
        )
        var new_buffer = Self(size=length)
        var idx = 0
        for i in range(start, end, step_value):
            new_buffer.ptr[idx] = self.ptr[i]
            idx += 1

        return new_buffer^

    fn __setitem__(mut self, idx: Int, value: Scalar[Self.element_type]) raises:
        """
        Set the element at the given index.

        Args:
            idx: Index of the element.
            value: Value to set.
        """
        var index = idx if idx >= 0 else self.ndim + idx
        if index < 0 or index >= self.ndim:
            raise Error(
                NumojoError(
                    category="index",
                    message="index out of bounds",
                    location="IndexBuffer.__setitem__(idx: Int)",
                )
            )
        self.ptr[index] = value

    fn __setitem__(mut self, slice: Slice, value: Self) raises:
        """
        Set a sub-buffer using a slice.

        Args:
            slice: Slice object defining the sub-buffer.
            value: Buffer to set.
        """
        var step = slice.step.or_else(1)
        if step == 0:
            raise Error(
                NumojoError(
                    category="value",
                    message="slice step cannot be zero",
                    location="IndexBuffer.__setitem__(slice: Slice)",
                )
            )

        var start, end, step_value, length = InternalSlice.get_slice_info(
            slice, self.ndim
        )
        if length != len(value):
            raise Error(
                NumojoError(
                    category="shape",
                    message=(
                        "attempt to assign sequence of size {} to slice of"
                        " size {}".format(len(value), length)
                    ),
                    location="IndexBuffer.__setitem__(slice: Slice)",
                )
            )

        var idx = 0
        for i in range(start, end, step_value):
            self.ptr[i] = value.ptr[idx]
            idx += 1

    fn load[width: Int = 1](self, idx: Int) -> SIMD[Self.element_type, width]:
        """
        Load a SIMD vector from the buffer at the given index.

        Parameters:
            width: Width of the SIMD vector.

        Args:
            idx: Index to load from.

        Returns:
            SIMD vector loaded from the buffer.
        """
        return self.ptr.load[width=width](idx)

    fn store[
        width: Int = 1
    ](self, idx: Int, value: SIMD[Self.element_type, width]):
        """
        Store a SIMD vector to the buffer at the given index.

        Parameters:
            width: Width of the SIMD vector.

        Args:
            idx: Index to store to.
            value: SIMD vector to store.
        """
        self.ptr.store[width=width](idx, value)

    fn unsafe_load[
        width: Int = 1
    ](self, idx: Int) -> SIMD[Self.element_type, width]:
        """
        Unsafely load a SIMD vector from the buffer at the given index.

        Parameters:
            width: Width of the SIMD vector.

        Args:
            idx: Index to load from.

        Returns:
            SIMD vector loaded from the buffer.
        """
        return self.ptr.load[width=width](idx)

    fn unsafe_store[
        width: Int = 1
    ](self, idx: Int, value: SIMD[Self.element_type, width]):
        """
        Unsafely store a SIMD vector to the buffer at the given index.

        Parameters:
            width: Width of the SIMD vector.

        Args:
            idx: Index to store to.
            value: SIMD vector to store.
        """
        self.ptr.store[width=width](idx, value)

    # ===----------------------------------------------------------------------=== #
    # Transformation Methods
    # ===----------------------------------------------------------------------=== #

    fn extend(self, *values: Int) -> Self:
        """
        Extend the buffer by appending additional integer values.

        Args:
            values: Variadic list of sizes of extended dimensions.

        Returns:
            A new IndexBuffer with the additional values appended.
        """
        var total_dims = self.ndim + len(values)
        var new_buf = Self(size=total_dims)
        for i in range(self.ndim):
            new_buf.ptr[i] = self.ptr[i]
        for j in range(len(values)):
            new_buf.ptr[self.ndim + j] = values[j]
        return new_buf^

    fn extend(self, values: List[Int]) -> Self:
        """
        Extend the buffer by appending additional integer values from a List.

        Args:
            values: List of sizes of extended dimensions.

        Returns:
            A new IndexBuffer with the additional values appended.
        """
        var total_dims = self.ndim + len(values)
        var new_buf = Self(size=total_dims)
        for i in range(self.ndim):
            new_buf.ptr[i] = self.ptr[i]
        for j in range(len(values)):
            new_buf.ptr[self.ndim + j] = values[j]
        return new_buf^

    fn flip(mut self):
        """
        Flip the items in-place.
        """
        for i in range(self.ndim // 2):
            var temp = self.ptr[i]
            self.ptr[i] = self.ptr[self.ndim - 1 - i]
            self.ptr[self.ndim - 1 - i] = temp

    fn flipped(self) -> Self:
        """
        Returns a new IndexBuffer by reversing the items.

        Returns:
            A new IndexBuffer with the items reversed.
        """
        var res = Self(size=self.ndim)
        for i in range(self.ndim):
            res.ptr[i] = self.ptr[self.ndim - 1 - i]
        return res^

    fn move_axis_to_end(self, axis: Int) -> Self:
        """
        Returns a new IndexBuffer by moving the value at axis to the end.

        Args:
            axis: The axis (index) to move. It should be in [-ndim, ndim).

        Returns:
            A new IndexBuffer with the axis moved to the end.
        """
        var ax = axis
        if ax < 0:
            ax += self.ndim
        var res = Self(size=self.ndim)
        var idx = 0
        for i in range(self.ndim):
            if i != ax:
                res.ptr[idx] = self.ptr[i]
                idx += 1
        res.ptr[self.ndim - 1] = self.ptr[ax]
        return res^

    fn pop(self, axis: Int) raises -> Self:
        """
        Drops the item at the given axis (index).

        Args:
            axis: The axis (index) to drop. It should be in [0, ndim).

        Returns:
            A new IndexBuffer with the item at the given axis dropped.
        """
        if self.ndim < 1:
            raise Error(
                NumojoError(
                    category="value",
                    message="cannot pop from an empty IndexBuffer",
                    location="IndexBuffer.pop()",
                )
            )
        var ax = axis
        if ax < 0:
            ax += self.ndim
        if ax < 0 or ax >= self.ndim:
            raise Error(
                NumojoError(
                    category="value",
                    message="axis out of bounds in IndexBuffer.pop()",
                    location="IndexBuffer.pop",
                )
            )
        var res = Self(size=self.ndim - 1)
        var idx = 0
        for i in range(self.ndim):
            if i == ax:
                continue
            res.ptr[idx] = self.ptr[i]
            idx += 1
        return res^

    fn insert(self, axis: Int, value: Int) raises -> Self:
        """
        Inserts a value at the given axis (index).

        Args:
            axis: The axis (index) to insert at. It should be in [0, ndim].
            value: The value to insert.

        Returns:
            A new IndexBuffer with the value inserted at the given axis.
        """
        if axis < 0 or axis > self.ndim:
            raise Error(
                NumojoError(
                    category="value",
                    message="axis out of bounds in IndexBuffer.insert()",
                    location="IndexBuffer.insert",
                )
            )
        var res = Self(size=self.ndim + 1)
        for i in range(axis):
            res.ptr[i] = self.ptr[i]
        res.ptr[axis] = value
        for i in range(axis, self.ndim):
            res.ptr[i + 1] = self.ptr[i]
        return res^

    fn join(self, *others: Self) -> Self:
        """
        Join multiple IndexBuffers into a single IndexBuffer.

        Args:
            others: Variable number of IndexBuffer objects.

        Returns:
            A new IndexBuffer with all values concatenated.
        """
        var total_dims = self.ndim
        for i in range(len(others)):
            total_dims += others[i].ndim

        var res = Self(size=total_dims)
        var offset = 0
        memcpy(dest=res.ptr, src=self.ptr, count=self.ndim)
        offset += self.ndim
        for i in range(len(others)):
            memcpy(
                dest=res.ptr + offset, src=others[i].ptr, count=others[i].ndim
            )
            offset += others[i].ndim
        return res^

    fn join(self, others: List[Self]) -> Self:
        """
        Join multiple IndexBuffers into a single IndexBuffer from a List.

        Args:
            others: List of IndexBuffer objects.

        Returns:
            A new IndexBuffer with all values concatenated.
        """
        var total_dims = self.ndim
        for i in range(len(others)):
            total_dims += others[i].ndim

        var res = Self(size=total_dims)
        var offset = 0
        memcpy(dest=res.ptr, src=self.ptr, count=self.ndim)
        offset += self.ndim
        for i in range(len(others)):
            memcpy(
                dest=res.ptr + offset, src=others[i].ptr, count=others[i].ndim
            )
            offset += others[i].ndim
        return res^

    fn sort(mut self, order: Bool):
        """
        Sort the IndexBuffer in-place.

        Args:
            order: If True, sort in ascending order; if False, sort in descending order.
        """
        for i in range(self.ndim):
            for j in range(0, self.ndim - i - 1):
                if (order and self.ptr[j] > self.ptr[j + 1]) or (
                    not order and self.ptr[j] < self.ptr[j + 1]
                ):
                    var temp = self.ptr[j]
                    self.ptr[j] = self.ptr[j + 1]
                    self.ptr[j + 1] = temp

    fn sorted(self, order: Bool) -> Self:
        """
        Returns a new IndexBuffer that is sorted.

        Args:
            order: If True, sort in ascending order; if False, sort in descending order.

        Returns:
            A new IndexBuffer that is sorted.
        """
        var res = Self(size=self.ndim)
        memcpy(dest=res.ptr, src=self.ptr, count=self.ndim)
        res.sort(order)
        return res^

    # ===----------------------------------------------------------------------=== #
    # Static Constructors
    # ===----------------------------------------------------------------------=== #
    @staticmethod
    fn arange(start: Int, end: Int, step: Int = 1) raises -> Self:
        """
        Create a IndexBuffer with a range of values.

        Args:
            start: Start of the range.
            end: End of the range.
            step: Step size of the range.

        Returns:
            IndexBuffer with the range of values.
        """
        if step == 0:
            raise Error(
                NumojoError(
                    category="value",
                    message="step must be non-zero in range()",
                    location="IndexBuffer.range",
                )
            )

        var size: Int
        if step > 0:
            size = max(0, (end - start + step - 1) // step)
        else:
            size = max(0, (start - end - step - 1) // (-step))

        var result = Self(size=size)
        var val = start
        for i in range(size):
            result.ptr[i] = val
            val += step

        return result^

    @staticmethod
    fn fill(size: Int, value: Int) -> Self:
        """
        Create a IndexBuffer filled with the given value.

        Args:
            size: Number of elements in the buffer.
            value: Value to fill the buffer with.

        Returns:
            IndexBuffer filled with the given value.
        """
        var res = Self(size=size)
        for i in range(size):
            res.ptr[i] = value
        return res^

    @staticmethod
    fn zeros(size: Int) -> Self:
        """
        Create a IndexBuffer filled with zeros.

        Args:
            size: Number of elements in the buffer.

        Returns:
            IndexBuffer filled with zeros.
        """
        var res = Self(size=size)
        memset_zero(res.ptr, size)
        return res^

    @staticmethod
    fn ones(size: Int) -> Self:
        """
        Create a IndexBuffer filled with ones.

        Args:
            size: Number of elements in the buffer.

        Returns:
            IndexBuffer filled with ones.
        """
        var res = Self(size=size)
        for i in range(size):
            res.ptr[i] = 1
        return res^

    @staticmethod
    fn linspace(start: Int, end: Int, num: Int) raises -> Self:
        """
        Create a IndexBuffer with linearly spaced values.

        Args:
            start: Start of the range.
            end: End of the range.
            num: Number of elements in the buffer.

        Returns:
            IndexBuffer with linearly spaced values.
        """
        if num <= 0:
            raise Error(
                NumojoError(
                    category="value",
                    message="num must be positive in linspace()",
                    location="IndexBuffer.linspace",
                )
            )

        var res = Self(size=num)
        if num == 1:
            res.ptr[0] = start
            return res^

        var step = (end - start) / (num - 1)
        for i in range(num):
            res.ptr[i] = Scalar[Self.element_type](start + i * step)

        return res^

    @staticmethod
    fn invert_permutation(perm: Self) -> Self:
        """
        Invert a permutation.

        Args:
            perm: IndexBuffer representing a permutation.

        Returns:
            IndexBuffer representing the inverse permutation.
        """
        var n = len(perm)
        var inverted = Self(size=n)
        for i in range(n):
            inverted.ptr[perm.ptr[i]] = i
        return inverted^

    # ===----------------------------------------------------------------------=== #
    # Properties
    # ===----------------------------------------------------------------------=== #
    fn rank(self) -> Int:
        """
        Get the number of elements in the IndexBuffer.

        Returns:
            Number of elements in the IndexBuffer.
        """
        return self.ndim

    fn is_empty(self) -> Bool:
        """
        Check if the IndexBuffer is empty.

        Returns:
            True if the IndexBuffer is empty, False otherwise.
        """
        return self.ndim == 0

    fn sum(self) -> Scalar[Self.element_type]:
        """
        Compute the sum of all elements in the IndexBuffer.

        Returns:
            Sum of all elements in the IndexBuffer.
        """
        var total: Scalar[Self.element_type] = 0
        # fn closure[width: Int](i: Int) unified {mut total, read self}:
        #     total += self.load[width](i).reduce_add()
        # vectorize[Self.simd_width](self.ndim, closure)
        for i in range(self.ndim):
            total += self.ptr[i]
        return total

    fn product(self) -> Scalar[Self.element_type]:
        """
        Compute the product of all elements in the IndexBuffer.

        Returns:
            Product of all elements in the IndexBuffer.
        """
        var total: Scalar[Self.element_type] = 1
        # fn closure[width: Int](i: Int) unified {mut total, read self}:
        #     total += self.load[width](i).reduce_mul()
        # vectorize[Self.simd_width](self.ndim, closure)
        for i in range(self.ndim):
            total *= self.ptr[i]
        return total

    # ===----------------------------------------------------------------------=== #
    # Traits
    # ===----------------------------------------------------------------------=== #
    fn __len__(self) -> Int:
        """
        Get the number of elements in the IndexBuffer.

        Returns:
            Number of elements in the IndexBuffer.
        """
        return self.ndim

    fn __repr__(self) -> String:
        """
        Get the official string representation of the IndexBuffer.

        Returns:
            Official string representation of the IndexBuffer.
        """
        return self.__str__()

    fn __str__(self) -> String:
        """
        Get the string representation of the IndexBuffer.

        Returns:
            String representation of the IndexBuffer.
        """
        var res = String("IndexBuffer([")
        for i in range(self.ndim):
            res += String(self.ptr[i])
            if i < self.ndim - 1:
                res += String(", ")
        res += String("])")
        return res^

    fn write_to[W: Writer](self, mut writer: W):
        """
        Write the IndexBuffer to a writer.
        """
        writer.write("IndexBuffer([")
        for i in range(self.ndim):
            writer.write(String(self.ptr[i]))
            if i < self.ndim - 1:
                writer.write(", ")
        writer.write("])")

    fn __contains__(self, value: Scalar[Self.element_type]) -> Bool:
        """
        Check if the IndexBuffer contains the given value.

        Args:
            value: Value to check for.

        Returns:
            True if the value is in the IndexBuffer, False otherwise.
        """
        for i in range(self.ndim):
            if self.ptr[i] == value:
                return True
        return False

    # TODO: We can remove this overload once Mojo unified Scalar[DType.int] with Int
    fn __contains__(self, value: Int) -> Bool:
        """
        Check if the IndexBuffer contains the given value.

        Args:
            value: Value to check for.

        Returns:
            True if the value is in the IndexBuffer, False otherwise.
        """
        for i in range(self.ndim):
            if self.ptr[i] == value:
                return True
        return False

    fn __eq__(self, other: IndexBuffer) -> Bool:
        """
        Check if two IndexBuffers are equal.

        Args:
            other: The other IndexBuffer to compare with.

        Returns:
            True if the IndexBuffers are equal, False otherwise.
        """
        if self.ndim != other.ndim:
            return False
        for i in range(self.ndim):
            if self.ptr[i] != other.ptr[i]:
                return False
        return True

    fn __ne__(self, other: IndexBuffer) -> Bool:
        """
        Check if two IndexBuffers are not equal.

        Args:
            other: The other IndexBuffer to compare with.

        Returns:
            True if the IndexBuffers are not equal, False otherwise.
        """
        return not self.__eq__(other)

    # ===----------------------------------------------------------------------=== #
    # Utility Methods
    # ===----------------------------------------------------------------------=== #
    fn init_value(mut self, idx: Int, value: Scalar[Self.element_type]):
        """
        Initialize the element at the given index.
        No bounds checking.

        Args:
            idx: Index of the element.
            value: Value to set.
        """
        self.ptr[idx] = value

    fn tolist(self) -> List[Scalar[Self.element_type]]:
        """
        Convert the buffer to a list.
        """
        var result: List[Scalar[Self.element_type]] = List[
            Scalar[Self.element_type]
        ]()
        for i in range(self.ndim):
            result.append(self.ptr[i])
        return result^

    # ===----------------------------------------------------------------------=== #
    # Iterators
    # ===----------------------------------------------------------------------=== #
    fn __iter__(ref self) -> _IndexBufferIter[origin_of(self), True]:
        """
        Get a forward iterator for the IndexBuffer.

        Returns:
            Forward iterator for the IndexBuffer.
        """
        return _IndexBufferIter[origin_of(self), True](
            Pointer(to=self), self.ndim
        )

    fn __reversed__(ref self) -> _IndexBufferIter[origin_of(self), False]:
        """
        Get a backward iterator for the IndexBuffer.

        Returns:
            Backward iterator for the IndexBuffer.
        """
        return _IndexBufferIter[origin_of(self), False](
            Pointer(to=self), self.ndim
        )


# ===----------------------------------------------------------------------=== #
# IndexBuffer Iterator
# ===----------------------------------------------------------------------=== #
struct _IndexBufferIter[
    origin: ImmutOrigin = ImmutExternalOrigin,
    forward: Bool = True,
](ImplicitlyCopyable, Movable):
    """Iterator for Item.

    Parameters:
        origin: The mutability origin of the iterator.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var item: Pointer[IndexBuffer, Self.origin]
    var length: Int

    fn __init__(
        out self,
        item: Pointer[IndexBuffer, Self.origin],
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
            return self.index
