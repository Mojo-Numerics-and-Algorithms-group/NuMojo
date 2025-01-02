"""
Dear insiders,

This file is for internal discussion only.

I would like to demonstrate an approach to construct views of array.
This approach does not requires creation of an extra struct `NDArrayView`.
The view itself is a special case of `NDArray`.
Thus, we do not need to bother adding tons of permutations of operations 
between `NDArray` and `NDArrayView`.

In order to achieve this, we need to add one layer of container which holds
the underlying buffer. The container can either be `OwnedData`, which is 
`NDArray` as we know, or be `RefData`, which is the view we want to construct.

The code works well on this example Matrix type.
Because parameterized traits are not supported yet by Mojo,
I use Float16 as the underlying data type.

If you find this approach good, we can apply it on `NDArray` type in future.

Sincerely,
Yuhao Zhu
"""

from memory import UnsafePointer, memcpy
from sys import simdwidthof
from algorithm import parallelize, vectorize
from python import PythonObject, Python
from random import random_float64
from math import ceil

# ===----------------------------------------------------------------------===#
# Matrix16 struct
# ===----------------------------------------------------------------------===#


trait Bufferable:
    # TODO: fields in traits are not supported yet by Mojo
    # Use `get_ptr`
    # var ptr: UnsafePointer[Float16]
    # var size: Int

    fn __init__(out self, size: Int):
        ...

    fn __init__(out self, ptr: UnsafePointer[Float16], offset: Int = 0):
        ...

    fn __copyinit__(out self, other: Self):
        ...

    fn __moveinit__(out self, owned other: Self):
        ...

    fn get_ptr(self) -> UnsafePointer[Float16]:
        ...


struct OwnedData(Bufferable):
    var ptr: UnsafePointer[Float16]

    fn __init__(out self, size: Int):
        self.ptr = UnsafePointer[Float16]().alloc(size)

    fn __init__(out self, ptr: UnsafePointer[Float16], offset: Int = 0):
        self.ptr = ptr + offset

    fn __copyinit__(out self, other: Self):
        self.ptr = other.ptr

    fn __moveinit__(out self, owned other: Self):
        self.ptr = other.ptr

    fn get_ptr(self) -> UnsafePointer[Float16]:
        return self.ptr


struct RefData[is_mutable: Bool, //, origin: Origin[is_mutable]](Bufferable):
    var ptr: UnsafePointer[Float16]

    fn __init__(out self, size: Int):
        self.ptr = UnsafePointer[Float16]().alloc(size)

    fn __init__(out self, ptr: UnsafePointer[Float16], offset: Int = 0):
        self.ptr = ptr + offset

    fn __copyinit__(out self, other: Self):
        self.ptr = other.ptr

    fn __moveinit__(out self, owned other: Self):
        self.ptr = other.ptr

    fn get_ptr(self) -> UnsafePointer[Float16]:
        return self.ptr


# alias Matrix16 = Matrix16[Buffer=OwnedData]


struct Matrix16[Buffer: Bufferable = OwnedData](Stringable, Writable):
    """
    `Matrix16` is a special case of fixed-dimensional array.
    It can either own data (alias `Matrix16`) or not.

    Parameters:
        Buffer: Buffer type. Default to `OwnedData`.

    The matrix can be uniquely defined by the following features:
        1. The data buffer of all items.
        2. The shape of the matrix.
        3. The strides of the matrix.

    Attributes:
        - _buf (saved as row-majored, C-type)
        - shape
        - size (shape[0] * shape[1])
        - strides (shape[1], 1)

    Default constructor:
    - , shape
    - , data

    """

    var shape: Tuple[Int, Int]
    """Shape of Matrix16."""

    # To be calculated at the initialization if not passed in
    var size: Int
    """Size of Matrix16."""
    var strides: Tuple[Int, Int]
    """Strides of matrix."""

    # To be filled by constructing functions.
    var _buf: Buffer
    """Data buffer of the items in the NDArray."""

    var base: Bool
    """Whether data buffer is owned by self."""

    alias width: Int = simdwidthof[DType.float16]()
    """Vector size of the data type."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(out self, shape: Tuple[Int, Int], order: String = "C") raises:
        """
        Initialize array that owns its data.

        Args:
            shape: List of shape.
            order: C or F.
        """

        self.shape = (shape[0], shape[1])
        if order == "C":
            self.strides = (shape[1], 1)
        elif order == "F":
            self.strides = (1, shape[0])
        else:
            raise Error(String("Invalid order: {}").format(order))
        self.size = shape[0] * shape[1]
        self._buf = Buffer(size=self.size)
        self.base = True

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: Tuple[Int, Int],
        strides: Tuple[Int, Int],
        offset: Int,
        ptr: UnsafePointer[Float16],
    ):
        """
        Initialize array that does not own the data.
        The data is owned by another array.

        Args:
            shape: Shape of the view.
            strides: Strides of the view.
            offset: Offset in pointer of the data buffer.
            ptr: Pointer to the data buffer of the original array.
        """

        self.shape = shape
        self.strides = strides
        self.size = shape[0] * shape[1]
        self._buf = Buffer(ptr=ptr.offset(offset))
        self.base = False

    @always_inline("nodebug")
    fn __copyinit__(mut self, other: Self):
        """
        Copy other into self.
        """
        self.shape = (other.shape[0], other.shape[1])
        self.strides = (other.strides[0], other.strides[1])
        self.size = other.size
        self._buf = Buffer(other.size)
        memcpy(self._buf.get_ptr(), other._buf.get_ptr(), other.size)
        self.base = other.base

    @always_inline("nodebug")
    fn __moveinit__(mut self, owned other: Self):
        """
        Move other into self.
        """
        self.shape = other.shape^
        self.strides = other.strides^
        self.size = other.size
        self._buf = other._buf^
        self.base = other.base

    @always_inline("nodebug")
    fn __del__(owned self):
        if self.base:
            self._buf.get_ptr().free()

    # ===-------------------------------------------------------------------===#
    # Dunder methods
    # ===-------------------------------------------------------------------===#

    fn __getitem__(self, owned x: Int, owned y: Int) raises -> Float16:
        """
        Return the scalar at the index.

        Args:
            x: The row number.
            y: The column number.

        Returns:
            A scalar matching the dtype of the array.
        """

        if x < 0:
            x = self.shape[0] + x

        if y < 0:
            y = self.shape[1] + y

        if (x >= self.shape[0]) or (y >= self.shape[1]):
            raise Error(
                String(
                    "Index ({}, {}) exceed the matrix shape ({}, {})"
                ).format(x, y, self.shape[0], self.shape[1])
            )

        return self._buf.get_ptr()[x * self.strides[0] + y * self.strides[1]]

    fn __getitem__(
        ref self, owned x: Int
    ) -> Matrix16[RefData[__origin_of(self)]]:
        """
        Return the corresponding row at the index.

        Args:
            x: The row number.
        """

        var res = Matrix16[RefData[__origin_of(self)]](
            shape=(1, self.shape[1]),
            strides=(self.strides[0], self.strides[1]),
            offset=x * self.strides[0],
            ptr=self._buf.get_ptr(),
        )
        return res

    fn __getitem__(
        ref self, x: Slice, y: Slice
    ) -> Matrix16[RefData[__origin_of(self)]]:
        """
        Get item from two slices.
        """
        start_x, end_x, step_x = x.indices(self.shape[0])
        start_y, end_y, step_y = y.indices(self.shape[1])

        # The new matrix with the corresponding shape
        var res = Matrix16[RefData[__origin_of(self)]](
            shape=(
                int(ceil((end_x - start_x) / step_x)),
                int(ceil((end_y - start_y) / step_y)),
            ),
            strides=(step_x * self.strides[0], step_y * self.strides[1]),
            offset=start_x * self.strides[0] + start_y * self.strides[1],
            ptr=self._buf.get_ptr(),
        )

        return res

    fn __setitem__(self, x: Int, y: Int, value: Float16) raises:
        """
        Return the scalar at the index.

        Args:
            x: The row number.
            y: The column number.
            value: The value to be set.
        """

        if (x >= self.shape[0]) or (y >= self.shape[1]):
            raise Error(
                String(
                    "Index ({}, {}) exceed the matrix shape ({}, {})"
                ).format(x, y, self.shape[0], self.shape[1])
            )

        self._buf.get_ptr().store(
            x * self.strides[0] + y * self.strides[1], value
        )

    fn __setitem__(self, owned x: Int, value: Self) raises:
        """
        Set the corresponding row at the index with the given matrix.

        Args:
            x: The row number.
            value: Matrix16 (row vector).
        """

        if x < 0:
            x = self.shape[0] + x

        if x >= self.shape[0]:
            raise Error(
                String(
                    "Error: Elements of `index` ({}) \n"
                    "exceed the matrix shape ({})."
                ).format(x, self.shape[0])
            )

        if value.shape[0] != 1:
            raise Error(
                String(
                    "Error: The value should has only 1 row, "
                    "but it has {} rows."
                ).format(value.shape[0])
            )

        if self.shape[1] != value.shape[1]:
            raise Error(
                String(
                    "Error: Matrix16 has {} columns, "
                    "but the value has {} columns."
                ).format(self.shape[1], value.shape[1])
            )

        if (self.strides[1] == 1) and (
            value.strides[1] == 1
        ):  # Continuous memory
            var ptr = self._buf.get_ptr().offset(x * self.shape[1])
            memcpy(ptr, value._buf.get_ptr(), value.size)
        else:
            for col in range(value.size):
                self._buf.get_ptr()[
                    x * self.strides[0] + col * self.strides[1]
                ] = value[1, col]

    fn _store[
        width: Int = 1
    ](mut self, x: Int, y: Int, simd: SIMD[DType.float16, width]):
        """
        `__setitem__` with width.
        Unsafe: No boundary check!
        """
        self._buf.get_ptr().store(x * self.strides[0] + y, simd)

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        fn print_row(self: Self, row: Int, sep: String) raises -> String:
            var result: String = str("[")
            var number_of_sep: Int = 1
            if self.shape[1] <= 6:
                for col in range(self.shape[1]):
                    if col == self.shape[1] - 1:
                        number_of_sep = 0
                    result += str(self[row, col]) + sep * number_of_sep
            else:
                for col in range(3):
                    result += str(self[row, col]) + sep
                result += str("...") + sep
                for col in range(self.shape[1] - 3, self.shape[1]):
                    if col == self.shape[1] - 1:
                        number_of_sep = 0
                    result += str(self[row, col]) + sep * number_of_sep
            result += str("]")
            return result

        var sep: String = str("\t")
        var newline: String = str("\n ")
        var number_of_newline: Int = 1
        var result: String = "["

        try:
            if self.shape[0] <= 6:
                for row in range(self.shape[0]):
                    if row == self.shape[0] - 1:
                        number_of_newline = 0
                    result += (
                        print_row(self, row, sep) + newline * number_of_newline
                    )
            else:
                for row in range(3):
                    result += print_row(self, row, sep) + newline
                result += str("...") + newline
                for row in range(self.shape[0] - 3, self.shape[0]):
                    if row == self.shape[0] - 1:
                        number_of_newline = 0
                    result += (
                        print_row(self, row, sep) + newline * number_of_newline
                    )
        except e:
            print("Cannot transfer matrix to string!", e)
        result += str("]")
        writer.write(
            result
            + "\nShape: "
            + str(self.shape[0])
            + "x"
            + str(self.shape[1])
            + "  Strides: "
            + "("
            + str(self.strides[0])
            + ","
            + str(self.strides[1])
            + ")  DType: "
            + str("Float16")
            + "  Base: "
            + str(self.base)
        )

    fn __iter__(ref self) raises -> _Matrix16Iter[__origin_of(self)]:
        """Iterate over elements of the Matrix16, returning copied value.

        Returns:
            An iterator of Matrix16 elements.
        """

        return _Matrix16Iter[__origin_of(self)](
            ptr=self._buf.get_ptr(),
            shape1=self.shape[1],
            length=self.shape[0],
        )

    fn __reversed__(
        ref self,
    ) raises -> _Matrix16Iter[__origin_of(self), forward=False]:
        """Iterate backwards over elements of the Matrix16, returning
        copied value.

        Returns:
            A reversed iterator of Matrix16 elements.
        """

        return _Matrix16Iter[__origin_of(self), forward=False](
            ptr=self._buf.get_ptr(),
            shape1=self.shape[1],
            length=self.shape[0],
        )


# ===-----------------------------------------------------------------------===#
# MatrixIter struct
# ===-----------------------------------------------------------------------===#


@value
struct _Matrix16Iter[
    is_mutable: Bool, //,
    origin: Origin[is_mutable],
    forward: Bool = True,
]:
    """Iterator for Matrix16.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        origin: The lifetime of the underlying Matrix16 data.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var ptr: UnsafePointer[Float16]
    var shape1: Int
    var length: Int

    fn __init__(
        out self,
        ptr: UnsafePointer[Float16],
        shape1: Int,
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.shape1 = shape1
        self.ptr = ptr

    fn __iter__(ref self) -> Self:
        return self

    fn __next__(mut self) -> Matrix16[RefData[origin]]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return Matrix16[RefData[origin]](
                shape=(1, self.shape1),
                strides=(self.shape1, 1),
                offset=current_index * self.shape1,
                ptr=self.ptr,
            )
        else:
            var current_index = self.index
            self.index -= 1
            return Matrix16[RefData[origin]](
                shape=(1, self.shape1),
                strides=(self.shape1, 1),
                offset=current_index * self.shape1,
                ptr=self.ptr,
            )

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index

    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index > 0


# ===-----------------------------------------------------------------------===#
# Functions
# ===-----------------------------------------------------------------------===#


fn rand(shape: Tuple[Int, Int]) raises -> Matrix16:
    """Return a matrix with random values uniformed distributed between 0 and 1.
    """
    var result = Matrix16(shape)
    for i in range(result.size):
        result._buf.get_ptr().store(
            i, random_float64(0, 1).cast[DType.float16]()
        )
    return result^


fn flip(ref a: Matrix16, axis: Int) raises -> Matrix16[RefData[__origin_of(a)]]:
    """
    Return flipped array.
    """

    if axis == 0:
        return Matrix16[RefData[__origin_of(a)]](
            shape=(a.shape[1], a.shape[0]),
            strides=(-a.strides[0], a.strides[1]),
            offset=(a.shape[0] - 1) * a.strides[0] + 0 * a.strides[1],
            ptr=a._buf.get_ptr(),
        )
    elif axis == 1:
        return Matrix16[RefData[__origin_of(a)]](
            shape=(a.shape[1], a.shape[0]),
            strides=(a.strides[0], -a.strides[1]),
            offset=0 * a.strides[0] + (a.shape[1] - 1) * a.strides[1],
            ptr=a._buf.get_ptr(),
        )
    else:
        raise Error("Invalid axis")


fn transpose(ref a: Matrix16) -> Matrix16[RefData[__origin_of(a)]]:
    """
    Return transposed array.
    """

    var res = Matrix16[RefData[__origin_of(a)]](
        shape=(a.shape[1], a.shape[0]),
        strides=(a.strides[1], a.strides[0]),
        offset=0,
        ptr=a._buf.get_ptr(),
    )
    return res
