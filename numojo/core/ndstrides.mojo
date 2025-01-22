"""
Implements NDArrayStrides type.

`NDArrayStrides` is a series of `Int` on the heap.
"""

from utils import Variant
from memory import UnsafePointer, memcpy

alias Strides = NDArrayStrides


@register_passable
struct NDArrayStrides(Stringable):
    """Implements the NDArrayStrides."""

    # Fields
    var _buf: UnsafePointer[Int]
    """Data buffer."""
    var ndim: Int
    """Number of dimensions of array."""

    @always_inline("nodebug")
    fn __init__(out self, *strides: Int):
        self.ndim = len(strides)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(strides[i])

    @always_inline("nodebug")
    fn __init__(out self, strides: List[Int]):
        self.ndim = len(strides)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(strides[i])

    @always_inline("nodebug")
    fn __init__(out self, strides: VariadicList[Int]):
        self.ndim = len(strides)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        for i in range(self.ndim):
            (self._buf + i).init_pointee_copy(strides[i])

    @always_inline("nodebug")
    fn __init__(out self, strides: NDArrayStrides):
        self.ndim = strides.ndim
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        memcpy(self._buf, strides._buf, strides.ndim)

    @always_inline("nodebug")
    fn __init__(out self, *shape: Int, order: String) raises:
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        if order == "C":
            for i in range(self.ndim):
                var temp: Int = 1
                for j in range(i + 1, self.ndim):
                    temp = temp * shape[j]
                (self._buf + i).init_pointee_copy(temp)
        elif order == "F":
            self._buf.init_pointee_copy(1)
            for i in range(0, self.ndim - 1):
                (self._buf + i + 1).init_pointee_copy(self._buf[i] * shape[i])
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(out self, shape: List[Int], order: String = "C") raises:
        self.ndim = len(shape)
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        if order == "C":
            for i in range(self.ndim):
                var temp: Int = 1
                for j in range(i + 1, self.ndim):
                    temp = temp * shape[j]
                (self._buf + i).init_pointee_copy(temp)
        elif order == "F":
            self._buf.init_pointee_copy(1)
            for i in range(0, self.ndim - 1):
                (self._buf + i + 1).init_pointee_copy(self._buf[i] * shape[i])
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: VariadicList[Int],
        order: String = "C",
    ) raises:
        self.ndim = shape.__len__()
        self._buf = UnsafePointer[Int]().alloc(self.ndim)
        if order == "C":
            for i in range(self.ndim):
                var temp: Int = 1
                for j in range(i + 1, self.ndim):
                    temp = temp * shape[j]
                (self._buf + i).init_pointee_copy(temp)
        elif order == "F":
            self._buf.init_pointee_copy(1)
            for i in range(0, self.ndim - 1):
                (self._buf + i + 1).init_pointee_copy(self._buf[i] * shape[i])
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        out self,
        owned shape: NDArrayShape,
        order: String = "C",
    ) raises:
        self.ndim = shape.ndim
        self._buf = UnsafePointer[Int]().alloc(shape.ndim)
        if order == "C":
            if shape.ndim == 1:
                self._buf.init_pointee_copy(1)
            else:
                for i in range(shape.ndim):
                    var temp: Int = 1
                    for j in range(i + 1, shape.ndim):
                        temp = temp * shape[j]
                    (self._buf + i).init_pointee_copy(temp)
        elif order == "F":
            self._buf.init_pointee_copy(1)
            for i in range(0, self.ndim - 1):
                (self._buf + i + 1).init_pointee_copy(self._buf[i] * shape[i])
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        self.ndim = other.ndim
        self._buf = UnsafePointer[Int]().alloc(other.ndim)
        memcpy(self._buf, other._buf, other.ndim)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        if index >= self.ndim:
            raise Error("Index out of bound")
        if index >= 0:
            return self._buf[index]
        else:
            return self._buf[self.ndim + index]

    @always_inline("nodebug")
    fn __setitem__(mut self, index: Int, val: Int) raises:
        if index >= self.ndim:
            raise Error("Index out of bound")
        if index >= 0:
            self._buf[index] = val
        else:
            self._buf[self.ndim + index] = val

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return self.ndim

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """
        Return a string of the strides of the array.
        """
        return "numojo.Strides" + str(self)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """
        Return a string of the strides of the array.
        """
        var result: String = "("
        for i in range(self.ndim):
            result += str(self._buf[i])
            if i < self.ndim - 1:
                result += ","
        result = result + ")"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Strides: " + str(self) + "  " + "ndim: " + str(self.ndim))

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Bool:
        for i in range(self.ndim):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, val: Int) raises -> Bool:
        for i in range(self.ndim):
            if self[i] == val:
                return True
        return False

    # ===-------------------------------------------------------------------===#
    # Other private methods
    # ===-------------------------------------------------------------------===#

    fn _flip(self) raises -> Self:
        """
        Returns a new strides by flipping the items.

        UNSAFE! No boundary check!

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

    fn _move_axis_to_end(self, owned axis: Int) raises -> Self:
        """
            Returns a new strides by moving the value of axis to the end.

            UNSAFE! No boundary check!

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

        var value = strides[axis]
        for i in range(axis, strides.ndim - 1):
            strides._buf[i] = strides._buf[i + 1]
        strides._buf[strides.ndim - 1] = value
        return strides

    fn _pop(self, axis: Int) -> Self:
        """
        drop information of certain axis.
        """
        var res = Self()
        var buffer = UnsafePointer[Int].alloc(self.ndim - 1)
        memcpy(dest=buffer, src=self._buf, count=axis)
        memcpy(
            dest=buffer + axis,
            src=self._buf.offset(axis + 1),
            count=self.ndim - axis - 1,
        )
        res.ndim = self.ndim - 1
        res._buf = buffer
        return res


# @always_inline("nodebug")
# fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
#     # if index >= self.ndim:
#     #     raise Error("Index out of bound")
#     return self._buf.ptr.load[width=width](index)

# @always_inline("nodebug")
# fn store[
#     width: Int = 1
# ](mut self, index: Int, val: SIMD[dtype, width]) raises:
#     # if index >= self.ndim:
#     #     raise Error("Index out of bound")
#     self._buf.ptr.store(index, val)

# @always_inline("nodebug")
# fn load_unsafe[width: Int = 1](self, index: Int) -> Int:
#     return self._buf.ptr.load[width=width](index).__int__()

# @always_inline("nodebug")
# fn store_unsafe[
#     width: Int = 1
# ](mut self, index: Int, val: SIMD[dtype, width]):
#     self._buf.ptr.store(index, val)
