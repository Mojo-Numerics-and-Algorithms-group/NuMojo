"""
Implements NDArrayStrides type.

`NDArrayStrides` is a series of `DType.index` on the heap.
"""

from utils import Variant
from builtin.type_aliases import Origin
from memory import UnsafePointer, memset_zero, memcpy


@register_passable("trivial")
struct NDArrayStrides[dtype: DType = DType.index](Stringable):
    """Implements the NDArrayStrides."""

    # Fields
    var offset: Int
    var _buf: UnsafePointer[Scalar[dtype]]
    var ndim: Int

    @always_inline("nodebug")
    fn __init__(
        mut self, *strides: Int, offset: Int = 0
    ):  # maybe we should add checks for offset?
        self.offset = offset
        self.ndim = strides.__len__()
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(strides.__len__())
        for i in range(strides.__len__()):
            self._buf[i] = strides[i]

    @always_inline("nodebug")
    fn __init__(mut self, strides: List[Int], offset: Int = 0):
        self.offset = offset
        self.ndim = strides.__len__()
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(self.ndim)
        memset_zero(self._buf, self.ndim)
        for i in range(self.ndim):
            self._buf[i] = strides[i]

    @always_inline("nodebug")
    fn __init__(mut self, strides: VariadicList[Int], offset: Int = 0):
        self.offset = offset
        self.ndim = strides.__len__()
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(self.ndim)
        memset_zero(self._buf, self.ndim)
        for i in range(self.ndim):
            self._buf[i] = strides[i]

    @always_inline("nodebug")
    fn __init__(mut self, strides: NDArrayStrides[dtype]):
        self.offset = strides.offset
        self.ndim = strides.ndim
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(strides.ndim)
        for i in range(self.ndim):
            self._buf[i] = strides._buf[i]

    @always_inline("nodebug")
    fn __init__(
        mut self, strides: NDArrayStrides[dtype], offset: Int = 0
    ):  # separated two methods to remove if condition
        self.offset = offset
        self.ndim = strides.ndim
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(strides.ndim)
        for i in range(self.ndim):
            self._buf[i] = strides._buf[i]

    @always_inline("nodebug")
    fn __init__(
        mut self, *shape: Int, offset: Int = 0, order: String = "C"
    ) raises:
        self.offset = offset
        self.ndim = shape.__len__()
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(self.ndim)
        memset_zero(self._buf, self.ndim)
        if order == "C":
            for i in range(self.ndim):
                var temp: Int = 1
                for j in range(i + 1, self.ndim):
                    temp = temp * shape[j]
                self._buf[i] = temp
        elif order == "F":
            self._buf[0] = 1
            for i in range(0, self.ndim - 1):
                self._buf[i + 1] = self._buf[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        mut self, shape: List[Int], offset: Int = 0, order: String = "C"
    ) raises:
        self.offset = offset
        self.ndim = shape.__len__()
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(self.ndim)
        memset_zero(self._buf, self.ndim)
        if order == "C":
            for i in range(self.ndim):
                var temp: Int = 1
                for j in range(i + 1, self.ndim):
                    temp = temp * shape[j]
                self._buf[i] = temp
        elif order == "F":
            self._buf[0] = 1
            for i in range(0, self.ndim - 1):
                self._buf[i + 1] = self._buf[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        mut self,
        shape: VariadicList[Int],
        offset: Int = 0,
        order: String = "C",
    ) raises:
        self.offset = offset
        self.ndim = shape.__len__()
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(self.ndim)
        memset_zero(self._buf, self.ndim)
        if order == "C":
            for i in range(self.ndim):
                var temp: Int = 1
                for j in range(i + 1, self.ndim):
                    temp = temp * shape[j]
                self._buf[i] = temp
        elif order == "F":
            self._buf[0] = 1
            for i in range(0, self.ndim - 1):
                self._buf[i + 1] = self._buf[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        mut self,
        owned shape: NDArrayShape,
        offset: Int = 0,
        order: String = "C",
    ) raises:
        self.offset = offset
        self.ndim = shape.ndim
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(shape.ndim)
        memset_zero(self._buf, shape.ndim)
        if order == "C":
            if shape.ndim == 1:
                self._buf[0] = 1
            else:
                for i in range(shape.ndim):
                    var temp: Int = 1
                    for j in range(i + 1, shape.ndim):
                        temp = temp * shape[j]
                    self._buf[i] = temp
        elif order == "F":
            self._buf[0] = 1
            for i in range(0, self.ndim - 1):
                self._buf[i + 1] = self._buf[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    fn __copy__(mut self, other: Self):
        self.offset = other.offset
        self.ndim = other.ndim
        self._buf = UnsafePointer[Scalar[dtype]]().alloc(other.ndim)
        memcpy(self._buf, other._buf, other.ndim)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        if index >= self.ndim:
            raise Error("Index out of bound")
        if index >= 0:
            return self._buf[index].__int__()
        else:
            return self._buf[self.ndim + index].__int__()

    @always_inline("nodebug")
    fn __setitem__(mut self, index: Int, val: Int) raises:
        if index >= self.ndim:
            raise Error("Index out of bound")
        if index >= 0:
            self._buf[index] = val
        else:
            self._buf[self.ndim + index] = val

    @always_inline("nodebug")
    fn len(self) -> Int:
        return self.ndim

    @always_inline("nodebug")
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        var result: String = "Stride: ["
        for i in range(self.ndim):
            if i == self.ndim - 1:
                result += self._buf[i].__str__()
            else:
                result += self._buf[i].__str__() + ", "
        result = result + "]"
        writer.write(result)

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

    @always_inline("nodebug")
    fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
        # if index >= self.ndim:
        #     raise Error("Index out of bound")
        return self._buf.load[width=width](index)

    @always_inline("nodebug")
    fn store[
        width: Int = 1
    ](mut self, index: Int, val: SIMD[dtype, width]) raises:
        # if index >= self.ndim:
        #     raise Error("Index out of bound")
        self._buf.store(index, val)

    @always_inline("nodebug")
    fn load_unsafe[width: Int = 1](self, index: Int) -> Int:
        return self._buf.load[width=width](index).__int__()

    @always_inline("nodebug")
    fn store_unsafe[
        width: Int = 1
    ](mut self, index: Int, val: SIMD[dtype, width]):
        self._buf.store(index, val)
