"""
Implements NDArrayStride type.

`NDArrayShape` is a series of `DType.int32` on the heap.
"""

from utils import Variant
from builtin.type_aliases import AnyLifetime
from memory import memset_zero, memcpy


@register_passable("trivial")
struct NDArrayStride[dtype: DType = DType.int32](Stringable, Formattable):
    """Implements the NDArrayStride."""

    # Fields
    var ndoffset: Int
    var ndstride: UnsafePointer[Scalar[dtype]]
    var ndlen: Int

    @always_inline("nodebug")
    fn __init__(
        inout self, *stride: Int, offset: Int = 0
    ):  # maybe we should add checks for offset?
        self.ndoffset = offset
        self.ndlen = stride.__len__()
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(stride.__len__())
        for i in range(stride.__len__()):
            self.ndstride[i] = stride[i]

    @always_inline("nodebug")
    fn __init__(inout self, stride: List[Int], offset: Int = 0):
        self.ndoffset = offset
        self.ndlen = stride.__len__()
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        for i in range(self.ndlen):
            self.ndstride[i] = stride[i]

    @always_inline("nodebug")
    fn __init__(inout self, stride: VariadicList[Int], offset: Int = 0):
        self.ndoffset = offset
        self.ndlen = stride.__len__()
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        for i in range(self.ndlen):
            self.ndstride[i] = stride[i]

    @always_inline("nodebug")
    fn __init__(inout self, stride: NDArrayStride[dtype]):
        self.ndoffset = stride.ndoffset
        self.ndlen = stride.ndlen
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(stride.ndlen)
        for i in range(self.ndlen):
            self.ndstride[i] = stride.ndstride[i]

    @always_inline("nodebug")
    fn __init__(
        inout self, stride: NDArrayStride[dtype], offset: Int = 0
    ):  # separated two methods to remove if condition
        self.ndoffset = offset
        self.ndlen = stride.ndlen
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(stride.ndlen)
        for i in range(self.ndlen):
            self.ndstride[i] = stride.ndstride[i]

    @always_inline("nodebug")
    fn __init__(
        inout self, *shape: Int, offset: Int = 0, order: String = "C"
    ) raises:
        self.ndoffset = offset
        self.ndlen = shape.__len__()
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        if order == "C":
            for i in range(self.ndlen):
                var temp: Int = 1
                for j in range(i + 1, self.ndlen):
                    temp = temp * shape[j]
                self.ndstride[i] = temp
        elif order == "F":
            self.ndstride[0] = 1
            for i in range(0, self.ndlen - 1):
                self.ndstride[i + 1] = self.ndstride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        inout self, shape: List[Int], offset: Int = 0, order: String = "C"
    ) raises:
        self.ndoffset = offset
        self.ndlen = shape.__len__()
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        if order == "C":
            for i in range(self.ndlen):
                var temp: Int = 1
                for j in range(i + 1, self.ndlen):
                    temp = temp * shape[j]
                self.ndstride[i] = temp
        elif order == "F":
            self.ndstride[0] = 1
            for i in range(0, self.ndlen - 1):
                self.ndstride[i + 1] = self.ndstride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        inout self,
        shape: VariadicList[Int],
        offset: Int = 0,
        order: String = "C",
    ) raises:
        self.ndoffset = offset
        self.ndlen = shape.__len__()
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(self.ndlen)
        memset_zero(self.ndstride, self.ndlen)
        if order == "C":
            for i in range(self.ndlen):
                var temp: Int = 1
                for j in range(i + 1, self.ndlen):
                    temp = temp * shape[j]
                self.ndstride[i] = temp
        elif order == "F":
            self.ndstride[0] = 1
            for i in range(0, self.ndlen - 1):
                self.ndstride[i + 1] = self.ndstride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    @always_inline("nodebug")
    fn __init__(
        inout self,
        owned shape: NDArrayShape,
        offset: Int = 0,
        order: String = "C",
    ) raises:
        self.ndoffset = offset
        self.ndlen = shape.ndlen
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(shape.ndlen)
        memset_zero(self.ndstride, shape.ndlen)
        if order == "C":
            if shape.ndlen == 1:
                self.ndstride[0] = 1
            else:
                for i in range(shape.ndlen):
                    var temp: Int = 1
                    for j in range(i + 1, shape.ndlen):
                        temp = temp * shape[j]
                    self.ndstride[i] = temp
        elif order == "F":
            self.ndstride[0] = 1
            for i in range(0, self.ndlen - 1):
                self.ndstride[i + 1] = self.ndstride[i] * shape[i]
        else:
            raise Error(
                "Invalid order: Only C style row major `C` & Fortran style"
                " column major `F` are supported"
            )

    fn __copy__(inout self, other: Self):
        self.ndoffset = other.ndoffset
        self.ndlen = other.ndlen
        self.ndstride = UnsafePointer[Scalar[dtype]]().alloc(other.ndlen)
        memcpy(self.ndstride, other.ndstride, other.ndlen)

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) raises -> Int:
        if index >= self.ndlen:
            raise Error("Index out of bound")
        if index >= 0:
            return self.ndstride[index].__int__()
        else:
            return self.ndstride[self.ndlen + index].__int__()

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, val: Int) raises:
        if index >= self.ndlen:
            raise Error("Index out of bound")
        if index >= 0:
            self.ndstride[index] = val
        else:
            self.ndstride[self.ndlen + index] = val

    @always_inline("nodebug")
    fn len(self) -> Int:
        return self.ndlen

    @always_inline("nodebug")
    fn __str__(self) -> String:
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        var result: String = "Stride: ["
        for i in range(self.ndlen):
            if i == self.ndlen - 1:
                result += self.ndstride[i].__str__()
            else:
                result += self.ndstride[i].__str__() + ", "
        result = result + "]"
        writer.write(result)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Bool:
        for i in range(self.ndlen):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, val: Int) raises -> Bool:
        for i in range(self.ndlen):
            if self[i] == val:
                return True
        return False

    @always_inline("nodebug")
    fn load[width: Int = 1](self, index: Int) raises -> SIMD[dtype, width]:
        # if index >= self.ndlen:
        #     raise Error("Index out of bound")
        return self.ndstride.load[width=width](index)

    @always_inline("nodebug")
    fn store[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[dtype, width]) raises:
        # if index >= self.ndlen:
        #     raise Error("Index out of bound")
        self.ndstride.store[width=width](index, val)

    @always_inline("nodebug")
    fn load_unsafe[width: Int = 1](self, index: Int) -> Int:
        return self.ndstride.load[width=width](index).__int__()

    @always_inline("nodebug")
    fn store_unsafe[
        width: Int = 1
    ](inout self, index: Int, val: SIMD[dtype, width]):
        self.ndstride.store[width=width](index, val)
