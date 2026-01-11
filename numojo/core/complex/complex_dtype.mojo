# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt

# Portions of this code are derived from the Modular Mojo repository
# Copyright (c) 2025, Modular Inc. All rights reserved.
# Original source: https://github.com/modularml/mojo
# ===----------------------------------------------------------------------=== #

from hashlib.hasher import Hasher
from os import abort
from sys import CompilationTarget
from sys.info import bit_width_of, size_of
from sys.intrinsics import _type_is_eq

comptime _mIsSigned = UInt8(1)
comptime _mIsInteger = UInt8(1 << 7)
comptime _mIsNotInteger = UInt8(~(1 << 7))
comptime _mIsFloat = UInt8(1 << 6)

# rust like comptimes for complex data types.
comptime ci8 = ComplexDType.int8
"""Data type alias for ComplexDType.int8"""
comptime ci16 = ComplexDType.int16
"""Data type alias for ComplexDType.int16"""
comptime ci32 = ComplexDType.int32
"""Data type alias for ComplexDType.int32"""
comptime ci64 = ComplexDType.int64
"""Data type alias for ComplexDType.int64"""
comptime ci128 = ComplexDType.int128
"""Data type alias for ComplexDType.int128"""
comptime ci256 = ComplexDType.int256
"""Data type alias for ComplexDType.int256"""
comptime cint = ComplexDType.int
"""Data type alias for ComplexDType.int"""
comptime cu8 = ComplexDType.uint8
"""Data type alias for ComplexDType.uint8"""
comptime cu16 = ComplexDType.uint16
"""Data type alias for ComplexDType.uint16"""
comptime cu32 = ComplexDType.uint32
"""Data type alias for ComplexDType.uint32"""
comptime cu64 = ComplexDType.uint64
"""Data type alias for ComplexDType.uint64"""
comptime cu128 = ComplexDType.uint128
"""Data type alias for ComplexDType.uint128"""
comptime cu256 = ComplexDType.uint256
"""Data type alias for ComplexDType.uint256"""
comptime cuint = ComplexDType.uint
"""Data type alias for ComplexDType.uint"""
comptime cbf16 = ComplexDType.bfloat16
"""Data type alias for ComplexDType.bfloat16"""
comptime cf16 = ComplexDType.float16
"""Data type alias for ComplexDType.float16"""
comptime cf32 = ComplexDType.float32
"""Data type alias for ComplexDType.float32"""
comptime cf64 = ComplexDType.float64
"""Data type alias for ComplexDType.float64"""
comptime cboolean = ComplexDType.bool
"""Data type alias for ComplexDType.bool"""
comptime cinvalid = ComplexDType.invalid
"""Data type alias for ComplexDType.invalid"""

# ===----------------------------------------------------------------------=== #
# Implements the Complex Datatype.
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct ComplexDType(
    Copyable,
    EqualityComparable,
    Hashable,
    Identifiable,
    KeyElement,
    Movable,
    Representable,
    Stringable,
    Writable,
):
    """
    Represents a complex data type specification and provides methods for working
    with it.

    `ComplexDType` behaves like an enum rather than a typical object. You don't
    instantiate it, but instead use its compile-time constants (aliases) to
    declare data types for complex SIMD vectors, tensors, and other data structures.
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#
    # Refer to DType documentation for details on each data type.
    comptime _mlir_type = __mlir_type.`!kgen.dtype`
    comptime invalid = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<invalid> : !kgen.dtype`
    )
    comptime bool = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<bool> : !kgen.dtype`
    )
    comptime int = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`
    )
    comptime uint = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<uindex> : !kgen.dtype`
    )
    comptime _uint1 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui1> : !kgen.dtype`
    )
    comptime _uint2 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui2> : !kgen.dtype`
    )
    comptime _uint4 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui4> : !kgen.dtype`
    )
    comptime uint8 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui8> : !kgen.dtype`
    )
    comptime int8 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si8> : !kgen.dtype`
    )
    comptime uint16 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui16> : !kgen.dtype`
    )
    comptime int16 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si16> : !kgen.dtype`
    )
    comptime uint32 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui32> : !kgen.dtype`
    )
    comptime int32 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`
    )
    comptime uint64 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui64> : !kgen.dtype`
    )
    comptime int64 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si64> : !kgen.dtype`
    )
    comptime uint128 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui128> : !kgen.dtype`
    )
    comptime int128 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si128> : !kgen.dtype`
    )
    comptime uint256 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui256> : !kgen.dtype`
    )
    comptime int256 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si256> : !kgen.dtype`
    )
    comptime float8_e3m4 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f8e3m4> : !kgen.dtype`
    )
    comptime float8_e4m3fn = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f8e4m3fn> : !kgen.dtype`
    )
    comptime float8_e4m3fnuz = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f8e4m3fnuz> : !kgen.dtype`
    )
    comptime float8_e5m2 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f8e5m2> : !kgen.dtype`
    )
    comptime float8_e5m2fnuz = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f8e5m2fnuz> : !kgen.dtype`
    )
    comptime bfloat16 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<bf16> : !kgen.dtype`
    )
    comptime float16 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f16> : !kgen.dtype`
    )
    comptime float32 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`
    )
    comptime float64 = ComplexDType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`
    )

    # ===----------------------------------------------------------------------=== #
    # Fields.
    # ===----------------------------------------------------------------------=== #

    var _dtype: DType
    """The underlying storage for the ComplexDType value."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __init__(out self, *, mlir_value: Self._mlir_type):
        """Construct a ComplexDType from MLIR ComplexDType.

        Args:
            mlir_value: The MLIR ComplexDType.
        """
        self._dtype = DType(mlir_value=mlir_value)

    @staticmethod
    fn _from_str(str: StringSlice) -> ComplexDType:
        """Construct a ComplexDType from a string.

        Args:
            str: The name of the ComplexDType.
        """
        if str.startswith("ComplexDType."):
            return Self._from_str(str.removeprefix("ComplexDType."))
        elif str == "int8":
            return ComplexDType.int8
        elif str == "int16":
            return ComplexDType.int16
        elif str == "int32":
            return ComplexDType.int32
        elif str == "int64":
            return ComplexDType.int64
        elif str == "int128":
            return ComplexDType.int128
        elif str == "int256":
            return ComplexDType.int256
        elif str == "int":
            return ComplexDType.int
        elif str == "uint8":
            return ComplexDType.uint8
        elif str == "uint16":
            return ComplexDType.uint16
        elif str == "uint32":
            return ComplexDType.uint32
        elif str == "uint64":
            return ComplexDType.uint64
        elif str == "uint128":
            return ComplexDType.uint128
        elif str == "uint256":
            return ComplexDType.uint256
        elif str == "uint":
            return ComplexDType.uint
        elif str == "float8_e3m4":
            return ComplexDType.float8_e3m4
        elif str == "float8_e4m3fn":
            return ComplexDType.float8_e4m3fn
        elif str == "float8_e4m3fnuz":
            return ComplexDType.float8_e4m3fnuz
        elif str == "float8_e5m2":
            return ComplexDType.float8_e5m2
        elif str == "float8_e5m2fnuz":
            return ComplexDType.float8_e5m2fnuz
        elif str == "bfloat16":
            return ComplexDType.bfloat16
        elif str == "float16":
            return ComplexDType.float16
        elif str == "float32":
            return ComplexDType.float32
        elif str == "float64":
            return ComplexDType.float64
        elif str == "bool":
            return ComplexDType.bool
        else:
            return ComplexDType.invalid

    @no_inline
    fn __str__(self) -> String:
        """Gets the name of the ComplexDType.

        Returns:
            The name of the ComplexDType.
        """

        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this ComplexDType to the provided Writer.

        Args:
            writer: The object to write to.
        """

        if self is ComplexDType.int8:
            return writer.write("int8")
        elif self is ComplexDType.int64:
            return writer.write("int64")
        elif self is ComplexDType.int128:
            return writer.write("int128")
        elif self is ComplexDType.int256:
            return writer.write("int256")
        elif self is ComplexDType.int:
            return writer.write("int")
        elif self is ComplexDType.uint8:
            return writer.write("uint8")
        elif self is ComplexDType.uint16:
            return writer.write("uint16")
        elif self is ComplexDType.uint32:
            return writer.write("uint32")
        elif self is ComplexDType.uint64:
            return writer.write("uint64")
        elif self is ComplexDType.uint128:
            return writer.write("uint128")
        elif self is ComplexDType.uint256:
            return writer.write("uint256")
        elif self is ComplexDType.uint:
            return writer.write("uint")
        elif self is ComplexDType.float8_e3m4:
            return writer.write("float8_e3m4")
        elif self is ComplexDType.float8_e4m3fn:
            return writer.write("float8_e4m3fn")
        elif self is ComplexDType.float8_e4m3fnuz:
            return writer.write("float8_e4m3fnuz")
        elif self is ComplexDType.float8_e5m2:
            return writer.write("float8_e5m2")
        elif self is ComplexDType.float8_e5m2fnuz:
            return writer.write("float8_e5m2fnuz")
        elif self is ComplexDType.bfloat16:
            return writer.write("bfloat16")
        elif self is ComplexDType.float16:
            return writer.write("float16")
        elif self is ComplexDType.float32:
            return writer.write("float32")
        elif self is ComplexDType.float64:
            return writer.write("float64")
        elif self is ComplexDType.bool:
            return writer.write("bool")
        elif self is ComplexDType.invalid:
            return writer.write("invalid")

        return writer.write("<<unknown>>")

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """Gets the representation of the ComplexDType e.g. `"ComplexDType.float32"`.

        Returns:
            The representation of the ComplexDType.
        """
        return String.write("ComplexDType.", self)

    @always_inline("nodebug")
    fn get_value(self) -> __mlir_type.`!kgen.dtype`:
        """Gets the associated internal kgen.ComplexDType value.

        Returns:
            The kgen.ComplexDType value.
        """
        return self._dtype.get_value()

    @doc_private
    @staticmethod
    @always_inline("nodebug")
    fn _from_ui8(ui8: UInt8._mlir_type) -> ComplexDType:
        var res = __mlir_op.`pop.dtype.from_ui8`(
            __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui8](ui8)
        )
        return ComplexDType(mlir_value=res)

    @doc_private
    @always_inline("nodebug")
    fn _as_ui8(self) -> UInt8._mlir_type:
        return __mlir_op.`pop.cast_from_builtin`[_type = UInt8._mlir_type](
            __mlir_op.`pop.dtype.to_ui8`(self._dtype.get_value())
        )

    @doc_private
    @always_inline("nodebug")
    fn _match(self, mask: UInt8) -> Bool:
        var res = __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
            __mlir_op.`pop.simd.and`(self._as_ui8(), mask._mlir_value),
            __mlir_attr.`#pop.simd<0> : !pop.scalar<ui8>`,
        )
        return Bool(mlir_value=res)

    @always_inline("nodebug")
    fn __is__(self, rhs: ComplexDType) -> Bool:
        """Compares one ComplexDType to another for equality.

        Args:
            rhs: The ComplexDType to compare against.

        Returns:
            True if the ComplexDTypes are the same and False otherwise.
        """
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: ComplexDType) -> Bool:
        """Compares one ComplexDType to another for equality.

        Args:
            rhs: The ComplexDType to compare against.

        Returns:
            True if the ComplexDTypes are the same and False otherwise.
        """
        return ~(self == rhs)

    @always_inline("nodebug")
    fn __eq__(self, rhs: ComplexDType) -> Bool:
        """Compares one ComplexDType to another for equality.

        Args:
            rhs: The ComplexDType to compare against.

        Returns:
            True if the ComplexDTypes are the same and False otherwise.
        """
        var res = __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred eq>`](
            self._as_ui8(), rhs._as_ui8()
        )
        return Bool(mlir_value=res)

    @always_inline("nodebug")
    fn __ne__(self, rhs: ComplexDType) -> Bool:
        """Compares one ComplexDType to another for inequality.

        Args:
            rhs: The ComplexDType to compare against.

        Returns:
            False if the ComplexDTypes are the same and True otherwise.
        """
        var res = __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
            self._as_ui8(), rhs._as_ui8()
        )
        return Bool(mlir_value=res)

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with this `ComplexDType` value.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_simd(UInt8(mlir_value=self._as_ui8()))

    @always_inline("nodebug")
    fn is_unsigned(self) -> Bool:
        """Returns True if the type parameter is unsigned and False otherwise.

        Returns:
            Returns True if the input type parameter is unsigned.
        """
        return self._is_non_index_integral() and not self._match(_mIsSigned)

    @always_inline("nodebug")
    fn is_signed(self) -> Bool:
        """Returns True if the type parameter is signed and False otherwise.

        Returns:
            Returns True if the input type parameter is signed.
        """
        if self.is_floating_point():
            return True
        return self.is_integral() and self._match(_mIsSigned)

    @always_inline("nodebug")
    fn _is_non_index_integral(self) -> Bool:
        """Returns True if the type parameter is a non-index integer value and False otherwise.

        Returns:
            Returns True if the input type parameter is a non-index integer.
        """
        return self._match(_mIsInteger)

    @always_inline("nodebug")
    fn is_integral(self) -> Bool:
        """Returns True if the type parameter is an integer and False otherwise.

        Returns:
            Returns True if the input type parameter is an integer.
        """
        return (
            self in (ComplexDType.int, ComplexDType.uint)
            or self._is_non_index_integral()
        )

    @always_inline("nodebug")
    fn is_floating_point(self) -> Bool:
        """Returns True if the type parameter is a floating-point and False
        otherwise.

        Returns:
            Returns True if the input type parameter is a floating-point.
        """
        return self._match(_mIsFloat)

    @always_inline("nodebug")
    fn is_float8(self) -> Bool:
        """Returns True if the ComplexDType is a 8bit-precision floating point type,
        e.g. float8_e5m2, float8_e5m2fnuz, float8_e4m3fn and float8_e4m3fnuz.

        Returns:
            True if the ComplexDType is a 8bit-precision float, false otherwise.
        """

        return self in (
            ComplexDType.float8_e3m4,
            ComplexDType.float8_e4m3fn,
            ComplexDType.float8_e4m3fnuz,
            ComplexDType.float8_e5m2,
            ComplexDType.float8_e5m2fnuz,
        )

    @always_inline("nodebug")
    fn is_half_float(self) -> Bool:
        """Returns True if the ComplexDType is a half-precision floating point type,
        e.g. either fp16 or bf16.

        Returns:
            True if the ComplexDType is a half-precision float, false otherwise..
        """

        return self in (ComplexDType.bfloat16, ComplexDType.float16)

    @always_inline("nodebug")
    fn is_numeric(self) -> Bool:
        """Returns True if the type parameter is numeric (i.e. you can perform
        arithmetic operations on).

        Returns:
            Returns True if the input type parameter is either integral or
              floating-point.
        """
        return self.is_integral() or self.is_floating_point()

    @always_inline
    fn size_of(self) -> Int:
        """Returns the size in bytes of the current DType.

        Returns:
            Returns the size in bytes of the current DType.
        """

        if self._is_non_index_integral():
            return Int(
                UInt8(
                    mlir_value=__mlir_op.`pop.shl`(
                        UInt8(1)._mlir_value,
                        __mlir_op.`pop.sub`(
                            __mlir_op.`pop.shr`(
                                __mlir_op.`pop.simd.and`(
                                    self._as_ui8(),
                                    _mIsNotInteger._mlir_value,
                                ),
                                UInt8(1)._mlir_value,
                            ),
                            UInt8(3)._mlir_value,
                        ),
                    )
                )
            )

        elif self is ComplexDType.bool:
            return size_of[DType.bool]()
        elif self is ComplexDType.int:
            return size_of[DType.int]()
        elif self is ComplexDType.uint:
            return size_of[DType.uint]()

        elif self is ComplexDType.float8_e3m4:
            return size_of[DType.float8_e3m4]()
        elif self is ComplexDType.float8_e4m3fn:
            return size_of[DType.float8_e4m3fn]()
        elif self is ComplexDType.float8_e4m3fnuz:
            return size_of[DType.float8_e4m3fnuz]()
        elif self is ComplexDType.float8_e5m2:
            return size_of[DType.float8_e5m2]()
        elif self is ComplexDType.float8_e5m2fnuz:
            return size_of[DType.float8_e5m2fnuz]()

        elif self is ComplexDType.bfloat16:
            return size_of[DType.bfloat16]()
        elif self is ComplexDType.float16:
            return size_of[DType.float16]()

        elif self is ComplexDType.float32:
            return size_of[DType.float32]()

        elif self is ComplexDType.float64:
            return size_of[DType.float64]()

        return size_of[DType.invalid]()

    @always_inline
    fn bitwidth(self) -> Int:
        """Returns the size in bits of the current ComplexDType.

        Returns:
            Returns the size in bits of the current ComplexDType.
        """
        return (
            2 * 8 * self.size_of()
        )  # 2 * because complex number has real and imaginary parts

    fn component_bitwidth(self) -> Int:
        """Returns the size in bits of the component type of the current ComplexDType.

        Returns:
            Returns the size in bits of the component type of the current ComplexDType.
        """
        return self.bitwidth() // 2

    # ===-------------------------------------------------------------------===#
    # __mlir_type
    # ===-------------------------------------------------------------------===#
    @always_inline("nodebug")
    fn __mlir_type(self) -> __mlir_type.`!kgen.deferred`:
        """Returns the MLIR type of the current DType as an MLIR type.

        Returns:
            The MLIR type of the current ComplexDType.
        """
        if self is ComplexDType.bool:
            return __mlir_attr.i1

        if self is ComplexDType.int:
            return __mlir_attr.index

        if self is ComplexDType.uint8:
            return __mlir_attr.ui8
        if self is ComplexDType.int8:
            return __mlir_attr.si8
        if self is ComplexDType.uint16:
            return __mlir_attr.ui16
        if self is ComplexDType.int16:
            return __mlir_attr.si16
        if self is ComplexDType.uint32:
            return __mlir_attr.ui32
        if self is ComplexDType.int32:
            return __mlir_attr.si32
        if self is ComplexDType.uint64:
            return __mlir_attr.ui64
        if self is ComplexDType.int64:
            return __mlir_attr.si64
        if self is ComplexDType.uint128:
            return __mlir_attr.ui128
        if self is ComplexDType.int128:
            return __mlir_attr.si128
        if self is ComplexDType.uint256:
            return __mlir_attr.ui256
        if self is ComplexDType.int256:
            return __mlir_attr.si256

        if self is ComplexDType.float8_e3m4:
            return __mlir_attr.f8E3M4
        if self is ComplexDType.float8_e4m3fn:
            return __mlir_attr.f8E4M3
        if self is ComplexDType.float8_e4m3fnuz:
            return __mlir_attr.f8E4M3FNUZ
        if self is ComplexDType.float8_e5m2:
            return __mlir_attr.f8E5M2
        if self is ComplexDType.float8_e5m2fnuz:
            return __mlir_attr.f8E5M2FNUZ

        if self is ComplexDType.bfloat16:
            return __mlir_attr.bf16
        if self is ComplexDType.float16:
            return __mlir_attr.f16

        if self is ComplexDType.float32:
            return __mlir_attr.f32

        if self is ComplexDType.float64:
            return __mlir_attr.f64

        return abort[__mlir_type.`!kgen.deferred`]("invalid dtype")

    fn component_dtype(self) -> DType:
        return self._dtype


fn _concise_dtype_str(cdtype: ComplexDType) -> String:
    """Returns a concise string representation of the complex data type."""
    if cdtype == ci8:
        return "ci8"
    elif cdtype == ci16:
        return "ci16"
    elif cdtype == ci32:
        return "ci32"
    elif cdtype == ci64:
        return "ci64"
    elif cdtype == ci128:
        return "ci128"
    elif cdtype == ci256:
        return "ci256"
    elif cdtype == cint:
        return "cint"
    elif cdtype == cu8:
        return "cu8"
    elif cdtype == cu16:
        return "cu16"
    elif cdtype == cu32:
        return "cu32"
    elif cdtype == cu64:
        return "cu64"
    elif cdtype == cu128:
        return "cu128"
    elif cdtype == cu256:
        return "cu256"
    elif cdtype == cuint:
        return "cuint"
    elif cdtype == cbf16:
        return "cbf16"
    elif cdtype == cf16:
        return "cf16"
    elif cdtype == cf32:
        return "cf32"
    elif cdtype == cf64:
        return "cf64"
    elif cdtype == cboolean:
        return "cboolean"
    elif cdtype == cinvalid:
        return "cinvalid"
    else:
        return "Unknown"
