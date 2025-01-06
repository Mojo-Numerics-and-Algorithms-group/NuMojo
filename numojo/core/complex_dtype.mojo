# Code for CDType is adapted from the Mojo Standard Library
# (https://github.com/modularml/mojo)
# licensed under the Apache License, Version 2.0.
# Modifications were made for the purposes of this project to
# support a Complex SIMD type.

from collections import KeyElement
from hashlib._hasher import _HashableWithHasher, _Hasher
from sys import bitwidthof, os_is_windows, sizeof
from math import sqrt

alias _mIsSigned = UInt8(1)
alias _mIsInteger = UInt8(1 << 7)
alias _mIsNotInteger = UInt8(~(1 << 7))
alias _mIsFloat = UInt8(1 << 6)


@value
@register_passable("trivial")
struct CDType(
    Stringable,
    Writable,
    Representable,
    KeyElement,
    CollectionElementNew,
    _HashableWithHasher,
):
    """Represents CDType and provides methods for working with it."""

    alias type = __mlir_type.`!kgen.dtype`
    var re_value: Self.type
    var im_value: Self.type

    alias invalid = CDType(
        __mlir_attr.`#kgen.dtype.constant<invalid> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<invalid> : !kgen.dtype`,
    )
    """Represents an invalid or unknown data type."""
    alias bool = CDType(
        __mlir_attr.`#kgen.dtype.constant<bool> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<bool> : !kgen.dtype`,
    )
    """Represents a boolean data type."""
    alias int8 = CDType(
        __mlir_attr.`#kgen.dtype.constant<si8> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<si8> : !kgen.dtype`,
    )
    """Represents a signed integer type whose bitwidth is 8."""
    alias uint8 = CDType(
        __mlir_attr.`#kgen.dtype.constant<ui8> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<ui8> : !kgen.dtype`,
    )
    """Represents an unsigned integer type whose bitwidth is 8."""
    alias int16 = CDType(
        __mlir_attr.`#kgen.dtype.constant<si16> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<si16> : !kgen.dtype`,
    )
    """Represents a signed integer type whose bitwidth is 16."""
    alias uint16 = CDType(
        __mlir_attr.`#kgen.dtype.constant<ui16> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<ui16> : !kgen.dtype`,
    )
    """Represents an unsigned integer type whose bitwidth is 16."""
    alias int32 = CDType(
        __mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`,
    )
    """Represents a signed integer type whose bitwidth is 32."""
    alias uint32 = CDType(
        __mlir_attr.`#kgen.dtype.constant<ui32> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<ui32> : !kgen.dtype`,
    )
    """Represents an unsigned integer type whose bitwidth is 32."""
    alias int64 = CDType(
        __mlir_attr.`#kgen.dtype.constant<si64> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<si64> : !kgen.dtype`,
    )
    """Represents a signed integer type whose bitwidth is 64."""
    alias uint64 = CDType(
        __mlir_attr.`#kgen.dtype.constant<ui64> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<ui64> : !kgen.dtype`,
    )
    """Represents an unsigned integer type whose bitwidth is 64."""
    alias float8e5m2 = CDType(
        __mlir_attr.`#kgen.dtype.constant<f8e5m2> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<f8e5m2> : !kgen.dtype`,
    )
    """Represents a FP8E5M2 floating point format from the [OFP8
    standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1).

    The 8 bits are encoded as `seeeeemm`:
    - (s)ign: 1 bit
    - (e)xponent: 5 bits
    - (m)antissa: 2 bits
    - exponent bias: 15
    - nan: {0,1}11111{01,10,11}
    - inf: 01111100
    - -inf: 11111100
    - -0: 10000000
    """
    alias float8e5m2fnuz = CDType(
        __mlir_attr.`#kgen.dtype.constant<f8e5m2fnuz> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<f8e5m2fnuz> : !kgen.dtype`,
    )
    """Represents a FP8E5M2FNUZ floating point format.

    The 8 bits are encoded as `seeeeemm`:
    - (s)ign: 1 bit
    - (e)xponent: 5 bits
    - (m)antissa: 2 bits
    - exponent bias: 16
    - nan: 10000000
    - fn: finite (no inf or -inf encodings)
    - uz: unsigned zero (no -0 encoding)
    """
    alias float8e4m3 = CDType(
        __mlir_attr.`#kgen.dtype.constant<f8e4m3> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<f8e4m3> : !kgen.dtype`,
    )
    """Represents a FP8E4M3 floating point format from the [OFP8
    standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1).

    This type is named `float8_e4m3fn` (the "fn" stands for "finite") in some
    frameworks, as it does not encode -inf or inf.

    The 8 bits are encoded as `seeeemmm`:
    - (s)ign: 1 bit
    - (e)xponent: 4 bits
    - (m)antissa: 3 bits
    - exponent bias: 7
    - nan: 01111111, 11111111
    - -0: 10000000
    - fn: finite (no inf or -inf encodings)
    """
    alias float8e4m3fnuz = CDType(
        __mlir_attr.`#kgen.dtype.constant<f8e4m3fnuz> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<f8e4m3fnuz> : !kgen.dtype`,
    )
    """Represents a FP8E4M3FNUZ floating point format.

    The 8 bits are encoded as `seeeemmm`:
    - (s)ign: 1 bit
    - (e)xponent: 4 bits
    - (m)antissa: 3 bits
    - exponent bias: 8
    - nan: 10000000
    - fn: finite (no inf or -inf encodings)
    - uz: unsigned zero (no -0 encoding)
    """
    alias bfloat16 = CDType(
        __mlir_attr.`#kgen.dtype.constant<bf16> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<bf16> : !kgen.dtype`,
    )
    """Represents a brain floating point value whose bitwidth is 16."""
    alias float16 = CDType(
        __mlir_attr.`#kgen.dtype.constant<f16> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<f16> : !kgen.dtype`,
    )
    """Represents an IEEE754-2008 `binary16` floating point value."""
    alias float32 = CDType(
        __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`,
    )
    """Represents an IEEE754-2008 `binary32` floating point value."""
    alias tensor_float32 = CDType(
        __mlir_attr.`#kgen.dtype.constant<tf32> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<tf32> : !kgen.dtype`,
    )
    """Represents a special floating point format supported by NVIDIA Tensor
    Cores, with the same range as float32 and reduced precision (>=10 bits).
    Note that this type is only available on NVIDIA GPUs.
    """
    alias float64 = CDType(
        __mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`,
    )
    """Represents an IEEE754-2008 `binary64` floating point value."""
    alias index = CDType(
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    )
    """Represents an integral type whose bitwidth is the maximum integral value
    on the system."""

    @always_inline
    fn __init__(out self, *, other: Self):
        """Copy this CDType.

        Arguments:
            other: The CDType to copy.
        """
        self = other

    @always_inline
    # @implicit
    fn __init__(out self, re_value: Self.type, im_value: Self.type):
        """Construct a CDType from MLIR dtype.

        Arguments:
            value: The MLIR dtype.
        """
        self.re_value = re_value
        self.im_value = im_value

    @staticmethod
    @parameter
    fn _from_dtype[dtype: DType]() -> CDType:
        """Construct a CDType from a DType.

        Arguments:
            dtype: The DType to convert.
        """

        @parameter
        if dtype == DType.bool:
            return CDType.bool
        if dtype == DType.int8:
            return CDType.int8
        if dtype == DType.uint8:
            return CDType.uint8
        if dtype == DType.int16:
            return CDType.int16
        if dtype == DType.uint16:
            return CDType.uint16
        if dtype == DType.int32:
            return CDType.int32
        if dtype == DType.uint32:
            return CDType.uint32
        if dtype == DType.int64:
            return CDType.int64
        if dtype == DType.uint64:
            return CDType.uint64
        if dtype == DType.index:
            return CDType.index
        if dtype == DType.float8e5m2:
            return CDType.float8e5m2
        if dtype == DType.float8e5m2fnuz:
            return CDType.float8e5m2fnuz
        if dtype == DType.float8e4m3:
            return CDType.float8e4m3
        if dtype == DType.float8e4m3fnuz:
            return CDType.float8e4m3fnuz
        if dtype == DType.bfloat16:
            return CDType.bfloat16
        if dtype == DType.float16:
            return CDType.float16
        if dtype == DType.float32:
            return CDType.float32
        if dtype == DType.tensor_float32:
            return CDType.tensor_float32
        if dtype == DType.float64:
            return CDType.float64
        if dtype == DType.invalid:
            return CDType.invalid
        else:
            return CDType.invalid

    @staticmethod
    fn _from_dtype(dtype: DType) -> CDType:
        """Construct a CDType from a DType.

        Arguments:
            dtype: The DType to convert.
        """

        if dtype == DType.bool:
            return CDType.bool
        if dtype == DType.int8:
            return CDType.int8
        if dtype == DType.uint8:
            return CDType.uint8
        if dtype == DType.int16:
            return CDType.int16
        if dtype == DType.uint16:
            return CDType.uint16
        if dtype == DType.int32:
            return CDType.int32
        if dtype == DType.uint32:
            return CDType.uint32
        if dtype == DType.int64:
            return CDType.int64
        if dtype == DType.uint64:
            return CDType.uint64
        if dtype == DType.index:
            return CDType.index
        if dtype == DType.float8e5m2:
            return CDType.float8e5m2
        if dtype == DType.float8e5m2fnuz:
            return CDType.float8e5m2fnuz
        if dtype == DType.float8e4m3:
            return CDType.float8e4m3
        if dtype == DType.float8e4m3fnuz:
            return CDType.float8e4m3fnuz
        if dtype == DType.bfloat16:
            return CDType.bfloat16
        if dtype == DType.float16:
            return CDType.float16
        if dtype == DType.float32:
            return CDType.float32
        if dtype == DType.tensor_float32:
            return CDType.tensor_float32
        if dtype == DType.float64:
            return CDType.float64
        if dtype == DType.invalid:
            return CDType.invalid
        else:
            return CDType.invalid

    @staticmethod
    fn _from_str(str: String) -> CDType:
        """Construct a CDType from a string.

        Arguments:
            str: The name of the CDType.
        """
        if str.startswith(String("CDType.")):
            return Self._from_str(str.removeprefix("CDType."))
        elif str == String("bool"):
            return CDType.bool
        elif str == String("int8"):
            return CDType.int8
        elif str == String("uint8"):
            return CDType.uint8
        elif str == String("int16"):
            return CDType.int16
        elif str == String("uint16"):
            return CDType.uint16
        elif str == String("int32"):
            return CDType.int32
        elif str == String("uint32"):
            return CDType.uint32
        elif str == String("int64"):
            return CDType.int64
        elif str == String("uint64"):
            return CDType.uint64
        elif str == String("index"):
            return CDType.index
        elif str == String("float8e5m2"):
            return CDType.float8e5m2
        elif str == String("float8e5m2fnuz"):
            return CDType.float8e5m2fnuz
        elif str == String("float8e4m3"):
            return CDType.float8e4m3
        elif str == String("float8e4m3fnuz"):
            return CDType.float8e4m3fnuz
        elif str == String("bfloat16"):
            return CDType.bfloat16
        elif str == String("float16"):
            return CDType.float16
        elif str == String("float32"):
            return CDType.float32
        elif str == String("float64"):
            return CDType.float64
        elif str == String("tensor_float32"):
            return CDType.tensor_float32
        elif str == String("invalid"):
            return CDType.invalid
        else:
            return CDType.invalid

    @staticmethod
    @parameter
    fn to_dtype[other: Self]() -> DType:
        """Find the equivalent DType.

        Returns:
            True if the DTypes are the same and False otherwise.
        """

        @parameter
        if other == CDType.bool:
            return DType.bool
        if other == CDType.int8:
            return DType.int8
        if other == CDType.uint8:
            return DType.uint8
        if other == CDType.int16:
            return DType.int16
        if other == CDType.uint16:
            return DType.uint16
        if other == CDType.int32:
            return DType.int32
        if other == CDType.uint32:
            return DType.uint32
        if other == CDType.int64:
            return DType.int64
        if other == CDType.uint64:
            return DType.uint64
        if other == CDType.index:
            return DType.index
        if other == CDType.float8e5m2:
            return DType.float8e5m2
        if other == CDType.float8e5m2fnuz:
            return DType.float8e5m2fnuz
        if other == CDType.float8e4m3:
            return DType.float8e4m3
        if other == CDType.float8e4m3fnuz:
            return DType.float8e4m3fnuz
        if other == CDType.bfloat16:
            return DType.bfloat16
        if other == CDType.float16:
            return DType.float16
        if other == CDType.float32:
            return DType.float32
        if other == CDType.tensor_float32:
            return DType.tensor_float32
        if other == CDType.float64:
            return DType.float64
        if other == CDType.invalid:
            return DType.invalid
        else:
            return DType.invalid

    @no_inline
    fn __str__(self) -> String:
        """Gets the name of the CDType.

        Returns:
            The name of the dtype.
        """

        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this dtype to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Arguments:
            writer: The object to write to.
        """

        if self == CDType.bool:
            return writer.write("cbool")
        if self == CDType.int8:
            return writer.write("cint8")
        if self == CDType.uint8:
            return writer.write("cuint8")
        if self == CDType.int16:
            return writer.write("cint16")
        if self == CDType.uint16:
            return writer.write("cuint16")
        if self == CDType.int32:
            return writer.write("cint32")
        if self == CDType.uint32:
            return writer.write("cuint32")
        if self == CDType.int64:
            return writer.write("cint64")
        if self == CDType.uint64:
            return writer.write("cuint64")
        if self == CDType.index:
            return writer.write("cindex")
        if self == CDType.float8e5m2:
            return writer.write("cfloat8e5m2")
        if self == CDType.float8e5m2fnuz:
            return writer.write("cfloat8e5m2fnuz")
        if self == CDType.float8e4m3:
            return writer.write("cfloat8e4m3")
        if self == CDType.float8e4m3fnuz:
            return writer.write("cfloat8e4m3fnuz")
        if self == CDType.bfloat16:
            return writer.write("cbfloat16")
        if self == CDType.float16:
            return writer.write("cfloat16")
        if self == CDType.float32:
            return writer.write("cfloat32")
        if self == CDType.tensor_float32:
            return writer.write("ctensor_float32")
        if self == CDType.float64:
            return writer.write("cfloat64")
        if self == CDType.invalid:
            return writer.write("cinvalid")
        return writer.write("<<unknown>>")

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """Gets the representation of the CDType e.g. `"CDType.float32"`.

        Returns:
            The representation of the dtype.
        """
        return String.write("CDType.", self)

    @always_inline("nodebug")
    fn get_value(self) -> __mlir_type.`!kgen.dtype`:
        """Gets the associated internal kgen.dtype value.

        Returns:
            The kgen.dtype value.
        """
        return self.re_value

    @staticmethod
    fn _from_ui8(ui8: __mlir_type.ui8) -> CDType:
        return CDType._from_dtype(__mlir_op.`pop.dtype.from_ui8`(ui8))

    @staticmethod
    fn _from_ui8(ui8: __mlir_type.`!pop.scalar<ui8>`) -> CDType:
        return CDType._from_ui8(
            __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui8](ui8)
        )

    @always_inline("nodebug")
    fn _as_i8(
        self,
    ) -> __mlir_type.`!pop.scalar<ui8>`:
        var val = __mlir_op.`pop.dtype.to_ui8`(self.re_value)
        return __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<ui8>`
        ](val)

    @always_inline("nodebug")
    fn __is__(self, rhs: CDType) -> Bool:
        """Compares one CDType to another for equality.

        Arguments:
            rhs: The CDType to compare against.

        Returns:
            True if the DTypes are the same and False otherwise.
        """
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: CDType) -> Bool:
        """Compares one CDType to another for inequality.

        Arguments:
            rhs: The CDType to compare against.

        Returns:
            True if the DTypes are the same and False otherwise.
        """
        return self != rhs

    @always_inline("nodebug")
    fn __eq__(self, rhs: CDType) -> Bool:
        """Compares one CDType to another for equality.

        Arguments:
            rhs: The CDType to compare against.

        Returns:
            True if the DTypes are the same and False otherwise.
        """
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred eq>`](
            self._as_i8(), rhs._as_i8()
        )

    @always_inline("nodebug")
    fn __ne__(self, rhs: CDType) -> Bool:
        """Compares one CDType to another for inequality.

        Arguments:
            rhs: The CDType to compare against.

        Returns:
            False if the DTypes are the same and True otherwise.
        """
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
            self._as_i8(), rhs._as_i8()
        )

    fn __hash__(self) -> UInt:
        """Return a 64-bit hash for this `CDType` value.

        Returns:
            A 64-bit integer hash of this `CDType` value.
        """
        return hash(UInt8(self._as_i8()))

    fn __hash__[H: _Hasher](self, mut hasher: H):
        """Updates hasher with this `CDType` value.

        Parameters:
            H: The hasher type.

        Arguments:
            hasher: The hasher instance.
        """
        hasher._update_with_simd(UInt8(self._as_i8()))

    @always_inline("nodebug")
    fn is_unsigned(self) -> Bool:
        """Returns True if the type parameter is unsigned and False otherwise.

        Returns:
            Returns True if the input type parameter is unsigned.
        """
        if not self.is_integral():
            return False
        return Bool(
            __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred eq>`](
                __mlir_op.`pop.simd.and`(self._as_i8(), _mIsSigned.value),
                UInt8(0).value,
            )
        )

    @always_inline("nodebug")
    fn is_signed(self) -> Bool:
        """Returns True if the type parameter is signed and False otherwise.

        Returns:
            Returns True if the input type parameter is signed.
        """
        if self is CDType.index or self.is_floating_point():
            return True
        if not self.is_integral():
            return False
        return Bool(
            __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
                __mlir_op.`pop.simd.and`(self._as_i8(), _mIsSigned.value),
                UInt8(0).value,
            )
        )

    @always_inline("nodebug")
    fn _is_non_index_integral(self) -> Bool:
        """Returns True if the type parameter is a non-index integer value and False otherwise.

        Returns:
            Returns True if the input type parameter is a non-index integer.
        """
        return Bool(
            __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
                __mlir_op.`pop.simd.and`(self._as_i8(), _mIsInteger.value),
                UInt8(0).value,
            )
        )

    @always_inline("nodebug")
    fn is_integral(self) -> Bool:
        """Returns True if the type parameter is an integer and False otherwise.

        Returns:
            Returns True if the input type parameter is an integer.
        """
        if self is CDType.index:
            return True
        return self._is_non_index_integral()

    @always_inline("nodebug")
    fn is_floating_point(self) -> Bool:
        """Returns True if the type parameter is a floating-point and False
        otherwise.

        Returns:
            Returns True if the input type parameter is a floating-point.
        """
        if self.is_integral():
            return False
        return Bool(
            __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
                __mlir_op.`pop.simd.and`(self._as_i8(), _mIsFloat.value),
                UInt8(0).value,
            )
        )

    @always_inline("nodebug")
    fn is_float8(self) -> Bool:
        """Returns True if the type is a 8bit-precision floating point type,
        e.g. float8e5m2, float8e5m2fnuz, float8e4m3 and float8e4m3fnuz.

        Returns:
            True if the type is a 8bit-precision float, false otherwise.
        """

        return self in (
            CDType.float8e5m2,
            CDType.float8e4m3,
            CDType.float8e5m2fnuz,
            CDType.float8e4m3fnuz,
        )

    @always_inline("nodebug")
    fn is_half_float(self) -> Bool:
        """Returns True if the type is a half-precision floating point type,
        e.g. either fp16 or bf16.

        Returns:
            True if the type is a half-precision float, false otherwise..
        """

        return self in (CDType.bfloat16, CDType.float16)

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
    fn sizeof(self) -> Int:
        """Returns the size in bytes of the current CDType.

        Returns:
            Returns the size in bytes of the current CDType.
        """

        if self._is_non_index_integral():
            return int(
                UInt8(
                    __mlir_op.`pop.shl`(
                        UInt8(1).value,
                        __mlir_op.`pop.sub`(
                            __mlir_op.`pop.shr`(
                                __mlir_op.`pop.simd.and`(
                                    self._as_i8(), _mIsNotInteger.value
                                ),
                                UInt8(1).value,
                            ),
                            UInt8(3).value,
                        ),
                    )
                )
            )

        if self == CDType.bool:
            return 2 * sizeof[DType.bool]()
        if self == CDType.index:
            return 2 * sizeof[DType.index]()
        if self == CDType.float8e5m2:
            return 2 * sizeof[DType.float8e5m2]()
        if self == CDType.float8e5m2fnuz:
            return 2 * sizeof[DType.float8e5m2fnuz]()
        if self == CDType.float8e4m3:
            return 2 * sizeof[DType.float8e4m3]()
        if self == CDType.float8e4m3fnuz:
            return 2 * sizeof[DType.float8e4m3fnuz]()
        if self == CDType.bfloat16:
            return 2 * sizeof[DType.bfloat16]()
        if self == CDType.float16:
            return 2 * sizeof[DType.float16]()
        if self == CDType.float32:
            return 2 * sizeof[DType.float32]()
        if self == CDType.tensor_float32:
            return 2 * sizeof[DType.tensor_float32]()
        if self == CDType.float64:
            return 2 * sizeof[DType.float64]()
        return 2 * sizeof[DType.invalid]()

    @always_inline
    fn bitwidth(self) -> Int:
        """Returns the size in bits of the current CDType.

        Returns:
            Returns the size in bits of the current CDType.
        """
        return 2 * 8 * self.sizeof()

    # ===-------------------------------------------------------------------===#
    # dispatch_integral
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn dispatch_integral[
        func: fn[type: CDType] () capturing [_] -> None
    ](self) raises:
        """Dispatches an integral function corresponding to the current CDType.

        Constraints:
            CDType must be integral.

        Parameters:
            func: A parametrized on dtype function to dispatch.
        """
        if self is CDType.uint8:
            func[CDType.uint8]()
        elif self is CDType.int8:
            func[CDType.int8]()
        elif self is CDType.uint16:
            func[CDType.uint16]()
        elif self is CDType.int16:
            func[CDType.int16]()
        elif self is CDType.uint32:
            func[CDType.uint32]()
        elif self is CDType.int32:
            func[CDType.int32]()
        elif self is CDType.uint64:
            func[CDType.uint64]()
        elif self is CDType.int64:
            func[CDType.int64]()
        elif self is CDType.index:
            func[CDType.index]()
        else:
            raise Error("only integral types are supported")

    # ===-------------------------------------------------------------------===#
    # dispatch_floating
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn dispatch_floating[
        func: fn[type: CDType] () capturing [_] -> None
    ](self) raises:
        """Dispatches a floating-point function corresponding to the current CDType.

        Constraints:
            CDType must be floating-point or integral.

        Parameters:
            func: A parametrized on dtype function to dispatch.
        """
        if self is CDType.float16:
            func[CDType.float16]()
        # TODO(#15473): Enable after extending LLVM support
        # elif self is CDType.bfloat16:
        #     func[CDType.bfloat16]()
        elif self is CDType.float32:
            func[CDType.float32]()
        elif self is CDType.float64:
            func[CDType.float64]()
        else:
            raise Error("only floating point types are supported")

    @always_inline
    fn _dispatch_bitwidth[
        func: fn[type: CDType] () capturing [_] -> None,
    ](self) raises:
        """Dispatches a function corresponding to the current CDType's bitwidth.
        This should only be used if func only depends on the bitwidth of the dtype,
        and not other properties of the dtype.

        Parameters:
            func: A parametrized on dtype function to dispatch.
        """
        var bitwidth = self.bitwidth()
        if bitwidth == 8:
            func[CDType.uint8]()
        elif bitwidth == 16:
            func[CDType.uint16]()
        elif bitwidth == 32:
            func[CDType.uint32]()
        elif bitwidth == 64:
            func[CDType.uint64]()
        else:
            raise Error(
                "bitwidth_dispatch only supports types with bitwidth [8, 16,"
                " 32, 64]"
            )
        return

    @always_inline
    fn _dispatch_custom[
        func: fn[type: CDType] () capturing [_] -> None, *dtypes: CDType
    ](self) raises:
        """Dispatches a function corresponding to current CDType if it matches
        any type in the dtypes parameter.

        Parameters:
            func: A parametrized on dtype function to dispatch.
            dtypes: A list of DTypes on which to do dispatch.
        """
        alias dtype_var = VariadicList[CDType](dtypes)

        @parameter
        for idx in range(len(dtype_var)):
            alias dtype = dtype_var[idx]
            if self == dtype:
                return func[dtype]()

        raise Error(
            "dispatch_custom: dynamic_type does not match any dtype parameters"
        )

    # ===-------------------------------------------------------------------===#
    # dispatch_arithmetic
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn dispatch_arithmetic[
        func: fn[type: CDType] () capturing [_] -> None
    ](self) raises:
        """Dispatches a function corresponding to the current CDType.

        Parameters:
            func: A parametrized on dtype function to dispatch.
        """
        if self.is_floating_point():
            self.dispatch_floating[func]()
        elif self.is_integral():
            self.dispatch_integral[func]()
        else:
            raise Error("only arithmetic types are supported")


# ===-------------------------------------------------------------------===#
# integral_type_of
# ===-------------------------------------------------------------------===#


@always_inline("nodebug")
fn _integral_type_of[type: CDType]() -> CDType:
    """Gets the integral type which has the same bitwidth as the input type."""

    @parameter
    if type.is_integral():
        return type

    @parameter
    if type.is_float8():
        return CDType.int8

    @parameter
    if type.is_half_float():
        return CDType.int16

    @parameter
    if type is CDType.float32 or type is CDType.tensor_float32:
        return CDType.int32

    @parameter
    if type is CDType.float64:
        return CDType.int64

    return type.invalid


@always_inline("nodebug")
fn _uint_type_of[type: CDType]() -> CDType:
    """Gets the unsigned integral type which has the same bitwidth as the input
    type."""

    @parameter
    if type.is_integral() and type.is_unsigned():
        return type

    @parameter
    if type.is_float8() or type is CDType.int8:
        return CDType.uint8

    @parameter
    if type.is_half_float() or type is CDType.int16:
        return CDType.uint16

    @parameter
    if (
        type is CDType.float32
        or type is CDType.tensor_float32
        or type is CDType.int32
    ):
        return CDType.uint32

    @parameter
    if type is CDType.float64 or type is CDType.int64:
        return CDType.uint64

    return type.invalid


# ===-------------------------------------------------------------------===#
# _unsigned_integral_type_of
# ===-------------------------------------------------------------------===#


@always_inline("nodebug")
fn _unsigned_integral_type_of[type: CDType]() -> CDType:
    """Gets the unsigned integral type which has the same bitwidth as
    the input type."""

    @parameter
    if type.is_integral():
        return _uint_type_of_width[bitwidthof[CDType.to_dtype[type]()]()]()

    @parameter
    if type.is_float8():
        return CDType.uint8

    @parameter
    if type.is_half_float():
        return CDType.uint16

    @parameter
    if type is CDType.float32 or type is CDType.tensor_float32:
        return CDType.uint32

    @parameter
    if type is CDType.float64:
        return CDType.uint64

    return type.invalid


# ===-------------------------------------------------------------------===#
# _scientific_notation_digits
# ===-------------------------------------------------------------------===#


fn _scientific_notation_digits[type: CDType]() -> StringLiteral:
    """Get the number of digits as a StringLiteral for the scientific notation
    representation of a float.
    """
    constrained[type.is_floating_point(), "expected floating point type"]()

    @parameter
    if type.is_float8():
        return "2"
    elif type.is_half_float():
        return "4"
    elif type is CDType.float32 or type is CDType.tensor_float32:
        return "8"
    else:
        constrained[type is CDType.float64, "unknown floating point type"]()
        return "16"


# ===-------------------------------------------------------------------===#
# _int_type_of_width
# ===-------------------------------------------------------------------===#


@parameter
@always_inline
fn _int_type_of_width[width: Int]() -> CDType:
    constrained[
        width == 8 or width == 16 or width == 32 or width == 64,
        "width must be either 8, 16, 32, or 64",
    ]()

    @parameter
    if width == 8:
        return CDType.int8
    elif width == 16:
        return CDType.int16
    elif width == 32:
        return CDType.int32
    else:
        return CDType.int64


# ===-------------------------------------------------------------------===#
# _uint_type_of_width
# ===-------------------------------------------------------------------===#


@parameter
@always_inline
fn _uint_type_of_width[width: Int]() -> CDType:
    constrained[
        width == 8 or width == 16 or width == 32 or width == 64,
        "width must be either 8, 16, 32, or 64",
    ]()

    @parameter
    if width == 8:
        return CDType.uint8
    elif width == 16:
        return CDType.uint16
    elif width == 32:
        return CDType.uint32
    else:
        return CDType.uint64


# ===-------------------------------------------------------------------===#
# printf format
# ===-------------------------------------------------------------------===#


@always_inline
fn _index_printf_format() -> StringLiteral:
    @parameter
    if bitwidthof[Int]() == 32:
        return "%d"
    elif os_is_windows():
        return "%lld"
    else:
        return "%ld"


@always_inline
fn _get_dtype_printf_format[type: CDType]() -> StringLiteral:
    @parameter
    if type is CDType.bool:
        return _index_printf_format()
    elif type is CDType.uint8:
        return "%hhu"
    elif type is CDType.int8:
        return "%hhi"
    elif type is CDType.uint16:
        return "%hu"
    elif type is CDType.int16:
        return "%hi"
    elif type is CDType.uint32:
        return "%u"
    elif type is CDType.int32:
        return "%i"
    elif type is CDType.int64:

        @parameter
        if os_is_windows():
            return "%lld"
        else:
            return "%ld"
    elif type is CDType.uint64:

        @parameter
        if os_is_windows():
            return "%llu"
        else:
            return "%lu"
    elif type is CDType.index:
        return _index_printf_format()

    elif type.is_floating_point():
        return "%.17g"

    else:
        constrained[False, "invalid dtype"]()

    return ""
