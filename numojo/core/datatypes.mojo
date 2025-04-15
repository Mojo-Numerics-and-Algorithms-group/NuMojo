"""
Datatypes Module - Implements datatypes aliases, conversions
"""
# ===----------------------------------------------------------------------=== #
# Datatypes Module - Implements datatypes aliases, conversions
# Last updated: 2024-06-18
# ===----------------------------------------------------------------------=== #

# Rust-like or numpy-like data type alias
"""alias for `DType.int8`"""
alias i8 = DType.int8
"""Data type alias for DType.int8"""
alias i16 = DType.int16
"""Data type alias for DType.int16"""
alias i32 = DType.int32
"""Data type alias for DType.int32"""
alias i64 = DType.int64
"""Data type alias for DType.int64"""
alias isize = DType.index
"""Data type alias for DType.index"""
alias intp = DType.index
"""Data type alias for DType.index"""
alias u8 = DType.uint8
"""Data type alias for DType.uint8"""
alias u16 = DType.uint16
"""Data type alias for DType.uint16"""
alias u32 = DType.uint32
"""Data type alias for DType.uint32"""
alias u64 = DType.uint64
"""Data type alias for DType.uint64"""
alias f16 = DType.float16
"""Data type alias for DType.float16"""
alias f32 = DType.float32
"""Data type alias for DType.float32"""
alias f64 = DType.float64
"""Data type alias for DType.float64"""
alias boolean = DType.bool
"""Data type alias for DType.bool"""

# ===----------------------------------------------------------------------=== #


# TODO: Optimize the conditions with dict and move it to compile time
# Dict can't be created at compile time rn
struct TypeCoercion:
    """Handles type coercion using a promotion matrix approach."""

    alias ranks: List[DType] = List[DType](
        i8, u8, f16, i16, u16, f32, i32, u32, i64, u64, f64
    )
    alias int_ranks: List[DType] = List[DType](
        i8, u8, i16, u16, i32, u32, i64, u64
    )
    alias float_ranks: List[DType] = List[DType](f16, f32, f64)

    @parameter
    @staticmethod
    fn get_type_rank[dtype: DType]() -> Int:
        try:
            return Self.ranks.index(dtype)
        except ValueError:
            return 10

    @parameter
    @staticmethod
    fn get_inttype_rank[dtype: DType]() -> Int:
        try:
            return Self.int_ranks.index(dtype)
        except ValueError:
            return 7

    @parameter
    @staticmethod
    fn get_floattype_rank[dtype: DType]() -> Int:
        try:
            return Self.float_ranks.index(dtype)
        except ValueError:
            return 2

    @parameter
    @staticmethod
    fn coerce_floats[T1: DType, T2: DType]() -> DType:
        """Coerces two floating point types."""
        if T1 == f16 or T2 == f16:
            if T1 == f64 or T2 == f64:
                return f64
            return f32
        var rank1 = Self.get_floattype_rank[T1]()
        var rank2 = Self.get_floattype_rank[T2]()
        return T1 if rank1 > rank2 else T2

    @parameter
    @staticmethod
    fn coerce_signed_ints[T1: DType, T2: DType]() -> DType:
        """Coerces two signed integer types."""
        var rank1 = Self.get_type_rank[T1]()
        var rank2 = Self.get_type_rank[T2]()
        var max_rank = max(rank1, rank2)
        if max_rank <= 3:
            return i16  # int8 -> int16
        if max_rank <= 6:
            return i32  # int16 -> int32
        if max_rank <= 8:
            return i64  # int32 -> int64
        return f64  # int64 -> float64

    @parameter
    @staticmethod
    fn coerce_unsigned_ints[T1: DType, T2: DType]() -> DType:
        """Coerces two unsigned integer types."""
        if T1.sizeof() >= T2.sizeof():
            return T1
        else:
            return T2

    @parameter
    @staticmethod
    fn coerce_mixed_ints[T1: DType, T2: DType]() -> DType:
        """Coerces a signed and unsigned integer type."""
        alias signed = T1 if T1.is_signed() else T2
        alias unsigned = T2 if T1.is_signed() else T1

        # Handle signed/unsigned pairs
        if signed == i8 and unsigned == u8:
            return i16
        if signed == i16 and unsigned == u16:
            return i32
        if signed == i32 and unsigned == u32:
            return i64
        if signed == i64 and unsigned == u64:
            return f64

        # If unsigned type is larger, use next larger signed type
        var signed_rank = Self.get_type_rank[signed]()
        var unsigned_rank = Self.get_type_rank[unsigned]()

        if unsigned_rank > signed_rank:
            if unsigned == u16:
                return i32
            if unsigned == u32:
                return i64
            if unsigned == u64:
                return f64

        return signed

    @parameter
    @staticmethod
    fn coerce_mixed[int_type: DType, float_type: DType]() -> DType:
        """Coerces a mixed integer and floating point type."""
        # Special case: float16 always promotes to at least float32
        if float_type == f16 and (int_type == i16 or int_type == u16):
            return f32
        if float_type == f16 and (int_type == i32 or int_type == u32):
            return f64
        if float_type == f16 and (int_type == i64 or int_type == u64):
            return f64
        # Special cases for int32/uint32 and larger with float32
        if float_type == f32:
            if int_type in (i32, u32, i64, u64):
                return f64
        return float_type

    @parameter
    @staticmethod
    fn result[T1: DType, T2: DType]() -> DType:
        """Returns the coerced output type for two input types."""
        if T1 == T2:
            return T1
        elif T1.is_floating_point() and T2.is_floating_point():
            return TypeCoercion.coerce_floats[T1, T2]()
        elif T1.is_integral() and T2.is_integral():
            if T1.is_signed() and T2.is_signed():
                return TypeCoercion.coerce_signed_ints[T1, T2]()
            elif T1.is_unsigned() and T2.is_unsigned():
                return TypeCoercion.coerce_unsigned_ints[T1, T2]()
            else:
                return TypeCoercion.coerce_mixed_ints[T1, T2]()
        elif T1.is_integral() and T2.is_floating_point():
            return TypeCoercion.coerce_mixed[T1, T2]()
        elif T1.is_floating_point() and T2.is_integral():
            return TypeCoercion.coerce_mixed[T2, T1]()
        return T1


fn _concise_dtype_str(dtype: DType) -> String:
    """Returns a concise string representation of the data type."""
    if dtype == i8:
        return "i8"
    elif dtype == i16:
        return "i16"
    elif dtype == i32:
        return "i32"
    elif dtype == i64:
        return "i64"
    elif dtype == isize:
        return "index"
    elif dtype == u8:
        return "u8"
    elif dtype == u16:
        return "u16"
    elif dtype == u32:
        return "u32"
    elif dtype == u64:
        return "u64"
    elif dtype == f16:
        return "f16"
    elif dtype == f32:
        return "f32"
    elif dtype == f64:
        return "f64"
    elif dtype == boolean:
        return "boolean"
    elif dtype == isize:
        return "isize"
    else:
        return "Unknown"
