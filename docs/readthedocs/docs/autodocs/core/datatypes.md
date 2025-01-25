



# datatypes

##  Module Summary
  
Datatypes Module - Implements datatypes aliases, conversions
## Aliases
  
`i8`: Data type alias for DType.int8  
`i16`: Data type alias for DType.int16  
`i32`: Data type alias for DType.int32  
`i64`: Data type alias for DType.int64  
`isize`: Data type alias for DType.index  
`intp`: Data type alias for DType.index  
`u8`: Data type alias for DType.uint8  
`u16`: Data type alias for DType.uint16  
`u32`: Data type alias for DType.uint32  
`u64`: Data type alias for DType.uint64  
`f16`: Data type alias for DType.float16  
`f32`: Data type alias for DType.float32  
`f64`: Data type alias for DType.float64  
`boolean`: Data type alias for DType.bool  
`ci8`: Data type alias for ComplexSIMD[DType.int16, 1]   
`ci16`: Data type alias for ComplexSIMD[DType.int32, 1]   
`ci32`: Data type alias for ComplexSIMD[DType.int64, 1]   
`ci64`: Data type alias for ComplexSIMD[DType.uint8, 1]   
`cu8`: Data type alias for ComplexSIMD[DType.uint16, 1]   
`cu16`: Data type alias for ComplexSIMD[DType.uint32, 1]   
`cu32`: Data type alias for ComplexSIMD[DType.uint64, 1]   
`cu64`: Data type alias for ComplexSIMD[DType.float16, 1]   
`cf16`: Data type alias for ComplexSIMD[DType.float32, 1]   
`cf32`: Data type alias for ComplexSIMD[DType.float64, 1]   
`cf64`: 
## TypeCoercion

### TypeCoercion Summary
  
  
Handles type coercion using a promotion matrix approach.  

### Parent Traits
  

- AnyType
- UnknownDestructibility

### Aliases
  
`ranks`:   
`int_ranks`:   
`float_ranks`:   

### Functions

#### get_type_rank


```Mojo
get_type_rank[dtype: DType]() -> Int
```  
Summary  
  
  
  
Parameters:  

- dtype

#### get_inttype_rank


```Mojo
get_inttype_rank[dtype: DType]() -> Int
```  
Summary  
  
  
  
Parameters:  

- dtype

#### get_floattype_rank


```Mojo
get_floattype_rank[dtype: DType]() -> Int
```  
Summary  
  
  
  
Parameters:  

- dtype

#### coerce_floats


```Mojo
coerce_floats[T1: DType, T2: DType]() -> DType
```  
Summary  
  
Coerces two floating point types.  
  
Parameters:  

- T1
- T2

#### coerce_signed_ints


```Mojo
coerce_signed_ints[T1: DType, T2: DType]() -> DType
```  
Summary  
  
Coerces two signed integer types.  
  
Parameters:  

- T1
- T2

#### coerce_unsigned_ints


```Mojo
coerce_unsigned_ints[T1: DType, T2: DType]() -> DType
```  
Summary  
  
Coerces two unsigned integer types.  
  
Parameters:  

- T1
- T2

#### coerce_mixed_ints


```Mojo
coerce_mixed_ints[T1: DType, T2: DType]() -> DType
```  
Summary  
  
Coerces a signed and unsigned integer type.  
  
Parameters:  

- T1
- T2

#### coerce_mixed


```Mojo
coerce_mixed[int_type: DType, float_type: DType]() -> DType
```  
Summary  
  
Coerces a mixed integer and floating point type.  
  
Parameters:  

- int_type
- float_type

#### result


```Mojo
result[T1: DType, T2: DType]() -> DType
```  
Summary  
  
Returns the coerced output type for two input types.  
  
Parameters:  

- T1
- T2
