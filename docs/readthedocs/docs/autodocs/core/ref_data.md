



# ref_data

##  Module Summary
  

## RefData

### RefData Summary
  
  
  

### Parent Traits
  

- AnyType
- Bufferable
- UnknownDestructibility

### Fields
  
  
* ptr `UnsafePointer[SIMD[float16, 1]]`  

### Functions

#### __init__


```Mojo
__init__(out self, size: Int)
```  
Summary  
  
Allocate given space on memory. The bytes allocated is `size` * `byte size of dtype`.  
  
Args:  

- self
- size


Notes:
Although it has the lifetime of another array, it owns the data.
`ndarray.flags['OWN_DATA']` should be set as True.
The memory should be freed by `__del__`.

```Mojo
__init__(out self, ptr: UnsafePointer[SIMD[float16, 1]])
```  
Summary  
  
Reads the underlying data of another array.  
  
Args:  

- self
- ptr


Notes:
`ndarray.flags['OWN_DATA']` should be set as False.
The memory should not be freed by `__del__`.
#### __moveinit__


```Mojo
__moveinit__(out self, owned other: Self)
```  
Summary  
  
  
  
Args:  

- self
- other

#### get_ptr


```Mojo
get_ptr(self) -> UnsafePointer[SIMD[float16, 1]]
```  
Summary  
  
  
  
Args:  

- self
