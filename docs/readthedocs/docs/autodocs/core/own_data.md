



# own_data

##  Module Summary
  

## OwnData

### OwnData Summary
  
  
  

### Parent Traits
  

- AnyType
- UnknownDestructibility

### Fields
  
  
* ptr `UnsafePointer[SIMD[dtype, 1]]`  

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
`ndarray.flags['OWN_DATA']` should be set as True.
The memory should be freed by `__del__`.

```Mojo
__init__(out self, ptr: UnsafePointer[SIMD[dtype, 1]])
```  
Summary  
  
Do not use this if you know what it means. If the pointer is associated with another array, it might cause dangling pointer problem.  
  
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
get_ptr(self) -> UnsafePointer[SIMD[dtype, 1]]
```  
Summary  
  
  
  
Args:  

- self
