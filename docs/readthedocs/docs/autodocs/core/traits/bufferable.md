



# bufferable

##  Module Summary
  

## Bufferable

### Bufferable Summary
  
  
Data buffer types that can be used as a container of the underlying buffer.  

### Parent Traits
  

- AnyType
- UnknownDestructibility
  

### Functions

#### __init__


```Mojo
__init__(out self: _Self, size: Int)
```  
Summary  
  
  
  
Args:  

- self
- size


```Mojo
__init__(out self: _Self, ptr: UnsafePointer[SIMD[float16, 1]])
```  
Summary  
  
  
  
Args:  

- self
- ptr

#### __moveinit__


```Mojo
__moveinit__(out self: _Self, owned other: _Self)
```  
Summary  
  
  
  
Args:  

- self
- other

#### get_ptr


```Mojo
get_ptr(self: _Self) -> UnsafePointer[SIMD[float16, 1]]
```  
Summary  
  
  
  
Args:  

- self
