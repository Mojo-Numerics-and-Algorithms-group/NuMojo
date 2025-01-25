



# indexer_collection_element

##  Module Summary
  

## IndexerCollectionElement

### IndexerCollectionElement Summary
  
  
The IndexerCollectionElement trait denotes a trait composition of the `Indexer` and `CollectionElement` traits.  

### Parent Traits
  

- AnyType
- CollectionElement
- Copyable
- Indexer
- Movable
- UnknownDestructibility
  

### Functions

#### __copyinit__


```Mojo
__copyinit__(out self: _Self, existing: _Self, /)
```  
Summary  
  
Create a new instance of the value by copying an existing one.  
  
Args:  

- self
- existing: The value to copy.

#### __moveinit__


```Mojo
__moveinit__(out self: _Self, owned existing: _Self, /)
```  
Summary  
  
Create a new instance of the value by moving the value of another.  
  
Args:  

- self
- existing: The value to move.

#### __index__


```Mojo
__index__(self: _Self) -> Int
```  
Summary  
  
Return the index value.  
  
Args:  

- self
