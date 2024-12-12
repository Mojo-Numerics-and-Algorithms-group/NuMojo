# Numojo Style Guide

In the interest of keeping our code clean and consistent, and enabling some automation for documentation the following simple standards will be required for new commits.

## File Level
All files must begin with a triple quoted docstring describing the functionality created by the file. It should be a single sentence with the first letter capitalized and ending with a period.
```python
"""
Document docstring describing what it does, if it is in an init file it will be the docstring for the module.
"""
```
All aliases and file-level variable definitions must have a docstring that describes what they are placed below the declaration.
```python
alias Example = Int
""" Aliases can be explained with docstrings and should if they exist in the global scope."""
```
Aliases should be snake_case if they are a value and CamelCase if they are a type. With the exception of the `DType` mapping types ex: `f32`. Alias names should clearly indicate what they are for and in addition to their docstring require no further information to understand assuming the reader understands the Mojo, and the domain.

## Functions

Functions should be snake_case, and describe what they do in as few words as possible, such that in addition to the docstring no further info is required.

The first line of a function docstring should summarize what the function does.
```python
"""
Description of the function.
"""
```
Next add the parameters, arguments, and returns if there are any separated from the summary by a new line. For functions and parameters start with either `Parameters:` or `Args:` followed by a new line-separated list of the parameters or arguments with the name of the parameter/arg followed by a `:` and a description the description should be a sentence starting with a capital letter and ending with a period. For returns separated from previous lines by a new line and start with `Returns:` then go to a new line and write a brief description of the return value, again as a sentence starting with a capitol letter and ending with a period. If the function does not return the `Returns:` section should be omitted. 

There is no need to add the type name to the arguments or parameters as the compiler handles that.
```rust
fn func[param:Copyable](arg1:param)->param:
    """

    Description of the function.

    Parameters:
        param: Each parameter should be listed and described.
        
    Args:
        arg1: Each argument should be listed and described.

    Returns:
        Describe what is returned.
    """
    ...
```

If the function has compile time constraints or raises `Error`s include sections similar to return that specify those constraints and possible errors.
```python
"""
Raises:
    A description of the errors raised by the function.

Constraints:
    If the functions use compile time constraints they should be listed here.
"""
```

## Structs
Structs should be CamelCase and describe what they do in as few words as possible, such that in addition to the docstring no further info is required.

The first line of a struct docstring should summarize what the struct does. It is not necessary to reiterate the structs name in the docstring. The parameters, and constraints of a struct should be included in the struct docstring in a similar way to functions.

```rust
struct AStruct[param:AnyType](AnyType):
    """
    Struct docstring describes basically what a struct does.

    Constraints:
        Limitations placed on the struct.

    Parameters:
        param: An example parameter.
    """
    ...
```

Fields and aliases should have a docstring below them describing what they are. They should be no longer than a single sentence and should start with a capital letter and end with a period.

```rust
struct AStruct[param:AnyType](AnyType):
    """
    Struct docstring describes basically what a struct does.

    Constraints:
        Limitations placed on the struct.

    Parameters:
        param: An example parameter.
    """
    
    var field: Int64
    """ Field Descriptions go below each field."""
    ...
```

Struct methods should follow the same rules as functions.

## Traits
Traits follow the same rules as Structs but there are no fields in traits.
