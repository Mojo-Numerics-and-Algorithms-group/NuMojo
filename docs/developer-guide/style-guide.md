# Numojo Style Guide

In the interest of keeping our code clean and consistent, and enabling some automation for documentation the following simple standards will be required for new commits.

Run `pixi run check_standards` before opening a PR — it reports (without rewriting anything) any file that doesn't match this guide: missing headers, missing module docstrings, stale `Exports` sections, malformed section separators, or function docstrings that don't match the actual signature. See [pre-pr-checks.md](pre-pr-checks.md).

## File Level

Every file starts with a license header, formatted as an 80-character separator, a one-line description of the file, the license lines, then a closing separator:
```mojo
# ===----------------------------------------------------------------------=== #
# NuMojo: One-line description of the file.
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
```

After the header, every file must have a triple-quoted module docstring: a title line naming the module (in the form `Title (dotted.module.path)`) underlined with `=` characters of the exact same length, a short description, and, for any module that exports names other consumers should import, an `Exports` section listing them.
```mojo
"""
Title (numojo.path.to.module).
===============================
One or two sentence description of what the module provides.

Exports
-------
- `name`: What it is.
"""
```

Below the module docstring, imports are grouped under their own 80-character separators (`Stdlib`, `External`, `NuMojo`) — see [pre-pr-checks.md](pre-pr-checks.md) and `pixi run check_standards` for the automated checks; `scripts/organize_mojo_imports.py` will sort and group them for you.

All comptimes and file-level variable definitions must have a docstring that describes what they are placed below the declaration.
```mojo
comptime Example = Int
""" Aliases can be explained with docstrings and should if they exist in the global scope."""
```
Aliases should be snake_case if they are a value and CamelCase if they are a type. With the exception of the `DType` mapping types ex: `f32`. Alias names should clearly indicate what they are for and in addition to their docstring require no further information to understand assuming the reader understands the Mojo, and the domain.

## Functions

Functions should be snake_case, and describe what they do in as few words as possible, such that in addition to the docstring no further info is required.

The first line of a function docstring should summarize what the function does.
```mojo
"""
Description of the function.
"""
```
Next add the parameters, arguments, and returns if there are any separated from the summary by a new line. For functions and parameters start with either `Parameters:` or `Args:` followed by a new line-separated list of the parameters or arguments with the name of the parameter/arg followed by a `:` and a description the description should be a sentence starting with a capital letter and ending with a period. For returns separated from previous lines by a new line and start with `Returns:` then go to a new line and write a brief description of the return value, again as a sentence starting with a capitol letter and ending with a period. If the function does not return the `Returns:` section should be omitted. 

There is no need to add the type name to the arguments or parameters as the compiler handles that.

```mojo
def func[param:Copyable](arg1:param)->param:
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

If the function has compile time constraints or raises errors, include sections after `Returns:` that specify those constraints and possible errors. All errors raised in NuMojo are `NumojoError`, so `Raises:` should describe the conditions under which a `NumojoError` is raised rather than naming a Python-style exception type.
```mojo
"""
Returns:
    Describe what is returned.

Raises:
    NumojoError: A description of the condition that raises it.

Constraints:
    If the functions use compile time constraints they should be listed here.
"""
```

## Structs
Structs should be CamelCase and describe what they do in as few words as possible, such that in addition to the docstring no further info is required.

The first line of a struct docstring should summarize what the struct does. It is not necessary to reiterate the structs name in the docstring. The parameters, and constraints of a struct should be included in the struct docstring in a similar way to functions.

```mojo
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

Fields and comptimes should have a docstring below them describing what they are. They should be no longer than a single sentence and should start with a capital letter and end with a period.

```mojo
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
