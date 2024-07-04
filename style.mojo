"""
Document docstring decribing what it does, if it is in an init file it will be the docstring for the module
"""

# ===----------------------------------------------------------------------=== #
# Subsection header, used to divide code in a file by functional parts (ingored by doc generation)
# ===----------------------------------------------------------------------=== #

alias Example = Int
"""Aliases can be explained with docstrings and should if they exist in the global scope."""

fn func[param:Copyable](arg1:param)->param:
    """
    Description of the function.

    Constraints:
        If the functions use compile time constraints they should be listed here.

    Parameters:
        param: Each parameter should be listed and described.
    
    Args:
        arg1: Each arguament should be listed and described.
    
    Returns:
        Describe what is returned.
    """
    return arg1


fn func1[param:Copyable](arg1:param)raises->param:
    """
    Description of the function.

    Parameters:
        param: Each parameter should be listed and described.
    
    Args:
        arg1: Each arguament should be listed and described.
    
    Raises:
        A description of the errors raised by the function.

    Returns:
        Describe what is returned.
    """
    return arg1

struct AStruct[param:AnyType](AnyType):
    """
    Struct docstring describing basically what a struct does.

    Constraints:
        Limitations placed on the struct.

    Parameters:
        param: An example parameter.
    """
    
    var field: Int64
    """Field Descriptions go below each field."""

    fn func(self)->None:
        """
        Function docstring like previosly shown.
        """
        return None

trait ATrait:
    """
    Describe the trait.
    """
    fn func(self)->None:
        """
        Function docstring like previosly shown.
        """
        pass