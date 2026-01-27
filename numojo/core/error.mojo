"""
Error handling for Numojo library operations.

This module provides a simple, unified error system for the Numojo library.
All errors use a single NumojoError type with different categories for
better organization while keeping the implementation simple. This provides a better user experience by
providing clear error message and suggestions for fixing the error.

Currently we have a few common error categories like
- IndexError
- ShapeError
- BroadcastError
- MemoryError
- ValueError
- ArithmeticError

We can expand this list in the future as needed.
"""
from os import abort

comptime RED_COLOR: String = "\033[31m"
comptime END_COLOR: String = "\033[0m"


# TODO: remove suggestion field and remove it from existing instances.
struct NumojoError(Stringable, Writable):
    """
    Unified error type for all Numojo operations.

    Parameters:

    Args:
        category: Type of error (e.g., "ShapeError", "IndexError").
        message: Main error description and suggestion.
        location: Optional context about where error occurred.
    """

    comptime ErrorDict: Dict[String, String] = {
        "index": "IndexError",
        "shape": "ShapeError",
        "broadcast": "BroadcastError",
        "memory": "MemoryError",
        "value": "ValueError",
        "arithmetic": "ArithmeticError",
    }
    var category: String
    var message: String
    var location: Optional[String]

    fn __init__(
        out self,
        category: StringLiteral,
        message: StringLiteral,
        location: StringLiteral,
    ):
        err_dict = materialize[Self.ErrorDict]()
        try:
            self.category = err_dict[category]
        except:
            abort("NumojoError: Invalid error type provided.")
        self.message = message
        self.location = location

    fn __init__(
        out self,
        category: StringLiteral,
        message: String,
        location: Optional[String] = None,
    ):
        err_dict = materialize[Self.ErrorDict]()
        try:
            self.category = err_dict[category]
        except:
            abort("NumojoError: Invalid error type provided.")
        self.message = message
        self.location = location

    fn __str__(self) -> String:
        var result = (
            RED_COLOR + String(self.category) + String(": ") + self.message
        )
        if self.location:
            result += String(" [at ") + self.location.value() + String("]")
        result += END_COLOR.__str__()
        return result

    fn write_to[W: Writer](self, mut writer: W):
        """Write error information to a writer."""
        writer.write(
            RED_COLOR + String(self.category) + String(": ") + self.message
        )
        if self.location:
            writer.write(String(" [at ") + self.location.value() + String("]"))
        writer.write(END_COLOR)


# Use this for fatal errors that should abort the program.
fn terminate(message: String):
    """Abort the program with the given error message."""
    abort(RED_COLOR + message + END_COLOR)
