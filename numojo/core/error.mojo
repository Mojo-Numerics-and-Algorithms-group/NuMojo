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


struct NumojoError[
    category: String,
](Stringable, Writable):
    """
    Unified error type for all Numojo operations.

    Parameters:
        category: Type of error (e.g., "ShapeError", "IndexError").

    Args:
        message: Main error description.
        suggestion: Optional hint for fixing the error.
        location: Optional context about where error occurred.
    """

    var message: String
    var suggestion: Optional[String]
    var location: Optional[String]

    fn __init__(
        out self,
        message: StringLiteral,
        suggestion: StringLiteral,
        location: StringLiteral,
    ):
        self.message = message
        self.suggestion = Optional[String](suggestion)
        self.location = Optional[String](location)

    fn __init__(
        out self,
        message: String,
        suggestion: Optional[String] = None,
        location: Optional[String] = None,
    ):
        self.message = message
        self.suggestion = suggestion
        self.location = location

    fn __str__(self) -> String:
        var result = String("NuMojo Error\n")
        result += String("\tCategory  : ") + String(Self.category) + "\n"
        result += String("\tMessage   : ") + self.message + "\n"
        if self.location:
            result += String("\tLocation  : ") + self.location.value() + "\n"
        if self.suggestion:
            result += String("\tSuggestion: ") + self.suggestion.value() + "\n"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        """Write error information to a writer."""
        writer.write(self.__str__())


# ===----------------------------------------------------------------------===#
# Error Category Constants
# ===----------------------------------------------------------------------===#
# common error categories, might expand in future
alias IndexError = NumojoError[category="IndexError"]
alias ShapeError = NumojoError[category="ShapeError"]
alias BroadcastError = NumojoError[category="BroadcastError"]
alias MemoryError = NumojoError[category="MemoryError"]
alias ValueError = NumojoError[category="ValueError"]
alias ArithmeticError = NumojoError[category="ArithmeticError"]
