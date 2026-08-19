# ===----------------------------------------------------------------------=== #
# NuMojo: Error handling for Numojo library operations.
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Error Handling (numojo.core.error)
==================================

Unified error system for NuMojo operations.

Provides a simple, categorized error type for all NuMojo operations with
clear error messages and suggestions for fixing issues.

Exports
-------
- `NumojoError`: Unified error type with categories and suggestions.

Categories:
    - index: Indexing errors
    - shape: Shape mismatch errors
    - broadcast: Broadcasting errors
    - memory: Memory allocation errors
    - value: Value errors
    - arithmetic: Arithmetic operation errors
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.format.tstring import TString
from std.os import abort

comptime RED_COLOR: String = "\033[31m"
comptime END_COLOR: String = "\033[0m"

# TODO: Remove suggestion field and remove it from existing instances.
struct NumojoError(Writable):
    """
    Unified error type for all Numojo operations.

    Args:
        category: Type of error (e.g., "ShapeError", "IndexError").
        message: Main error description and suggestion.
        location: Optional context about where error occurred.

    Notes:
        All NumojoErrors use a single unified type with different categories for better organization.
        Error messages follow the format: "Category: Specific problem. Expected X but got Y."
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

    def __init__(
        out self,
        category: StringLiteral,
        message: StringLiteral,
        location: StringLiteral,
    ):
        var err_dict = materialize[Self.ErrorDict]()
        try:
            self.category = err_dict[category]
        except:
            abort("NumojoError: Invalid error type provided.")
        self.message = message
        self.location = location

    def __init__(
        out self,
        category: StringLiteral,
        message: String,
        location: Optional[String] = None,
    ):
        var err_dict = materialize[Self.ErrorDict]()
        try:
            self.category = err_dict[category]
        except:
            abort("NumojoError: Invalid error type provided.")
        self.message = message
        self.location = location

    def __init__(
        out self,
        category: StringLiteral,
        message: TString,
        location: StringLiteral,
    ):
        var err_dict = materialize[Self.ErrorDict]()
        try:
            self.category = err_dict[category]
        except:
            abort("NumojoError: Invalid error type provided.")
        self.message = String(message)
        self.location = location

    def __str__(self) -> String:
        """Return string representation of the error with formatting."""
        var result = (
            RED_COLOR + String(self.category) + String(": ") + self.message
        )
        if self.location:
            result += String(" [at ") + self.location.value() + String("]")
        result += END_COLOR
        return result

    def write_to[W: Writer](self, mut writer: W):
        """Write error information to a writer."""
        writer.write(
            RED_COLOR + String(self.category) + String(": ") + self.message
        )
        if self.location:
            writer.write(String(" [at ") + self.location.value() + String("]"))
        writer.write(END_COLOR)


def terminate(message: String):
    """
    Abort the program with the given error message.

    Args:
        message: The error message to display before aborting.

    Notes:
        This function is used for fatal, unrecoverable errors that require immediate termination.
        The message will be displayed in red color before the program exits.
    """
    abort(RED_COLOR + message + END_COLOR)
