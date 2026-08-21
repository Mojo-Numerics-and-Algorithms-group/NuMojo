# ===----------------------------------------------------------------------=== #
# NuMojo: Array methods
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Array methods (numojo.core.layout.array_methods).
==================================================
The `NewAxis` struct, used to represent the insertion of new axes into array
shapes, similar to `None` / `np.newaxis` in NumPy.

Indicates where a new singleton dimension should be added to an array,
enabling advanced indexing and broadcasting operations.

Exports
-------
- `NewAxis`: Add singleton dimension.
- `newaxis`: Default `NewAxis` instance.

Examples:
    ```mojo
    var a = NewAxis()      # Adds a single new axis
    var b = NewAxis(3)     # Adds three new axes
    ```
"""

comptime newaxis = NewAxis()


# TODO: add an initializer with int field to specify number of new axes to add! Future work, for now, keep it simple.
struct NewAxis(Hashable, ImplicitlyCopyable, Movable, Writable):
    """
    Represents a new axis to be inserted into an array's shape.

    The `NewAxis` struct is typically used in advanced indexing to add singleton dimensions
    to arrays, facilitating broadcasting and reshaping operations.

    Attributes:
        num (Int): The number of new axes to add.
    """

    var num: Int

    def __init__(out self):
        """
        Initializes a `NewAxis` instance with a default of one new axis.

        Sets `num` to 0, which can be interpreted as a single new axis.
        """
        self.num = 0

    def __init__(out self, num: Int):
        """
        Initializes a `NewAxis` instance with a specified number of new axes.

        Args:
            num: The number of new axes to add.
        """
        self.num = num

    def __repr__(self) -> String:
        """
        Returns a string representation of the `NewAxis` instance.

        Returns:
            String: The string "numojo.newaxis()".
        """
        return "numojo.newaxis()"

    def __str__(self) -> String:
        """
        Returns a string representation of the `NewAxis` instance.

        Returns:
            String: The string "numojo.newaxis()".
        """
        return "numojo.newaxis()"

    def __eq__(self, other: Self) -> Bool:
        """
        Checks equality between two `NewAxis` instances.

        Returns:
            Bool: True if the instances are considered equal.
        """
        return True

    def __ne__(self, other: Self) -> Bool:
        """
        Checks inequality between two `NewAxis` instances.

        Returns:
            Bool: False if the instances are considered equal.
        """
        return False
