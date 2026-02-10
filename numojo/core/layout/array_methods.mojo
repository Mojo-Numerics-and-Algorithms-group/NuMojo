"""
This module defines the `NewAxis` struct, which is used to represent the insertion of new axes into array shapes,
similar to the concept of `None` or `np.newaxis` in NumPy. The `NewAxis` struct can be used to indicate where
a new singleton dimension should be added to an array, enabling advanced indexing and broadcasting operations.

Example usage:
    newaxis = NewAxis()        # Adds a single new axis
    newaxis3 = NewAxis(3)      # Adds three new axes

Attributes:
    num (Int): The number of new axes to add. Defaults to 0 (single new axis).
"""

comptime newaxis = NewAxis()


# TODO: add an initializer with int field to specify number of new axes to add! Future work, for now, keep it simple.
struct NewAxis(Hashable, ImplicitlyCopyable, Movable, Stringable):
    """
    Represents a new axis to be inserted into an array's shape.

    The `NewAxis` struct is typically used in advanced indexing to add singleton dimensions
    to arrays, facilitating broadcasting and reshaping operations.

    Attributes:
        num (Int): The number of new axes to add.
    """

    var num: Int

    fn __init__(out self):
        """
        Initializes a `NewAxis` instance with a default of one new axis.

        Sets `num` to 0, which can be interpreted as a single new axis.
        """
        self.num = 0

    fn __init__(out self, num: Int):
        """
        Initializes a `NewAxis` instance with a specified number of new axes.

        Args:
            num: The number of new axes to add.
        """
        self.num = num

    fn __repr__(self) -> String:
        """
        Returns a string representation of the `NewAxis` instance.

        Returns:
            String: The string "numojo.newaxis()".
        """
        return "numojo.newaxis()"

    fn __str__(self) -> String:
        """
        Returns a string representation of the `NewAxis` instance.

        Returns:
            String: The string "numojo.newaxis()".
        """
        return "numojo.newaxis()"

    fn __eq__(self, other: Self) -> Bool:
        """
        Checks equality between two `NewAxis` instances.

        Returns:
            Bool: True if the instances are considered equal.
        """
        return True

    fn __ne__(self, other: Self) -> Bool:
        """
        Checks inequality between two `NewAxis` instances.

        Returns:
            Bool: False if the instances are considered equal.
        """
        return False
