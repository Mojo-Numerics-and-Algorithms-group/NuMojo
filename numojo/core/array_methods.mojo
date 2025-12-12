comptime newaxis: NewAxis = NewAxis()
comptime ellipsis: Ellipsis = Ellipsis()


struct Ellipsis(Stringable):
    """
    Represents an ellipsis (`...`) used in array slicing to indicate the inclusion of all remaining dimensions.
    This can be used to simplify slicing operations when the exact number of dimensions is not known or when you want to include all remaining dimensions without explicitly specifying them.

    Example:
        ```
        from numojo.prelude import *
        from numojo.routines.creation import arange

        var arr = arange(Shape(3, 4, 5, 6))
        sliced_arr = arr[nm.ellipsis, 2]  # Equivalent to arr[:, :, :, 2]
        ```
    """

    fn __init__(out self):
        """
        Initializes an Ellipsis instance.
        """
        pass

    fn __repr__(self) -> String:
        """
        Returns a string representation of the Ellipsis instance.

        Returns:
            Str: The string "Ellipsis()".
        """
        return "numojo.ellipsis()"

    fn __str__(self) -> String:
        """
        Returns a string representation of the Ellipsis instance.

        Returns:
            Str: The string "Ellipsis()".
        """
        return "numojo.ellipsis()"

    fn __eq__(self, other: ellipsis) -> Bool:
        """
        Checks equality between two Ellipsis instances.
        """
        return True

    fn __ne__(self, other: ellipsis) -> Bool:
        """
        Checks inequality between two Ellipsis instances.
        """
        return False


struct NewAxis(Stringable):
    fn __init__(out self):
        """
        Initializes a NewAxis instance.
        """
        pass

    fn __repr__(self) -> String:
        """
        Returns a string representation of the NewAxis instance.

        Returns:
            Str: The string "NewAxis()".
        """
        return "numojo.newaxis()"

    fn __str__(self) -> String:
        """
        Returns a string representation of the NewAxis instance.

        Returns:
            Str: The string "NewAxis()".
        """
        return "numojo.newaxis()"

    fn __eq__(self, other: NewAxis) -> Bool:
        """
        Checks equality between two NewAxis instances.
        """
        return True

    fn __ne__(self, other: NewAxis) -> Bool:
        """
        Checks inequality between two NewAxis instances.
        """
        return False
