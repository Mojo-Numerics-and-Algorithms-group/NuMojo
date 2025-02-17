# ===----------------------------------------------------------------------=== #
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Implements Flags type.
"""

from numojo.core.ndshape import NDArrayShape
from numojo.core.ndstrides import NDArrayStrides


@register_passable
struct Flags:
    """
    Information about the memory layout of the array.
    The Flags object can be accessed dictionary-like.
    or by using lowercased attribute names.
    Short names are available for convenience when using dictionary-like access.
    """

    # attributes
    var C_CONTIGUOUS: Bool
    """C_CONTIGUOUS (C): The data is in a C-style contiguous segment."""
    var F_CONTIGUOUS: Bool
    """F_CONTIGUOUS (F): The data is in a Fortran-style contiguous segment."""
    var OWNDATA: Bool
    """OWNDATA (O): The array owns the underlying data buffer."""
    var WRITEABLE: Bool
    """
    The data area can be written to.
    If it is False, the data is read-only and be blocked from writing.
    The WRITEABLE field of a view or slice is inherited from the array where
    it is derived. If the parent object is not writeable, the child object is 
    also not writeable. If the parent object is writeable, the child object may 
    be not writeable.
    """
    var FORC: Bool
    """F_CONTIGUOUS or C_CONTIGUOUS."""

    # === ---------------------------------------------------------------- === #
    # Life cycle dunder methods
    # === ---------------------------------------------------------------- === #

    fn __init__(
        out self,
        c_contiguous: Bool,
        f_contiguous: Bool,
        owndata: Bool,
        writeable: Bool,
    ):
        """
        Initializes the Flags object with provided information.

        Args:
            c_contiguous: The data is in a C-style contiguous segment.
            f_contiguous: The data is in a Fortran-style contiguous segment.
            owndata: The array owns the underlying data buffer.
            writeable: The data area can be written to.
                If owndata is False, writeable is forced to be False.
        """

        self.C_CONTIGUOUS = c_contiguous
        self.F_CONTIGUOUS = f_contiguous
        self.OWNDATA = owndata
        self.WRITEABLE = writeable and owndata
        self.FORC = f_contiguous or c_contiguous

    fn __init__(
        out self,
        shape: NDArrayShape,
        strides: NDArrayStrides,
        owndata: Bool,
        writeable: Bool,
    ) raises:
        """
        Initializes the Flags object according the shape and strides information.

        Args:
            shape: The shape of the array.
            strides: The strides of the array.
            owndata: The array owns the underlying data buffer.
            writeable: The data area can be written to.
                If owndata is False, writeable is forced to be False.
        """

        self.C_CONTIGUOUS = (
            True if (strides[-1] == 1) or (shape[-1] == 1) else False
        )
        self.F_CONTIGUOUS = (
            True if (strides[0] == 1) or (shape[0] == 1) else False
        )
        self.OWNDATA = owndata
        self.WRITEABLE = writeable and owndata
        self.FORC = self.F_CONTIGUOUS or self.C_CONTIGUOUS

    fn __init__(
        out self,
        shape: Tuple[Int, Int],
        strides: Tuple[Int, Int],
        owndata: Bool,
        writeable: Bool,
    ):
        """
        Initializes the Flags object according the shape and strides information.

        Args:
            shape: The shape of the array.
            strides: The strides of the array.
            owndata: The array owns the underlying data buffer.
            writeable: The data area can be written to.
                If owndata is False, writeable is forced to be False.
        """

        self.C_CONTIGUOUS = (
            True if (strides[1] == 1) or (shape[1] == 1) else False
        )
        self.F_CONTIGUOUS = (
            True if (strides[0] == 1) or (shape[0] == 1) else False
        )
        self.OWNDATA = owndata
        self.WRITEABLE = writeable and owndata
        self.FORC = self.F_CONTIGUOUS or self.C_CONTIGUOUS

    fn __copyinit__(out self, other: Self):
        """
        Initializes the Flags object by copying the information from
        another Flags object.

        Args:
            other: The Flags object to copy information from.
        """

        self.C_CONTIGUOUS = other.C_CONTIGUOUS
        self.F_CONTIGUOUS = other.F_CONTIGUOUS
        self.OWNDATA = other.OWNDATA
        self.WRITEABLE = other.WRITEABLE
        self.FORC = other.FORC

    # === ---------------------------------------------------------------- === #
    # Get and set dunder methods
    # === ---------------------------------------------------------------- === #

    fn __getitem__(self, key: String) raises -> Bool:
        """
        Get the value of the fields with the given key.
        The Flags object can be accessed dictionary-like.
        Short names are available for convenience.

        Args:
            key: The key of the field to get.

        Returns:
            The value of the field with the given key.
        """
        if (
            (key != "C_CONTIGUOUS")
            and (key != "C")
            and (key != "F_CONTIGUOUS")
            and (key != "F")
            and (key != "OWNDATA")
            and (key != "O")
            and (key != "WRITEABLE")
            and (key != "W")
            and (key != "FORC")
        ):
            raise Error(
                String(
                    "\nError in `Flags.__getitem__()`: "
                    "Invalid field name or short name: {}".format(key)
                )
            )
        if (key == "C_CONTIGUOUS") or (key == "C"):
            return self.C_CONTIGUOUS
        elif (key == "F_CONTIGUOUS") or (key == "F"):
            return self.F_CONTIGUOUS
        elif (key == "OWNDATA") or (key == "O"):
            return self.OWNDATA
        elif (key == "WRITEABLE") or (key == "W"):
            return self.WRITEABLE
        else:
            return self.FORC
