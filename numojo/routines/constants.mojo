# ===----------------------------------------------------------------------=== #
# NuMojo: Constants
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Constants (numojo.routines.constants)

This module defines physical and mathematical constants for use in numerical computations.
The constants are defined as class attributes of the `Constants` class, which is designed to be immutable and efficient for compile-time evaluation.
"""


struct Constants(AnyType, Copyable, Movable):
    """Define constants.

    Use comptime for compile time evaluation of indefinite precision.
    ```mojo
    import numojo as nm
    def main():
        var pi: Float64 = nm.pi
        print("Float64:", pi*pi*pi*pi*pi*pi)
        print("Literal:", nm.pi*nm.pi*nm.pi*nm.pi*nm.pi*nm.pi)
    ```
    """

    comptime c = 299_792_458
    comptime pi = 3.1415926535897932384626433832795028841971693937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555954930381966446229489
    comptime e = 2.71828182845904523536028747135266249775724609375
    comptime hbar = 1.0545718176461563912626e-34

    def __init__(out self):
        """
        Initializes the constants.
        """
        pass

    def __del__(deinit self):
        """
        Deletes the constants.
        """
        pass
