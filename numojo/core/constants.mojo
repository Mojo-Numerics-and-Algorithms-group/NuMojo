"""
# ===----------------------------------------------------------------------=== #
# Implements Constants
# Last updated: 2024-06-16
# ===----------------------------------------------------------------------=== #
"""


@value
struct Constants(AnyType):
    """Define constants.

    Use alias for compile time evaluation of indefinite precision.
    ```mojo
    import numojo as nm
    fn main():
        var pi: Float64 = nm.pi
        print("Float64:", pi*pi*pi*pi*pi*pi)
        print("Literal:", nm.pi*nm.pi*nm.pi*nm.pi*nm.pi*nm.pi)
    ```
    ```console
    Float64: 961.38919357530415
    Literal: 961.38919357530449
    ```
    """

    alias c = 299_792_458
    alias pi = 3.1415926535897932384626433832795028841971693937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555954930381966446229489
    alias e = 2.71828182845904523536028747135266249775724609375

    fn __init__(inout self):
        """
        Initializes the constants.
        """
        pass

    fn __del__(owned self):
        """
        Deletes the constants.
        """
        pass
