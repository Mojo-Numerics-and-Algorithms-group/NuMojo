import numojo as nm
from numojo.prelude import *


def main() raises:
    var A = nm.random.randn(Shape(3, 3))
    var B = nm.random.randn(Shape(3, 3))

    var C = A @ B
    print("A @ B:")
    print(C)

    var D = nm.sin(A) + nm.cos(B)
    print("sin(A) + cos(B):")
    print(D)

    var E = nm.sum(C)
    print("sum(A @ B):", E)

    print("NDArray basics:")
    var X = nm.arange[i32](9).reshape(Shape(3, 3))
    print(X)
    var X_slice = X[1:3, 0:2]
    print("X[1:3, 0:2]:")
    print(X_slice)
    var x_item = X[Item(1, 2)]
    print("X[Item(1, 2)]:", x_item)

    print("ComplexNDArray basics:")
    var Z = nm.arange[cf32](
        CScalar[cf32](1.0, 2.0),
        CScalar[cf32](17.0, 18.0),
        CScalar[cf32](1.0, 1.0),
    )
    print(Z)
    var Z_sum = Z + Z
    print("Z + Z:")
    print(Z_sum)
    var z_item = Z.item(0)
    print("Z.item(0):", z_item.re, z_item.im)
    var z_slice = Z[1:3]
    print("Z[1:3]:")
    print(z_slice)
