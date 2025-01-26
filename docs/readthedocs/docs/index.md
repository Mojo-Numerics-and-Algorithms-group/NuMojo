# NuMojo

NuMojo is an numerics library for [Mojo](https://www.modular.com/mojo) similiar to [numpy](https://numpy.org/) for Python.

## Installation
[Installation](./getting_started/install.md)

## Examples
An example of n-dimensional array (`NDArray` type) goes as follows.

```mojo
import numojo as nm
from numojo.prelude import *


fn main() raises:
    # Generate two 1000x1000 matrices with random float64 values
    var A = nm.random.randn(Shape(1000, 1000))
    var B = nm.random.randn(Shape(1000, 1000))

    # Generate a 3x2 matrix from string representation
    var X = nm.fromstring[f32]("[[1.1, -0.32, 1], [0.1, -3, 2.124]]")

    # Print array
    print(A)

    # Array multiplication
    var C = A @ B

    # Array inversion
    var I = nm.inv(A)

    # Array slicing
    var A_slice = A[1:3, 4:19]

    # Get scalar from array
    var A_item = A[item(291, 141)]
    var A_item_2 = A.item(291, 141)
```

An example of matrix (`Matrix` type) goes as follows.

```mojo
from numojo import Matrix
from numojo.prelude import *


fn main() raises:
    # Generate two 1000x1000 matrices with random float64 values
    var A = Matrix.rand(shape=(1000, 1000))
    var B = Matrix.rand(shape=(1000, 1000))

    # Generate 1000x1 matrix (column vector) with random float64 values
    var C = Matrix.rand(shape=(1000, 1))

    # Generate a 4x3 matrix from string representation
    var F = Matrix.fromstring[i8](
        "[[12,11,10],[9,8,7],[6,5,4],[3,2,1]]", shape=(4, 3)
    )

    # Matrix slicing
    var A_slice = A[1:3, 4:19]
    var B_slice = B[255, 103:241:2]

    # Get scalar from matrix
    var A_item = A[291, 141]

    # Flip the column vector
    print(C[::-1, :])

    # Sort and argsort along axis
    print(nm.sort(A, axis=1))
    print(nm.argsort(A, axis=0))

    # Sum the matrix
    print(nm.sum(B))
    print(nm.sum(B, axis=1))

    # Matrix multiplication
    print(A @ B)

    # Matrix inversion
    print(A.inv())

    # Solve linear algebra
    print(nm.solve(A, B))

    # Least square
    print(nm.lstsq(A, C))
```

An example of ComplexNDArray is as follows, 
```mojo
import numojo as nm
from numojo.prelude import *


fn main() raises:
    # Create a complexscalar 5 + 5j
    var complexscalar = ComplexSIMD[cf32](re=5, im=5) 
    # Create complex array filled with (5 + 5j)
    var A = nm.full[cf32](Shape(1000, 1000), fill_value=complexscalar)
    # Create complex array filled with (1 + 1j)
    var B = nm.ones[cf32](Shape(1000, 1000))

    # Print array
    print(A)

    # Array slicing
    var A_slice = A[1:3, 4:19]

    # Array multiplication
    var C = A * B

    # Get scalar from array
    var A_item = A[item(291, 141)]
    # Set an element of the array
    A[item(291, 141)] = complexscalar
```
<!-- 
## Documentation
[Documenation](./docs/) -->

## Contibuting to NuMojo

If you would like to contribute to either [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) or its documentation pull requests are welcome.
