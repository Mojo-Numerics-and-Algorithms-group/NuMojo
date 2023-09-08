The goal of the ndarray package is to provide for Mojo what Numpy provided for Python,
a stable easy-to-use basis for array-based mathematical computation.The primary reason that we
are not just porting Numpy is that Mojo has the ability to be much faster than the C and 
Fortran-backed Numpy library and a Mojo native will be more portable and able to run 
on different compute types such as GPU or TPU than a Numpy port.

Mojo-Arrays was the first third-party Mojo package to be made publically available.

In the short term the goal is to nail down API behavior like calling functions 
or getting/setting data. Until more metaprogramming features come along we will 
be sticking with 2D(and maybe 3D or 4D if we need to prove something that can only 
exist at those dimensions) arrays to prove concepts. For advanced things like 
Linear Algebra we may lean on Numpy until we can do some serious work on algorithms,
or implement BLAS and LAPACK.

In the long term, the goal is to be at least 80% Mojo and MLIR and to implement the Tensor,
TOSA, and Linear Algebra MLIR specifications in Mojo as well  as any specification that would 
benefit the project.

Since the Mojo lang is very young and constantly changing there is no real rush to get something 
production-ready now, but If no one makes something like this very early on there will be 
dozens of competing standards and a huge mess of compatibility issues.

Current State:
* 2-dimensional Arrays with numerical types
* SIMD math is implemented and vectorized for all arithmetic operations as dunder methods (math, imath, but not rmath)
* SIMD level math is implemented for all single input functions from the math library
* Getting and setting is done with safety Errors for combinations of slices and ints
* Creation methods are eye(for identity), arrange, zeros, and once
* Functions can be accessed by calling the Array constructor without arguments
* A 2d boolean array type exists, but is far from ready

Next Steps:
* Clean up and parallelize the functions of the existing codebase
* Look into tiling and unrolling to determine where they make sense
* Create a matmul implementation for 2d arrays
* Implement methods for array2d which have boolean array results
* Add all SIMD boolean methods to the boolean array
* Determine a good methodology for increasing dimensions without exponentially increasing the size of the code base.
