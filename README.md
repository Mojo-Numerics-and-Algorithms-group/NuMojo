# Mojo-Arrays
## Description
A little project for vectorized N-Dimensional Arrays in native mojo
## Goals
* Vectorized, robustly statically typed, nd arrays with all of the standard matrix operations
* Eventually to be made compatible with mojo implementations or ABI of LAPACK and BLAS for linear algebra operations.
* Compatability with numpy arrays eventually for interactions with the rest of python.
* Once the standalone compiler is available and we can use more MLIR features [MLIR tensor](https://mlir.llvm.org/docs/Dialects/TensorOps/) and [Tensor Operations Standards](https://mlir.llvm.org/docs/Dialects/TOSA/) will be implemented
## Current
* The current state of this project is for toy usage and nailing down desirable behavior for the complete future version.
* DTypePointer-based data storage 2d arrays
* Basic guard rails for getting and setting
* add, subtract, mult, truediv, floordiv, and pow for both arrays of the same size, slices of arrays, and single values(but not commutatively yet for all)
* Arrange, transpose, shape, and meshgrid,to_numpy, and print basic implementations
* All single SIMD input operations from Standard Math: trig(except atan2) and hyperbolic trig, etc.
* Getting and setting from slices.
## Setup
* Mojo module creation is not working as of 7 August 2023 so for now you have to copy the full contents of VArray2D into a notebook.
