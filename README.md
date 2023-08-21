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
* Arrange, transpose, shape, to_numpy, and print basic implementations
* All single SIMD input operations from Standard Math: trig(except atan2) and hyperbolic trig, etc.
* Getting and setting from slices.
## Setup
* Copy the text from all of the files in NDArray into files with the same names as the sources into a folder named NDArray
* see [Introduction](https://github.com/MadAlex1997/Mojo-Arrays/blob/main/Introduction.ipynb) for basics on how to use NDArray in its current state.
