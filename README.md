# Mojo-Arrays
A little project for vectorized nd arrays in native mojo
## Goals
* Vectorized, robustly statically typed, nd arrays with all of the standard operations
* Eventually to be made compatible mojo implementations of LAPACK and BLAS for linear algebra operations.
* Compatability with numpy arrays eventaully. 
## Current
* DTypePointer based data storage
* Basic guard rails for getting and setting
* add, mult, truediv, floordiv, and pow for both arrays and single values
* Arrange and meshgrid basic implementations
* Getting and setting from slices(1d only so far, setting is a bit ruff)
