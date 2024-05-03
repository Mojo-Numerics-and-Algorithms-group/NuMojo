# Purpose 
NuMojo is a libary for numerical computing in Mojo similiar to numpy in python

NuMojo extends (most of) the standard library math functions to work on tensor inputs

NuMojo intends to capture a wide swath of numerics capability present in the python packages numpy,scipy and scikit such as
* Linear/Tensor Algebra
* Integration
* Derivatives
* Optimizers
* Function approximators

NuMojo intends to try and get the most out of the capabilities of Mojo including vectorization, parallelization, and GPU acceleration(once available).

NuMojo intends to be a building block for other Mojo packages that need fast math under the hood without the added weight of a ML back and forward propigation system

# What NuMojo is not
NuMojo is not a machine learning library, it will never include backproppagation in the base library. In order to make it as fast as possible for calculation and simulation
