# ROADMAP

Overall the goal is to make a powerful and broad featured library for numerical and scientific computing. In order to make a transition from python easy we should largely follow the conventions of numpy and scipy, but we should allow for the flexibility to do things differently where it improves user experience or greatly effects performance.

With that in mind NuMojo as a project is in an early stage of development. If you notice a missing feature either build it and make a pull request or make a feature request.

### TASKS
* Implement tensor version all SIMDable standard library math functions (mostly done waiting on std lib [issue 2492](https://github.com/modularml/mojo/issues/2492))
* Build statistics functions
* Build optimizers (newton raphson, bisection,etc)
* Build function approximators

## N Dimensional Arrays
Now that Modular has decided to no longer support Tensor and to open source and deprecate it NuMojo intends to take Tensor and Make it our own Once they do.

Which means that we will be trying to add many of the features from numpy.array that tensor currently lacks, while not sacrificing performance.

## Notional organization of functions and features
* Most common functions at top level like in numpy (trig, basic stats, masking, querying, and mapping)
* Other features should be organized either by type of math or intended utilization
    * Statistics probably merits its own sub module
    * Regression could either be a submodule of statistics or its own module
    * kernel density estimators almost certainly should be part of the statistics sub module
    * It is tempting to make a algebra, calculus and trig submodules however that may end up being confusing as those topics include so many things, it may be better to organize what would be there into functional applications instead
    * An Ordinary differential equations submodule would include solvers and utilities
    * The optimizer sub module could mirror scipy.optimize but minimally should include all of those functions
* There will need to be discussions and bike shedding about both organization of the library and what does and doesn't belong.
