Overall the goal is to make a powerful and broad featured library for numerical and scientific computing. In order to make a transition from python easy we should largely follow the conventions of numpy and scipy, but we should allow for the flexibility to do things differently where it improves user expereince or greatly effects performance.

* Implement tensor version all SIMDable standard library math functions (mostly done waiting on std lib [issue 2492](https://github.com/modularml/mojo/issues/2492))
* Build statistics functions
* Build optimizers (newton raphson, bisection,etc)
* Build function approximators

## Notional organization of functions and features
* Most common functions at top level like in numpy (trig, basic stats, masking, querying, and mapping)
* Other features should be organized either by type of math or intended utilization
    * Statistics probably warents its own sub module
    * Regression could either be a submodule of statistics or its own module
    * kernal density esimators almost certainly should be part of the statistics sub module
    * It is tempting to make a algebra, calculus and trig submodules however that may end up being confusing as those topics include so many things, it may be better to organize what would be there into functional applications instead
    * An Ordinary differential equations submodule would include solvers and utilities
    * The optimizer sub module could mirror scipy.optimize but minimally should include all of those functions
* There will need to be discussions and bike shedding about both organization of the library and what does and doesn't belong.
