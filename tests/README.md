# Testing guide

Inorder to ensure that all of our code works and that a new code doesn't break old code we need to instate a testing policy (which we can hopefully implement a CI/CD process for running).

Due to the similiarity with NumPy, a Numpy equivelent comparison will be our method for confirming validity.

The functions contained in utils_for_test.mojo `check` and `check_is_close` will be used to compare NuMojo results with NumPy equivelents.

Each test must be in it's own def function and begin with the word test example `test_arange`. This allows the `mojo test` command to find it. A single function can cover multiple similiar tests but they should have unique strings to identify which check failed.
