# Pre-PR Checks

This guide outlines the minimum checks you should run before opening a pull request.

## Install dependencies

```bash
pixi install
```

## Build the package

Creates `numojo.mojopkg` and copies it into `tests/`.

```bash
pixi run package
```

## Format the codebase

```bash
pixi run format
```

## Run tests

```bash
pixi run test
```

You can also run subsets:

```bash
pixi run test_core
pixi run test_routines
```

## Run the full validation suite

This runs formatting and tests together.

```bash
pixi run final
```

## Check documentation standards

Reports (without rewriting files) any missing/inconsistent headers, module
docstrings, `Exports` sections, section separators, or function
docstrings (parameters, args, returns, raises) that don't match the actual
function signature. See
[style-guide.md](style-guide.md) for the conventions it checks.

```bash
pixi run check_standards
```

## Checklist before opening a PR

- `pixi run format`
- `pixi run test` (or the relevant subset)
- `pixi run package` if your change affects packaging or tests
- `pixi run check_standards` if your change adds/edits docstrings, headers, or public functions
