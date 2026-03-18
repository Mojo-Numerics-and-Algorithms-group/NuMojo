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

## Checklist before opening a PR

- `pixi run format`
- `pixi run test` (or the relevant subset)
- `pixi run package` if your change affects packaging or tests
