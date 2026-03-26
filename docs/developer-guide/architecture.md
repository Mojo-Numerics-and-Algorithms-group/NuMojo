# Architecture Guide

This document explains the high-level layout of NuMojo and how the main modules work together.

## Overview

NuMojo is organized around three core areas:

- `core/`: foundational data structures and utilities
- `routines/`: user-facing APIs grouped by domain
- `routines/operations/`: backend execution helpers and SIMD kernels

## Directory layout

### `numojo/core/`
Defines the fundamental types and utilities used throughout the library, including:

- `NDArray` and related memory/layout helpers
- `Matrix` and matrix-specific helpers
- shape utilities and dtype plumbing

This layer is responsible for storage, indexing, and basic array semantics.

### `numojo/routines/`
Contains the public numerical APIs. Files and subpackages are grouped by domain, such as:

- `math/`, `linalg/`, `statistics/`, `logic/`, and more

These functions typically validate inputs and delegate the heavy lifting to the backend layer wherever applicable.

### `numojo/routines/operations/`
Provides the execution backend for element-wise and SIMD operations. This includes:

- `HostExecutor` helpers for unary/binary/ternary operations
- vectorized application logic for NDArrays

Most math routines call into this layer via `HostExecutor.apply_*` helpers.

## Related docs

- `ndarray-basic-structure.md`
- `style-guide.md`
