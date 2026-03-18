# Runnable Examples

This page contains small examples you can run locally with Pixi.

## Prerequisites

1. Clone the repository.
2. Install dependencies:

```bash
pixi install
```

## Quickstart example

Run the bundled example:

```bash
pixi run mojo run -I . examples/quickstart.mojo
```

## What this example demonstrates

- Creating NDArrays with random data
- Basic arithmetic and trigonometric operations
- Matrix multiplication

If you add your own examples under `examples/`, use the same command pattern:

```bash
pixi run mojo run -I . examples/your_example.mojo
```
