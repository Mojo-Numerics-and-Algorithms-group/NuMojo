#!/bin/bash
set -e  # Exit immediately if any command fails

for f in tests/core/*.mojo; do
    pixi run mojo run -I tests/ "$f"
done

# Run test_matrix.mojo again with F_CONTIGUOUS flag
pixi run mojo run -I tests/ -D F_CONTIGUOUS tests/core/test_matrix.mojo
