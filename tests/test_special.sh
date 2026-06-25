#!/bin/bash
set -e  # Exit immediately if any command fails

for f in tests/special/*.mojo; do
    pixi run mojo run -I tests/ "$f"
done
