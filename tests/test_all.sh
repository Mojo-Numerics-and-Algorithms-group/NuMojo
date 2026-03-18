#!/bin/bash
set -e  # Exit immediately if any command fails

bash tests/test_core.sh
bash tests/test_routines.sh
