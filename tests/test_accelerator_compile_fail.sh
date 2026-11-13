#!/bin/bash
# Each tests/accelerator/compile_fail/*.mojo program must fail to compile with
# the error named on its "# EXPECT-ERROR:" line. A program that compiles, or
# fails for another reason, fails this script.
set -u

status=0
out_dir=$(mktemp -d)
trap 'rm -rf "$out_dir"' EXIT

for f in tests/accelerator/compile_fail/*.mojo; do
    expected=$(sed -n 's/^# EXPECT-ERROR: //p' "$f" | head -n 1)
    if [ -z "$expected" ]; then
        echo "FAIL $f: missing '# EXPECT-ERROR:' line"
        status=1
        continue
    fi
    if output=$(pixi run mojo build -I tests/ "$f" -o "$out_dir/out" 2>&1); then
        echo "FAIL $f: compiled, expected error: $expected"
        status=1
    elif ! grep -qF -- "$expected" <<< "$output"; then
        echo "FAIL $f: expected error not found: $expected"
        echo "$output" | grep -m 3 "error"
        status=1
    else
        echo "PASS $f"
    fi
done

exit $status
