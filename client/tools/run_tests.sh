#!/usr/bin/env bash
# Engine-free unit tests, headless. Usage: tools/run_tests.sh [test_name]
set -uo pipefail
cd "$(dirname "$0")/.."
GODOT="${GODOT:-godot}"
# Imports take turns: parallel agents share the .godot import cache.
exec 9>"/tmp/$(basename "$PWD")_import.lock"
flock 9
# Refresh the class_name cache so new classes resolve. Quiet unless it fails.
"$GODOT" --headless --path . --import >/tmp/godot_import.log 2>&1 || { cat /tmp/godot_import.log; exit 1; }
flock -u 9  # only the import needs the lock; runs after it can overlap
ARGS=()
[ $# -gt 0 ] && ARGS=(-- "--only=$1")
"$GODOT" --headless --path . -s res://tests/runner.gd "${ARGS[@]}" 2>&1 \
  | grep -vE '^(Godot Engine v|$|WARNING: .*audio|   at: )' 
exit "${PIPESTATUS[0]}"
