#!/usr/bin/env bash
# Regression sims, headless. Usage: tools/run_sims.sh
# Runs tests/sim/sim_suite.tscn (sim_*.gd). Prints one "SIM <name> PASS|FAIL <detail>" line per scenario (INFO = report only)
# and one "SIMS passed=N failed=M total=T". Exit 1 on any failure or if the scene dies before printing its totals.
set -uo pipefail
cd "$(dirname "$0")/.."
GODOT="${GODOT:-godot}"
# Only the import needs the shared lock; the scene runs take none.
(flock 9; "$GODOT" --headless --path . --import) 9>"/tmp/$(basename "$PWD")_import.lock" >/tmp/godot_import.log 2>&1 || { cat /tmp/godot_import.log; exit 1; }
out=$(timeout 600 "$GODOT" --headless --path . res://tests/sim/sim_suite.tscn 2>&1 \
  | grep -vE '^(Godot Engine v|$|WARNING: .*audio|   at: )')
echo "$out" | grep -v '^SIMS passed='
totals=$(echo "$out" | grep '^SIMS passed=' | tail -1)
if [[ -z "$totals" ]]; then
  echo "SIM sim_suite FAIL scene ended without printing its SIMS totals (crash, script error or timeout)"
  echo "SIMS passed=0 failed=1 total=1"
  exit 1
fi
p=${totals#SIMS passed=}; p=${p%% *}
f=${totals##*failed=}
echo "SIMS passed=$p failed=$f total=$((p + f))"
(( f > 0 )) && exit 1
exit 0
