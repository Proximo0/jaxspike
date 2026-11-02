#!/usr/bin/env bash
# Every check at once, in parallel: unit tests, sims, lint, and two optional hooks that are skipped when their
# scene does not exist: the play-bot (tests/bot/bot.tscn) and the lab (tests/lab/lab.tscn).
# JOBS (default: CPU count) caps how many Godot processes run together.
# Usage: tools/check_all.sh. Logs go to /tmp/check.<repo>.XXXXXX/<name>.log (path printed in the header); the last lines print one summary row per check.
# Exit 1 if any check fails.
#
# Hook contract: a bot/lab scene prints "BOT ..." / "LAB ..." lines, any line containing " FAIL " fails the check,
# and a scene that exits non-zero fails it too. Add shards by adding TASKS rows (see the BOT block).
set -uo pipefail
cd "$(dirname "$0")/.."
GODOT="${GODOT:-godot}"
JOBS="${JOBS:-$(nproc)}"
OUT=$(mktemp -d "/tmp/check.$(basename "$PWD").XXXXXX")  # per run, so two check_all runs never share logs
FILTER='^(Godot Engine v|$|WARNING: .*audio|   at: )'
start=$(date +%s)
(flock 9; "$GODOT" --headless --path . --import) 9>"/tmp/$(basename "$PWD")_import.lock" >"$OUT/import.log" 2>&1 || { cat "$OUT/import.log"; echo "CHECK_ALL FAIL (import)"; exit 1; }

# One command per line: name|command.
BOT_SHARDS=3
TASKS=()
HAVE_BOT=0; [[ -f tests/bot/bot.tscn ]] && HAVE_BOT=1
HAVE_LAB=0; [[ -f tests/lab/lab.tscn ]] && HAVE_LAB=1
if (( HAVE_BOT )); then
  for ((i = 0; i < BOT_SHARDS; i++)); do
    TASKS+=("bot$i|timeout 240 $GODOT --headless --fixed-fps 60 --path . res://tests/bot/bot.tscn -- --shard=$i --nshards=$BOT_SHARDS")
  done
fi
(( HAVE_LAB )) && TASKS+=("lab|timeout 600 $GODOT --headless --fixed-fps 60 --path . res://tests/lab/lab.tscn")
TASKS+=("sims|tools/run_sims.sh")
TASKS+=("unit|timeout 600 $GODOT --headless --path . -s res://tests/runner.gd")
TASKS+=("lint|tools/lint.sh")

running=0
for t in "${TASKS[@]}"; do
  name="${t%%|*}"; cmd="${t#*|}"
  while (( running + 1 > JOBS )); do wait -n; running=$((running - 1)); done
  ( eval "$cmd" 2>&1 | grep -vE "$FILTER" >"$OUT/$name.log"; echo "${PIPESTATUS[0]}" >"$OUT/$name.rc" ) &
  running=$((running + 1))
done
wait

rc=0
row() { printf '%-10s %s\n' "$1" "$2"; }
echo "=== check_all ($(( $(date +%s) - start )) s, $JOBS jobs, logs $OUT) ==="
u=$(grep -E '^TESTS passed=' "$OUT/unit.log" | tail -1); row unit "${u:-CRASH}"
[[ "$u" == *"failed=0"* ]] || rc=1
s=$(grep -E '^SIMS passed=' "$OUT/sims.log" | tail -1); row sims "${s:-CRASH}"
[[ "$s" == *"failed=0"* ]] || rc=1
grep -E '^SIM .* FAIL ' "$OUT/sims.log" | sed 's/^/    /'
# Optional hooks: pass = every shard exited 0 and printed no " FAIL " line.
hook() {  # hook <label> <log glob prefix> <line prefix>
  local label="$1" pre="$2" lp="$3" n=0 bad=0 f
  for f in "$OUT/$pre"*.log; do
    [[ -e "$f" ]] || continue
    n=$((n + 1))
    [[ "$(cat "${f%.log}.rc" 2>/dev/null)" == "0" ]] || bad=$((bad + 1))
    grep -q "^$lp .* FAIL " "$f" && bad=$((bad + 1))
  done
  row "$label" "shards=$n failed=$bad"
  grep -hE "^$lp .* FAIL " "$OUT/$pre"*.log 2>/dev/null | sed 's/^/    /'
  (( bad == 0 ))
}
if (( HAVE_BOT )); then hook bot bot BOT || rc=1; else row bot "skipped (no tests/bot/bot.tscn)"; fi
if (( HAVE_LAB )); then hook lab lab LAB || rc=1; else row lab "skipped (no tests/lab/lab.tscn)"; fi
lint=$(tail -1 "$OUT/lint.log"); row lint "$lint"; [[ "$lint" == *"problems=0"* ]] || rc=1
if (( rc == 0 )); then echo "CHECK_ALL PASS"; else echo "CHECK_ALL FAIL (see rows above)"; fi
exit $rc
