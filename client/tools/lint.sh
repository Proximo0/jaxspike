#!/usr/bin/env bash
# Parse check every .gd under game/, tests/, tools/, then the static lint (tools/lint.py).
# Usage: tools/lint.sh. Quiet: prints problems and one "LINT problems=<n>" line. Exit 1 on problems.
set -uo pipefail
cd "$(dirname "$0")/.."
GODOT="${GODOT:-godot}"
# Imports take turns: parallel agents share the .godot import cache.
exec 9>"/tmp/$(basename "$PWD")_import.lock"
flock 9
"$GODOT" --headless --path . --import >/tmp/godot_import.log 2>&1 || { cat /tmp/godot_import.log; exit 1; }
flock -u 9  # only the import needs the lock; runs after it can overlap

# Every script, except the parse checker itself (it is the running script).
RES=()
while IFS= read -r f; do
  [ "$f" = "tools/lint_parse.gd" ] && continue
  RES+=("res://$f")
done < <(find game tests tools -name '*.gd' | sort)

# Parse check: each script loads in a headless SceneTree. Engine errors share this output.
PARSE_OUT=$("$GODOT" --headless --path . -s res://tools/lint_parse.gd -- "${RES[@]}" 2>&1)
# One line per failing script: the engine's compile message and line when Godot gives one.
PARSE_LINES=$(printf '%s\n' "$PARSE_OUT" | awk '
  /^(SCRIPT )?ERROR: / { msg = $0; sub(/^(SCRIPT )?ERROR: /, "", msg); next }
  /at: GDScript::reload \(res:\/\//  && match($0, /res:\/\/[^ )]+:[0-9]+/) {
    loc = substr($0, RSTART + 6, RLENGTH - 6); if (!(loc in seen)) seen[loc] = msg; next }
  /^PARSE_FAIL / { f = $2; sub(/^res:\/\//, "", f); fails[++n] = f; next }
  END {
    for (i = 1; i <= n; i++) {
      found = ""
      for (loc in seen) { split(loc, p, ":"); if (p[1] == fails[i]) { found = loc " parse " seen[loc]; break } }
      print (found != "" ? found : fails[i] ":1 parse failed to compile")
    }
  }')
PARSE_COUNT=$(printf '%s\n' "$PARSE_OUT" | grep -c '^PARSE_FAIL ' || true)
[ -n "$PARSE_LINES" ] && printf '%s\n' "$PARSE_LINES"

# Static checks. Its own summary line is replaced by the combined total below.
PY_OUT=$(python3 tools/lint.py)
PY_RC=$?
printf '%s\n' "$PY_OUT" | grep -v '^LINT problems=' || true
PY_COUNT=$(printf '%s\n' "$PY_OUT" | sed -n 's/^LINT problems=//p')

TOTAL=$((PARSE_COUNT + PY_COUNT))
echo "LINT problems=$TOTAL"
[ "$TOTAL" -gt 0 ] && exit 1
exit "$PY_RC"
