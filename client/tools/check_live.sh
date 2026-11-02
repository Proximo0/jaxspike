#!/usr/bin/env bash
# Opt-in live gate (not part of check_all.sh): starts the Python server on a free port, runs the 3 bot shards against
# it in parallel, stops only that server, and prints one "CHECK_LIVE PASS|FAIL" line gated on the BOT lines.
# Usage: tools/check_live.sh [--seconds=30]. PYTHON defaults to ../venv/Scripts/python.exe on Windows, python3 elsewhere.
set -uo pipefail
cd "$(dirname "$0")/.."
GODOT="${GODOT:-godot}"
case "$(uname -s)" in
	MINGW*|MSYS*|CYGWIN*) PYTHON="${PYTHON:-$PWD/../venv/Scripts/python.exe}" ;;
	*) PYTHON="${PYTHON:-python3}" ;;
esac
export PYTHON
SECS="--seconds=30"
for a in "$@"; do
	case "$a" in --seconds=*) SECS="$a" ;; esac
done
NSHARDS=3
PORT=$((20000 + $$ % 1000))
OUT=$(mktemp -d "/tmp/check_live.XXXXXX")
export DEV_SERVER_LOG="$OUT/server.log"
line=$(tools/dev_server.sh "$PORT") || { echo "$line"; echo "CHECK_LIVE FAIL (server did not start)"; exit 1; }
PID=$(echo "$line" | sed -n 's/.* pid=\([0-9]*\).*/\1/p')
trap 'kill "$PID" 2>/dev/null' EXIT
for ((i = 0; i < NSHARDS; i++)); do
	( timeout 300 "$GODOT" --headless --fixed-fps 60 --path . res://tests/bot/bot.tscn -- \
		--url="ws://127.0.0.1:$PORT" "$SECS" --shard=$i --nshards=$NSHARDS 2>&1 \
		| grep -vE '^(Godot Engine v|$|WARNING: .*audio|   at: )' >"$OUT/bot$i.log"; echo "${PIPESTATUS[0]}" >"$OUT/bot$i.rc" ) &
done
wait
bad=0
for ((i = 0; i < NSHARDS; i++)); do
	grep -h '^BOT ' "$OUT/bot$i.log" | sed "s/^/shard$i /"
	[[ "$(cat "$OUT/bot$i.rc" 2>/dev/null)" == "0" ]] || bad=$((bad + 1))
	grep -q '^BOT score=' "$OUT/bot$i.log" || bad=$((bad + 1))
	grep -q '^BOT .* FAIL ' "$OUT/bot$i.log" && bad=$((bad + 1))
done
if (( bad == 0 )); then
	echo "CHECK_LIVE PASS port=$PORT shards=$NSHARDS"
else
	echo "--- server log tail ---"; tail -20 "$DEV_SERVER_LOG"
	echo "CHECK_LIVE FAIL problems=$bad port=$PORT logs=$OUT"
fi
(( bad == 0 ))
