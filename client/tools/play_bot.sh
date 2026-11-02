#!/usr/bin/env bash
# Starts a server on a free port, runs the integration bot against it headless, then stops that server only.
# Usage: tools/play_bot.sh [--seconds=30] [--shard=0] [--nshards=1]. Ends with the bot's "BOT score=..." line.
# Exit code 0 only when the bot printed no FAIL line. On failure the server log tail is printed.
set -uo pipefail
cd "$(dirname "$0")/.."
GODOT="${GODOT:-godot}"
PY="${PYTHON:-C:/Users/crp27/jaxspike/venv/Scripts/python.exe}"
PORT=$("$PY" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
export DEV_SERVER_LOG="/tmp/play_bot.$PORT.log"
line=$(tools/dev_server.sh "$PORT") || { echo "$line"; echo "BOT server FAIL could not start"; exit 1; }
PID=$(echo "$line" | sed -n 's/.* pid=\([0-9]*\).*/\1/p')
trap 'kill "$PID" 2>/dev/null' EXIT
SECS="--seconds=30"
ARGS=()
for a in "$@"; do
	case "$a" in
		--seconds=*) SECS="$a" ;;
		*) ARGS+=("$a") ;;
	esac
done
out=$(timeout 300 "$GODOT" --headless --fixed-fps 60 --path . res://tests/bot/bot.tscn -- \
	--url="ws://127.0.0.1:$PORT" "$SECS" "${ARGS[@]:-}" 2>&1 | grep -vE '^(Godot Engine v|$|WARNING: .*audio|   at: )')
echo "$out"
if echo "$out" | grep -q '^BOT .* FAIL '; then
	echo "--- server log tail ---"
	tail -20 "$DEV_SERVER_LOG"
	exit 1
fi
if ! echo "$out" | grep -q '^BOT score='; then
	echo "BOT run FAIL no summary line"
	tail -20 "$DEV_SERVER_LOG"
	exit 1
fi
exit 0
