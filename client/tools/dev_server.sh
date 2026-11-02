#!/usr/bin/env bash
# Starts the Python game server in the background and leaves it running.
# Usage: tools/dev_server.sh [port]   (default 8765). Prints "DEV_SERVER port=<n> pid=<n>"; stop it with kill <pid>.
# Server log: $DEV_SERVER_LOG (default /tmp/dev_server.<port>.log).
set -uo pipefail
cd "$(dirname "$0")/../.."
PORT="${1:-8765}"
PY="${PYTHON:-C:/Users/crp27/jaxspike/venv/Scripts/python.exe}"
LOG="${DEV_SERVER_LOG:-/tmp/dev_server.$PORT.log}"
"$PY" -m net.server --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
PID=$!
for ((i = 0; i < 300; i++)); do
	if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
		echo "DEV_SERVER port=$PORT pid=$PID log=$LOG"
		exit 0
	fi
	kill -0 "$PID" 2>/dev/null || { echo "DEV_SERVER FAIL server exited"; tail -20 "$LOG"; exit 1; }
	sleep 0.2
done
echo "DEV_SERVER FAIL port $PORT not accepting after 60 s"
tail -20 "$LOG"
kill "$PID" 2>/dev/null
exit 1
