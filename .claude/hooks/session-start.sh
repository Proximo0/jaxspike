#!/bin/bash
# Cloud session setup: Godot (pinned), render/export helpers and the project import, so tests, lint, sims
# and renders work the moment a session or agent starts. Idempotent; cloud sessions only.
set -euo pipefail
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi
GODOT_VERSION="4.7.2"
cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
ROOT="$PWD"

# Engine (JAX core + server) Python deps, CPU only in cloud sessions.
if ! python3 -c "import jax, websockets, pytest" >/dev/null 2>&1; then
  pip install -q -r requirements.txt >/tmp/pip_install.log 2>&1 || echo "session-start: pip install failed, see /tmp/pip_install.log" >&2
fi

# The Godot client lives in client/.
cd "$ROOT/client"

# Godot editor binary (headless-capable), pinned to the project's version.
if ! godot --version 2>/dev/null | grep -q "^${GODOT_VERSION}\.stable"; then
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/godot.zip" \
    "https://github.com/godotengine/godot-builds/releases/download/${GODOT_VERSION}-stable/Godot_v${GODOT_VERSION}-stable_linux.x86_64.zip"
  unzip -q "$tmp/godot.zip" -d "$tmp"
  install -m 755 "$tmp/Godot_v${GODOT_VERSION}-stable_linux.x86_64" /usr/local/bin/godot
  rm -rf "$tmp"
fi

# xvfb for renders, zip for the Windows export (export templates are fetched by tools/export_windows.sh on first use).
missing=()
command -v xvfb-run >/dev/null 2>&1 || missing+=(xvfb)
command -v zip >/dev/null 2>&1 || missing+=(zip)
command -v unzip >/dev/null 2>&1 || missing+=(unzip)
if [ ${#missing[@]} -gt 0 ]; then
  (apt-get update -qq && apt-get install -y -qq "${missing[@]}") >/dev/null 2>&1 || echo "session-start: could not apt-get ${missing[*]}" >&2
fi

# Import once so the class cache and resources are ready (the tools take the same lock).
(flock 9; godot --headless --path . --import >/tmp/godot_import.log 2>&1) 9>/tmp/$(basename "$PWD")_import.lock || {
  echo "session-start: godot import failed, see /tmp/godot_import.log" >&2; exit 1; }
