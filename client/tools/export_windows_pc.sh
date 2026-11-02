#!/usr/bin/env bash
# Windows build from the user's PC (Git Bash): uses the Godot 4.7.2 console exe and the export templates kept on
# D: (D:/godot/templates, the Windows templates extracted from the official .tpz), writes the exe and a dated zip to
# D:/jaxspike-builds. The preset's custom_template lines are set for the export and restored afterwards, so the
# committed export_presets.cfg stays machine-independent. Quiet: errors plus one summary line.
# Usage: tools/export_windows_pc.sh [--godot <console exe>] [--templates <dir>] [--out <dir>]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GODOT_EXE="${GODOT:-/c/Users/crp27/Downloads/Godot_4.7.2/Godot_v4.7.2-stable_win64_console.exe}"
TEMPLATES="/d/godot/templates"
OUT="/d/jaxspike-builds"
while [[ $# -gt 0 ]]; do
	case "$1" in
		--godot) GODOT_EXE="$2"; shift 2 ;;
		--templates) TEMPLATES="$2"; shift 2 ;;
		--out) OUT="$2"; shift 2 ;;
		*) echo "export_windows_pc: unknown argument $1" >&2; exit 2 ;;
	esac
done
fail() { echo "export_windows_pc: $1" >&2; exit 1; }
[[ -x "$GODOT_EXE" ]] || fail "godot console exe not found: $GODOT_EXE"
[[ -f "$TEMPLATES/windows_release_x86_64.exe" ]] || fail "release template missing in $TEMPLATES"
[[ -f "$TEMPLATES/windows_debug_x86_64.exe" ]] || fail "debug template missing in $TEMPLATES"
command -v python >/dev/null 2>&1 || fail "python not found on PATH (used to zip)"

win() { cygpath -m "$1"; }  # Godot on Windows wants D:/... paths
STAMP="$(date +%Y%m%d-%H%M)"
EXE_DIR="$OUT/windows"
mkdir -p "$EXE_DIR"
EXE_OUT="$EXE_DIR/Jaxspike.exe"
ZIP_OUT="$OUT/Jaxspike-windows-$STAMP.zip"
PRESET="$ROOT/export_presets.cfg"
BACKUP="$(mktemp)"
cp "$PRESET" "$BACKUP"
restore() { cp "$BACKUP" "$PRESET"; rm -f "$BACKUP"; }
trap restore EXIT

REL="$(win "$TEMPLATES/windows_release_x86_64.exe")"
DBG="$(win "$TEMPLATES/windows_debug_x86_64.exe")"
# Only the first preset (Windows Desktop) gets the custom templates: edit lines 1..48 of the file.
python - "$PRESET" "$REL" "$DBG" <<'EOF'
import sys
path, rel, dbg = sys.argv[1:4]
lines = open(path, encoding="utf-8").read().split("\n")
out, section = [], ""
for ln in lines:
    if ln.startswith("[preset."):
        section = ln
    if section == "[preset.0.options]" and ln.startswith("custom_template/release="):
        ln = 'custom_template/release="%s"' % rel
    elif section == "[preset.0.options]" and ln.startswith("custom_template/debug="):
        ln = 'custom_template/debug="%s"' % dbg
    out.append(ln)
open(path, "w", encoding="utf-8", newline="\n").write("\n".join(out))
EOF

cd "$ROOT"
"$GODOT_EXE" --headless --path . --import >/dev/null 2>&1 || fail "import failed"
if ! "$GODOT_EXE" --headless --path . --export-release "Windows Desktop" "$(win "$EXE_OUT")" >"$OUT/export.log" 2>&1; then
	tail -20 "$OUT/export.log" >&2
	fail "godot export failed (log: $OUT/export.log)"
fi
[[ -f "$EXE_OUT" ]] || { tail -20 "$OUT/export.log" >&2; fail "exe not produced at $EXE_OUT"; }
cat >"$EXE_DIR/README.txt" <<'EOT'
Jaxspike client (Windows). Start the server first, from the repository:
    venv\Scripts\python.exe -m net.server --host 0.0.0.0 --port 8765
Then run Jaxspike.exe, type a name (and a password to create an account), host and port, Connect.
Keys: WASD move, left click walk, Space attack, E eat, C status, K skills, I inventory, R craft, P party,
Enter chat, 1-9 skill bar, F1 overlay, F2 tuning, F3 flight recorder, F10 frame stamp, Esc back to login.
EOT
rm -f "$ZIP_OUT"
python - "$(win "$EXE_DIR")" "$(win "$ZIP_OUT")" <<'EOF2' || fail "zip not produced"
import os, sys, zipfile
src, dst = sys.argv[1:3]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(src):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.join("windows", os.path.relpath(full, src)))
EOF2
SIZE=$(du -h "$ZIP_OUT" | cut -f1)
echo "EXPORT_WINDOWS_PC PASS exe=$(win "$EXE_OUT") zip=$(win "$ZIP_OUT") size=$SIZE"
