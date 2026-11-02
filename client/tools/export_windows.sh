#!/usr/bin/env bash
# Export the "Windows Desktop" preset (export_presets.cfg) and zip it to build/<slug>-windows.zip.
# Quiet: prints errors and one summary line. Exit 1 on failure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="Jaxspike"
PROJECT_SLUG="Jaxspike"  # no spaces: used for file names
GODOT_VERSION="4.7.2"
TEMPLATE_TAG="4.7.2-stable"
TEMPLATE_DIR="${HOME}/.local/share/godot/export_templates/${GODOT_VERSION}.stable"
TEMPLATE_EXE="${TEMPLATE_DIR}/windows_release_x86_64.exe"
TPZ_URL="https://github.com/godotengine/godot/releases/download/${TEMPLATE_TAG}/Godot_v${GODOT_VERSION}-stable_export_templates.tpz"
CACHE_DIR="${HOME}/.cache/godot-templates"
TPZ_PATH="${CACHE_DIR}/export_templates.tpz"
EXE_OUT="build/windows/${PROJECT_SLUG}.exe"
ZIP_OUT="build/${PROJECT_SLUG}-windows.zip"

fail() {
	echo "export_windows (${PROJECT_NAME}): $1" >&2
	exit 1
}

command -v godot >/dev/null 2>&1 || fail "godot not found on PATH"
command -v unzip >/dev/null 2>&1 || fail "unzip not found on PATH"
command -v zip >/dev/null 2>&1 || fail "zip not found on PATH"
command -v curl >/dev/null 2>&1 || fail "curl not found on PATH"

if [[ ! -f "${TEMPLATE_EXE}" ]]; then
	mkdir -p "${TEMPLATE_DIR}" "${CACHE_DIR}"
	if [[ ! -f "${TPZ_PATH}" ]]; then
		curl -sSfL --retry 3 -o "${TPZ_PATH}.part" "${TPZ_URL}" || fail "template download failed"
		mv "${TPZ_PATH}.part" "${TPZ_PATH}"
	fi
	# The tpz is a zip with a templates/ folder. Extract only the Windows templates.
	unzip -q -o -j "${TPZ_PATH}" 'templates/windows_*' -d "${TEMPLATE_DIR}" >/dev/null 2>&1 \
		|| fail "template extract failed"
	unzip -q -o -j "${TPZ_PATH}" 'templates/version.txt' -d "${TEMPLATE_DIR}" >/dev/null 2>&1 || true
	rm -f "${TPZ_PATH}"
	[[ -f "${TEMPLATE_EXE}" ]] || fail "windows_release_x86_64.exe missing after extract"
fi

cd "${ROOT}"
mkdir -p "$(dirname "${EXE_OUT}")"

if ! godot --headless --path . --export-release "Windows Desktop" "${EXE_OUT}" >/dev/null 2>&1; then
	fail "godot export failed (rerun with godot directly for details)"
fi
[[ -f "${EXE_OUT}" ]] || fail "exe not produced at ${EXE_OUT}"
# Optional mode launchers (test bypasses such as "unlock all") ship next to the exe, never in release code paths.
cp "${ROOT}"/tools/windows/*.bat build/windows/ 2>/dev/null || true
# Online-service credentials (docs/online.md), when the owner has them: next to the exe, where the game looks after
# res://config/. The git-ignored source files stay out of git.
rm -f build/windows/*credentials*
for c in "${ROOT}"/config/*credentials*; do
	if [[ -f "$c" ]]; then cp "$c" build/windows/; fi
done

rm -f "${ZIP_OUT}"
(cd build && zip -q -r "$(basename "${ZIP_OUT}")" windows) || fail "zip not produced"
[[ -f "${ZIP_OUT}" ]] || fail "zip not produced"

echo "export_windows: ok ${ZIP_OUT}"
