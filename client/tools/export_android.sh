#!/usr/bin/env bash
# Export the "Android" preset (export_presets.cfg) to build/<slug>-android.apk (debug-signed, arm64-v8a + armeabi-v7a).
# Idempotent: installs the Android SDK pieces, JDK 17 and export templates under $HOME on first run only.
# Quiet: prints errors and one summary line. Exit 1 on failure. See docs/android.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="Jaxspike"
PROJECT_SLUG="Jaxspike"  # no spaces: used for file names
GODOT_VERSION="4.7.2"
TEMPLATE_TAG="4.7.2-stable"
TEMPLATE_DIR="${HOME}/.local/share/godot/export_templates/${GODOT_VERSION}.stable"
TPZ_URL="https://github.com/godotengine/godot/releases/download/${TEMPLATE_TAG}/Godot_v${GODOT_VERSION}-stable_export_templates.tpz"
CACHE_DIR="${HOME}/.cache/godot-templates"
TPZ_PATH="${CACHE_DIR}/export_templates.tpz"
SDK="${ANDROID_SDK_ROOT:-${HOME}/android-sdk}"
JDK="${JAVA_HOME_17:-${HOME}/jdk-17}"
BUILD_TOOLS_VER="35.0.1"
KEYSTORE="${HOME}/.android/debug.keystore"
SETTINGS="${HOME}/.config/godot/editor_settings-4.7.tres"
APK_OUT="build/${PROJECT_SLUG}-android.apk"
DEBS="${HOME}/.cache/android-debs"

fail() {
	echo "export_android (${PROJECT_NAME}): $1" >&2
	exit 1
}

for c in godot unzip curl python3; do
	command -v "$c" >/dev/null 2>&1 || fail "$c not found on PATH"
done

# 1. Export templates (keep the cached tpz; Windows script may have removed it, so redownload only if needed).
if [[ ! -f "${TEMPLATE_DIR}/android_release.apk" || ! -f "${TEMPLATE_DIR}/android_debug.apk" ]]; then
	mkdir -p "${TEMPLATE_DIR}" "${CACHE_DIR}"
	if [[ ! -f "${TPZ_PATH}" ]]; then
		curl -sSfL --retry 3 -o "${TPZ_PATH}.part" "${TPZ_URL}" || fail "template download failed"
		mv "${TPZ_PATH}.part" "${TPZ_PATH}"
	fi
	unzip -q -o -j "${TPZ_PATH}" 'templates/android_*' 'templates/version.txt' -d "${TEMPLATE_DIR}" \
		|| fail "template extract failed"
fi

# 2. JDK 17 and the SDK tools. dl.google.com is not reachable from the cloud sandbox, so on Debian/Ubuntu the
#    SDK is assembled from distro packages (apt-get download + dpkg-deb -x: nothing is installed system-wide).
#    On a machine with a real SDK, set ANDROID_SDK_ROOT to it and this step is skipped.
need_apt=0
[[ -x "${JDK}/bin/keytool" ]] || need_apt=1
[[ -x "${SDK}/build-tools/${BUILD_TOOLS_VER}/apksigner" && -x "${SDK}/platform-tools/adb" ]] || need_apt=1
if [[ "${need_apt}" == 1 ]]; then
	command -v apt-get >/dev/null 2>&1 || fail "no Android SDK at ${SDK} and no apt-get; install the SDK (docs/android.md)"
	command -v dpkg-deb >/dev/null 2>&1 || fail "dpkg-deb not found"
	mkdir -p "${DEBS}"
	X="$(mktemp -d)"
	trap 'rm -rf "${X}"' EXIT
	(
		cd "${DEBS}"
		apt-get download openjdk-17-jre-headless openjdk-17-jdk-headless apksigner libapksig-java adb aapt zipalign \
			android-liblog android-libbase android-libboringssl android-libcutils android-libziparchive \
			libprotobuf32t64 android-libbacktrace android-libutils android-libandroidfw android-libaapt \
			libzopfli1 >/dev/null 2>&1 || true
		for f in *.deb; do dpkg-deb -x "$f" "${X}"; done
	) || fail "package fetch failed (run 'apt-get update' once)"
	if [[ ! -x "${JDK}/bin/keytool" ]]; then
		mkdir -p "${JDK}"
		cp -a "${X}"/usr/lib/jvm/java-17-openjdk-amd64/. "${JDK}/"
		# The deb symlinks its config into /etc; replace the dangling ones with copies.
		while IFS= read -r l; do
			t="$(readlink "$l")"
			if [[ "$t" == /etc/* && -e "${X}${t}" ]]; then rm -f "$l"; cp -a "${X}${t}" "$l"; fi
		done < <(find "${JDK}" -xtype l)
	fi
	BT="${SDK}/build-tools/${BUILD_TOOLS_VER}"
	mkdir -p "${BT}/lib" "${SDK}/platform-tools" "${SDK}/native" "${SDK}/lib/android" "${SDK}/platforms/android-35"
	cp "${X}"/usr/share/java/apksigner-*.jar "${BT}/lib/apksigner.jar"
	cp "${X}"/usr/share/java/apksig-*.jar "${BT}/lib/apksig.jar"
	cp "${X}"/usr/bin/{aapt,aapt2,adb,zipalign} "${SDK}/native/"
	cp -a "${X}"/usr/lib/x86_64-linux-gnu/android/. "${SDK}/lib/android/"
	cp -a "${X}"/usr/lib/x86_64-linux-gnu/libprotobuf* "${X}"/usr/lib/libzopfli* "${SDK}/lib/" 2>/dev/null || true
	wrap() { # name, relative path from the wrapper's dir to the SDK root, dest dir
		printf '#!/usr/bin/env bash\nS="$(cd "$(dirname "${BASH_SOURCE[0]}")/%s" && pwd)"\nLD_LIBRARY_PATH="$S/lib/android:$S/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$S/native/%s" "$@"\n' "$2" "$1" >"$3/$1"
		chmod +x "$3/$1"
	}
	wrap adb .. "${SDK}/platform-tools"
	for t in aapt aapt2 zipalign; do wrap "$t" ../.. "${BT}"; done
	cat >"${BT}/apksigner" <<'SIGN'
#!/usr/bin/env bash
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib"
exec "${JAVA_HOME:+$JAVA_HOME/bin/}java" -Xmx1024M -cp "$D/apksigner.jar:$D/apksig.jar" com.android.apksigner.ApkSignerTool "$@"
SIGN
	chmod +x "${BT}/apksigner"
fi
[[ -x "${SDK}/build-tools/${BUILD_TOOLS_VER}/apksigner" ]] || fail "apksigner missing in ${SDK}/build-tools/${BUILD_TOOLS_VER}"
[[ -x "${SDK}/platform-tools/adb" ]] || fail "adb missing in ${SDK}/platform-tools"

# 3. Debug keystore.
if [[ ! -f "${KEYSTORE}" ]]; then
	mkdir -p "$(dirname "${KEYSTORE}")"
	"${JDK}/bin/keytool" -genkeypair -keystore "${KEYSTORE}" -storepass android -alias androiddebugkey -keypass android \
		-keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Android Debug,O=Android,C=US" >/dev/null 2>&1 \
		|| fail "keystore generation failed"
fi

# 4. Godot editor settings (headless export reads these).
mkdir -p "$(dirname "${SETTINGS}")"
[[ -f "${SETTINGS}" ]] || printf '[gd_resource type="EditorSettings" format=3]\n\n[resource]\n' >"${SETTINGS}"
python3 -I - "${SETTINGS}" "${SDK}" "${JDK}" "${KEYSTORE}" <<'PY' || fail "editor settings update failed"
import re, sys
path, sdk, jdk, ks = sys.argv[1:5]
want = {
	"export/android/android_sdk_path": sdk,
	"export/android/java_sdk_path": jdk,
	"export/android/debug_keystore": ks,
	"export/android/debug_keystore_user": "androiddebugkey",
	"export/android/debug_keystore_pass": "android",
}
text = open(path).read()
for k, v in want.items():
	line = '%s = "%s"' % (k, v)
	pat = re.compile(r"^%s = .*$" % re.escape(k), re.M)
	text = pat.sub(line, text) if pat.search(text) else text.rstrip("\n") + "\n" + line + "\n"
open(path, "w").write(text)
PY

# 5. Export.
cd "${ROOT}"
# Online-service credentials (docs/online.md), if the game has any: an APK has no folder beside it, so a
# git-ignored config/*credentials* file must go into the pack via the Android preset's include_filter.
for c in config/*credentials*; do
	[[ -f "$c" ]] || continue
	awk -v f="$c" '/^\[preset\./{a=0} /^name="Android"/{a=1} a && /^include_filter=/ && index($0, f){h=1} END{exit !h}' export_presets.cfg \
		|| echo "export_android: $c is not in the Android preset's include_filter; the APK will run without it" >&2
done
mkdir -p "$(dirname "${APK_OUT}")"
rm -f "${APK_OUT}" "${APK_OUT}.idsig"
LOG="$(mktemp)"
if ! godot --headless --path . --export-debug "Android" "${APK_OUT}" >"${LOG}" 2>&1; then
	tail -n 15 "${LOG}" >&2
	rm -f "${LOG}"
	fail "godot export failed"
fi
[[ -f "${APK_OUT}" ]] || { tail -n 15 "${LOG}" >&2; rm -f "${LOG}"; fail "apk not produced at ${APK_OUT}"; }
rm -f "${LOG}" "${APK_OUT}.idsig"
SIZE="$(du -h "${APK_OUT}" | cut -f1)"
echo "export_android: ok ${APK_OUT} (${SIZE})"
