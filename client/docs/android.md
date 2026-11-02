# Android build and touch

`tools/export_android.sh` exports the "Android" preset to `build/<slug>-android.apk`: a debug-signed APK
(arm64-v8a and armeabi-v7a, Godot's non-Gradle path, landscape, immersive, INTERNET only). It is quiet: errors plus
one `export_android: ok <path> (<size>)` line. Safe to re-run. Proven on Sandblood.

## Build
```
tools/export_android.sh
```
First run sets up, under `$HOME` only (nothing system-wide):
- Godot 4.7.2 Android export templates in `~/.local/share/godot/export_templates/4.7.2.stable` (the tpz is cached in
  `~/.cache/godot-templates`, shared with `tools/export_windows.sh`).
- JDK 17 in `~/jdk-17` and an SDK in `~/android-sdk` (`build-tools/35.0.1`: apksigner, aapt, aapt2, zipalign;
  `platform-tools/adb`). `dl.google.com` is blocked in the cloud sandbox, so on Debian/Ubuntu these are unpacked from
  distro packages with `apt-get download` (cached in `~/.cache/android-debs`; run `apt-get update` once if the fetch
  fails). On a machine with the real Android SDK (cmdline-tools, build-tools 35.0.1, platform android-35), set
  `ANDROID_SDK_ROOT` and the script uses it instead. Gradle builds are not used, so no NDK or platform is needed.
- `~/.android/debug.keystore` (the standard public debug key: alias `androiddebugkey`), made with keytool if missing.
- Godot editor settings (`~/.config/godot/editor_settings-4.7.tres`): SDK path, JDK path, debug keystore.

Project settings the export needs (already in the kit's `project.godot`):
- `[rendering] textures/vram_compression/import_etc2_astc=true` (export refuses without it).
- `renderer/rendering_method.mobile="gl_compatibility"`.
- Optional: `[display] window/handheld/orientation=4` (sensor landscape) for a game that turns with the phone.

Per project, in `export_presets.cfg` (Android preset): set `package/unique_name` (the scaffolder writes
`com.example.<name>`; pick the real one before any store build, it cannot change after publishing) and the launcher
icons (192x192 main, 432x432 adaptive foreground and background).

## Install on a phone
- USB: enable Developer options and USB debugging, then `~/android-sdk/platform-tools/adb install -r build/<slug>-android.apk`.
- Without adb: copy the APK to the phone (cable, cloud drive, `adb push`), open it in a file manager and allow
  "Install unknown apps" for that app when asked.
- Debug builds are signed with the shared debug key; a build signed with another key must be uninstalled first.

## Touch controls (pattern)
- One input struct per player (`move`, `aim` as ints -1000..1000, a button bit mask). Mouse/keyboard, gamepad and touch
  all fill the same struct, so nothing downstream knows the device, and online input packing is unchanged.
- A floating virtual thumbstick anywhere on the left half moves; a drag on the right half aims (offset from where the
  drag started, springs back to centre on release) and can hold the main action while the finger is down; a few
  round buttons for the rest.
- Size everything from the screen's short side (a layout unit = 10% of it), with pixel floors (64 px buttons at the
  least) and an edge gap; keep the pure mapping in static functions with unit tests.
- Let the mouse stand in for one finger on a desktop (`--touch`), so touch layouts can be tested and captured on the
  cloud box.
- Default to the low quality level on `OS.has_feature("mobile")` (see `docs/performance.md`).

## Online-service credentials
An APK has no folder beside it, so a git-ignored `config/*credentials*` file must go into the pack: add it to the
Android preset's `include_filter`. The script warns when it is missing. See `docs/online.md`.

## Not covered yet
- Play Store: release signing with an upload keystore (kept out of git), `--export-release`, and an AAB (needs the
  Gradle build template, the full Android SDK with NDK, and the target SDK per current Play policy).
- Performance on real devices: profile on a phone, not on the cloud box (software GL).
