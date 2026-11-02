extends SceneTree
# Parse check, run by tools/lint.sh. Args after "--" are res:// script paths.
# A script that fails to compile loads as null or cannot instantiate; prints one "PARSE_FAIL <path>" line per such script.
# Engine parse errors are printed by Godot itself; lint.sh picks them up.
# Runs deferred so autoloads (Flags, Controls) exist when scripts resolve them.


func _init() -> void:
	call_deferred("_check")


func _check() -> void:
	var failures: int = 0
	for arg: String in OS.get_cmdline_user_args():
		var script: GDScript = ResourceLoader.load(arg, "GDScript", ResourceLoader.CACHE_MODE_IGNORE) as GDScript
		if script == null or not script.can_instantiate():
			failures += 1
			print("PARSE_FAIL %s" % arg)
	quit(1 if failures > 0 else 0)
