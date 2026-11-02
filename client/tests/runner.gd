extends SceneTree
## Runs every tests/unit/test_*.gd (or one file via -- --only=test_name). Prints failures only, then one summary line.
## Exit code 0 = all pass, 1 = any failure or a script that would not load.


## Deferred so autoload singletons (Flags, Controls) exist before test scripts compile.
func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var only: String = ""
	for a: String in OS.get_cmdline_user_args():
		if a.begins_with("--only="):
			only = a.trim_prefix("--only=")
	var passed: int = 0
	var failed: int = 0
	var dir: PackedStringArray = DirAccess.get_files_at("res://tests/unit")
	for f: String in dir:
		if not (f.begins_with("test_") and f.ends_with(".gd")):
			continue
		if only != "" and f.get_basename() != only:
			continue
		var script: GDScript = load("res://tests/unit/" + f)
		if script == null or not script.can_instantiate():
			printerr("LOAD FAIL %s" % f)
			failed += 1
			continue
		for m: Dictionary in script.get_script_method_list():
			var name: String = m["name"]
			if not name.begins_with("test_"):
				continue
			var t: TestCase = script.new()
			t.call(name)
			if t.failures.is_empty():
				passed += 1
			else:
				failed += 1
				for msg: String in t.failures:
					printerr("FAIL %s::%s  %s" % [f.get_basename(), name, msg])
	print("TESTS passed=%d failed=%d" % [passed, failed])
	quit(1 if failed > 0 or passed == 0 else 0)
