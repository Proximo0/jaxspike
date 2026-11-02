extends SceneTree
## Writes res://game/ui/theme/ro_theme.tres. Run: godot --headless --path . -s res://game/ui/theme/build_theme.gd

const OUT_PATH: String = "res://game/ui/theme/ro_theme.tres"


## Deferred so autoloads exist before game classes compile.
func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var builder: GDScript = load("res://game/ui/theme/ro_theme_builder.gd")
	var err: int = builder.call("save", OUT_PATH)
	if err != OK:
		printerr("build_theme: save failed (%d)" % err)
	quit(0 if err == OK else 1)
