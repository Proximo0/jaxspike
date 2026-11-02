extends Node
## Input actions, registered in code so they live in one readable place. Keep this minimal and add actions
## per game. `debug_stamp` is required by game/debug/frame_stamp.gd.

## action -> [keyboard keys]
const BUTTONS: Dictionary = {
	"debug_stamp": [KEY_F10],
	"pause": [KEY_ESCAPE],
}


func _ready() -> void:
	for action: String in BUTTONS:
		if not InputMap.has_action(action):
			InputMap.add_action(action)
		for k: Key in BUTTONS[action]:
			var e := InputEventKey.new()
			e.physical_keycode = k
			InputMap.action_add_event(action, e)
