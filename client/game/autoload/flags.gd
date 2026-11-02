extends Node
## Run-wide flags and the state hash.
## User args come after "--" on the command line: godot --path . -- --scripted --dev --seed=7

var args: Dictionary = {}
var frame: int = 0
var commands: int = 0  ## player commands issued; compare state hashes only on frames where this is 0
var _hash_providers: Array[Callable] = []


func _ready() -> void:
	for a: String in OS.get_cmdline_user_args():
		var kv: PackedStringArray = a.trim_prefix("--").split("=", true, 1)
		args[kv[0]] = kv[1] if kv.size() > 1 else "1"


func _physics_process(_delta: float) -> void:
	frame += 1


## Captures, tests and gates: no saves, no writes to player data.
func is_scripted_run() -> bool:
	return args.has("scripted") or args.has("capture") or DisplayServer.get_name() == "headless"


## Dev menu entries and cheats only with --dev.
func dev_menu() -> bool:
	return args.has("dev")


func get_arg(key: String, default: String = "") -> String:
	return args.get(key, default)


func seed_value() -> int:
	return int(get_arg("seed", "1"))


## Systems register a Callable returning an int that summarises their state.
func register_hash(provider: Callable) -> void:
	_hash_providers.append(provider)


func state_hash() -> int:
	var h: int = 1469598103934665603
	for p: Callable in _hash_providers:
		if p.is_valid():
			h = hash([h, p.call()])
	return h
