class_name Tuning
extends RefCounted
## Template for live-tunable gameplay numbers (pattern from the skate project's PhysicsTuning).
## Game code reads `Tuning.shared().<field>` every tick instead of hard-coding numbers, so a tuning panel,
## replays, sims and presets can change the feel without touching code.
## Add a row to FIELDS and a matching `var` below; nothing else is needed.
## Presets are JSON dictionaries of {field: value}; unknown keys are ignored.

signal changed(field: String, value: float)

## [field, default, min, max, step, group, doc]. The order here is the order a tuning panel shows.
const FIELDS: Array = [
	["gravity", 18.0, 5.0, 40.0, 0.5, "World", "Gravity, m/s². Real is 9.8; higher feels snappier, less floaty."],
	["move_speed", 6.0, 1.0, 20.0, 0.25, "Movement", "Top move speed, m/s."],
]

static var _shared: Tuning = null

var gravity: float
var move_speed: float


func _init() -> void:
	reset()


## The tuning the game uses unless it is given its own. A tuning panel edits this one.
static func shared() -> Tuning:
	if _shared == null:
		_shared = Tuning.new()
	return _shared


static func field_names() -> PackedStringArray:
	var out := PackedStringArray()
	for f: Array in FIELDS:
		out.append(f[0])
	return out


static func field_info(field: String) -> Dictionary:
	for f: Array in FIELDS:
		if f[0] == field:
			return {"name": f[0], "default": f[1], "min": f[2], "max": f[3], "step": f[4], "group": f[5], "doc": f[6]}
	return {}


func reset() -> void:
	for f: Array in FIELDS:
		set(f[0], float(f[1]))


## Sets one field, clamped to its range. Returns false for an unknown field.
func set_value(field: String, value: float) -> bool:
	var info: Dictionary = field_info(field)
	if info.is_empty():
		return false
	var v: float = clampf(value, float(info["min"]), float(info["max"]))
	set(field, v)
	changed.emit(field, v)
	return true


func to_dict() -> Dictionary:
	var d: Dictionary = {}
	for f: Array in FIELDS:
		d[f[0]] = get(f[0])
	return d


## Applies known keys from `d` (others are ignored). Returns the number of fields applied.
func apply_dict(d: Dictionary) -> int:
	var n: int = 0
	for key: Variant in d:
		if set_value(str(key), float(d[key])):
			n += 1
	return n


## Only the fields that differ from the defaults: what a preset needs to store.
func diff_from_defaults() -> Dictionary:
	var d: Dictionary = {}
	for f: Array in FIELDS:
		if not is_equal_approx(float(get(f[0])), float(f[1])):
			d[f[0]] = get(f[0])
	return d


func copy_from(other: Tuning) -> void:
	apply_dict(other.to_dict())


func save_json(path: String) -> Error:
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		return FileAccess.get_open_error()
	f.store_string(JSON.stringify(to_dict(), "\t", false))
	return OK


## Loads a preset file over the current values. Returns false if the file is missing or not a JSON object.
func load_json(path: String) -> bool:
	if not FileAccess.file_exists(path):
		return false
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	if not parsed is Dictionary:
		return false
	apply_dict(parsed)
	return true
