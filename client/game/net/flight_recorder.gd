class_name FlightRecorder
extends RefCounted
## Writes every inbound and outbound net text to a JSONL file, one line {f, d: "in"/"out", m}.
## ReplayLink plays it back.

var path: String = ""
var lines_written: int = 0
var _file: FileAccess = null


func is_recording() -> bool:
	return _file != null


## Opens `p` for writing (truncating). Returns false when the file cannot be opened.
func start(p: String) -> bool:
	stop()
	_file = FileAccess.open(p, FileAccess.WRITE)
	if _file == null:
		return false
	path = p
	lines_written = 0
	return true


func stop() -> void:
	if _file != null:
		_file.close()
		_file = null


func record_in(frame: int, text: String) -> void:
	_write(frame, "in", text)


func record_out(frame: int, text: String) -> void:
	_write(frame, "out", text)


## Reads a recording: Array of {f: int, d: String, m: String}. Bad lines are skipped.
static func read_file(p: String) -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	var f: FileAccess = FileAccess.open(p, FileAccess.READ)
	if f == null:
		return out
	while not f.eof_reached():
		var line: String = f.get_line()
		if line.is_empty():
			continue
		var json := JSON.new()
		if json.parse(line) != OK or typeof(json.data) != TYPE_DICTIONARY:
			continue
		var d: Dictionary = json.data
		if not (d.has("f") and d.has("d") and d.has("m")):
			continue
		out.append({"f": int(d["f"]), "d": String(d["d"]), "m": String(d["m"])})
	f.close()
	return out


func _write(frame: int, dir: String, text: String) -> void:
	if _file == null:
		return
	_file.store_line(JSON.stringify({"f": frame, "d": dir, "m": text}))
	lines_written += 1
