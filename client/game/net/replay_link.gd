class_name ReplayLink
extends NetLink
## Plays a FlightRecorder file back: inbound lines with f <= the current frame are returned by poll(); the frame
## advances once per tick(). Outbound lines in the file are ignored; whatever the app sends is kept in `sent`.

var frame: int = 0
var sent: Array[String] = []
var _lines: Array[Dictionary] = []
var _next: int = 0
var _state: int = NetLink.STATE_CLOSED


## Loads recorded lines (inbound only) from a file. Returns the number of inbound lines.
func load_file(path: String) -> int:
	return load_lines(FlightRecorder.read_file(path))


func load_lines(lines: Array[Dictionary]) -> int:
	_lines = []
	for l: Dictionary in lines:
		if String(l.get("d", "")) == "in":
			_lines.append(l)
	_next = 0
	frame = 0
	return _lines.size()


func finished() -> bool:
	return _next >= _lines.size()


func connect_to(_url: String) -> void:
	_state = NetLink.STATE_OPEN


func close() -> void:
	_state = NetLink.STATE_CLOSED


func state() -> int:
	return _state


func tick() -> void:
	frame += 1


func poll() -> Array[String]:
	var out: Array[String] = []
	if _state != NetLink.STATE_OPEN:
		return out
	while _next < _lines.size() and int(_lines[_next]["f"]) <= frame:
		out.append(String(_lines[_next]["m"]))
		_next += 1
	return out


func send(text: String) -> void:
	if _state == NetLink.STATE_OPEN:
		sent.append(text)
