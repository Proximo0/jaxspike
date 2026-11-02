class_name NetSession
extends RefCounted
## One client connection: a NetLink plus the state decoded from it. Callers poll(); nothing is signal-driven.

var link: NetLink
var store: EntityStore = EntityStore.new()
var map: MapData = MapData.new()
var recorder: FlightRecorder = null  ## optional; set it and start() it to log traffic

var slot: int = -1
var tick_ms: int = 0
var view_w: int = 0
var view_h: int = 0
var welcomed: bool = false
var last_error: String = ""
var frame: int = 0  ## poll counter, used as the recorder frame number
var rx_bytes: int = 0
var bad_messages: int = 0


func _init(l: NetLink = null) -> void:
	link = l if l != null else FakeLink.new()


func connect_to(url: String) -> void:
	link.connect_to(url)


func close() -> void:
	link.close()
	welcomed = false


func login(player_name: String) -> void:
	_send({"t": "login", "name": player_name})


func send_input(move: int, attack: bool, eat: bool) -> void:
	_send({"t": "input", "move": move, "attack": attack, "eat": eat})


func walk_to(x: int, y: int) -> void:
	_send({"t": "walk_to", "x": x, "y": y})


## Drains the link, decodes each message, applies welcome/snap/error, and returns the decoded messages in order.
func poll() -> Array[Dictionary]:
	frame += 1
	var out: Array[Dictionary] = []
	for text: String in link.poll():
		rx_bytes += text.length()
		if recorder != null:
			recorder.record_in(frame, text)
		var msg: Dictionary = Codec.decode(text)
		if msg.is_empty():
			bad_messages += 1
			continue
		_apply(msg)
		out.append(msg)
	return out


func _apply(msg: Dictionary) -> void:
	match String(msg["t"]):
		"welcome":
			slot = int(msg.get("slot", -1))
			tick_ms = int(msg.get("tick_ms", 0))
			view_w = int(msg.get("view_w", 0))
			view_h = int(msg.get("view_h", 0))
			welcomed = map.from_welcome(msg)
			if not welcomed:
				last_error = "bad welcome map"
		"snap":
			store.apply_snap(msg)
		"error":
			last_error = String(msg.get("msg", "error"))


## Sends any client message; `v` and `t` are the caller's responsibility for `t` (Codec stamps `v`).
func send(msg: Dictionary) -> void:
	_send(msg)


func _send(msg: Dictionary) -> void:
	var text: String = Codec.encode(msg)
	if recorder != null:
		recorder.record_out(frame, text)
	link.send(text)
