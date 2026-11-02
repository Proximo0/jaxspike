class_name Overlay
extends CanvasLayer
## F1 overlay: real frame time p50/p95/max over Tuning.overlay_window_s, server tick, snaps/s, rx bytes/s,
## entity count, input-to-screen ms and recorder state.
## Frame times come from Time.get_ticks_usec() between frames, taken in a child node with the lowest
## process priority so it runs after every other _process. Performance.TIME_PROCESS is never used.
## Wiring (by main): `bind(session)` (a NetSession: .store.tick, .store.ids(), .store.you, and .rx_bytes for the
## rx rate; `note_rx(bytes)` is the manual alternative), `note_walk_to()` when a walk_to is sent, `recorder_state` set by the recorder owner.

const LAYER: int = 125
const FONT_SIZE: int = 15
const PLATE_COLOR := Color(0.05, 0.06, 0.09, 0.85)
const TEXT_COLOR := Color(0.92, 0.94, 0.98)
const REFRESH_US: int = 250000
const LOWEST_PRIORITY: int = 2147483647

var recorder_state: String = "rec off"

var _session: Object = null
var _label: Label = null
var _clock: OverlayFrameClock = null
var _last_us: int = 0
var _last_refresh_us: int = 0
var _times_ms: PackedFloat64Array = PackedFloat64Array()
var _time_stamps_us: PackedInt64Array = PackedInt64Array()
var _snap_stamps_us: PackedInt64Array = PackedInt64Array()
var _rx_stamps_us: PackedInt64Array = PackedInt64Array()
var _rx_bytes: PackedInt64Array = PackedInt64Array()
var _last_tick: int = -1
var _last_session_rx: int = -1
var _walk_sent_us: int = -1
var _walk_start_pos: Variant = null
var _input_ms: float = -1.0


## Inner node: ticks the overlay after all other nodes have processed.
class OverlayFrameClock extends Node:
	var owner_overlay: Overlay = null

	func _init() -> void:
		process_priority = Overlay.LOWEST_PRIORITY

	func _process(_delta: float) -> void:
		if owner_overlay != null:
			owner_overlay.frame_end(Time.get_ticks_usec())


func _init() -> void:
	layer = LAYER
	visible = false


func _ready() -> void:
	var plate := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = PLATE_COLOR
	style.set_content_margin_all(6.0)
	plate.add_theme_stylebox_override("panel", style)
	plate.position = Vector2(8.0, 8.0)
	plate.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(plate)
	_label = Label.new()
	_label.add_theme_font_size_override("font_size", FONT_SIZE)
	_label.add_theme_color_override("font_color", TEXT_COLOR)
	plate.add_child(_label)
	_clock = OverlayFrameClock.new()
	_clock.owner_overlay = self
	add_child(_clock)
	if Flags.args.has("overlay"):
		visible = true


func _unhandled_input(event: InputEvent) -> void:
	var key := event as InputEventKey
	if key != null and key.pressed and not key.echo and key.physical_keycode == KEY_F1:
		visible = not visible
		get_viewport().set_input_as_handled()


func bind(session: Object) -> void:
	_session = session


func note_rx(bytes: int) -> void:
	_rx_stamps_us.append(Time.get_ticks_usec())
	_rx_bytes.append(bytes)


## Call when a walk_to is sent; the latency is measured to the first change of the own position.
func note_walk_to() -> void:
	_walk_sent_us = Time.get_ticks_usec()
	_walk_start_pos = _you_pos()


## Last measured input-to-screen time in ms (-1 before the first one).
func input_to_screen_ms() -> float:
	return _input_ms


## Frame-time stats over the window: {p50, p95, max, n}.
func stats() -> Dictionary:
	return DebugFrame.frame_stats(_times_ms)


## Called once per frame by the lowest-priority clock node with the current time in microseconds.
func frame_end(now_us: int) -> void:
	if _last_us > 0:
		_times_ms.append(float(now_us - _last_us) / 1000.0)
		_time_stamps_us.append(now_us)
	_last_us = now_us
	var window_us: int = int(Tuning.shared().overlay_window_s * 1000000.0)
	var cutoff: int = now_us - window_us
	_track_session(now_us)
	var n: int = _count_old(_time_stamps_us, cutoff)
	if n > 0:
		_time_stamps_us = _time_stamps_us.slice(n)
		_times_ms = _times_ms.slice(n)
	n = _count_old(_rx_stamps_us, cutoff)
	if n > 0:
		_rx_stamps_us = _rx_stamps_us.slice(n)
		_rx_bytes = _rx_bytes.slice(n)
	n = _count_old(_snap_stamps_us, cutoff)
	if n > 0:
		_snap_stamps_us = _snap_stamps_us.slice(n)
	if visible and now_us - _last_refresh_us >= REFRESH_US:
		_last_refresh_us = now_us
		_label.text = _text(now_us, window_us)


func _track_session(now_us: int) -> void:
	if _session == null:
		return
	var store: Object = _session.get("store")
	if store == null:
		return
	var session_rx: Variant = _session.get("rx_bytes")
	if session_rx != null:
		if _last_session_rx >= 0 and int(session_rx) > _last_session_rx:
			note_rx(int(session_rx) - _last_session_rx)
		_last_session_rx = int(session_rx)
	var tick: int = int(store.get("tick"))
	if tick != _last_tick:
		if _last_tick >= 0:
			_snap_stamps_us.append(now_us)
		_last_tick = tick
	if _walk_sent_us >= 0 and _you_pos() != _walk_start_pos:
		_input_ms = float(now_us - _walk_sent_us) / 1000.0
		_walk_sent_us = -1


func _you_pos() -> Variant:
	if _session == null:
		return null
	var store: Object = _session.get("store")
	if store == null:
		return null
	var you: Variant = store.get("you")
	if you is Dictionary:
		return (you as Dictionary).get("pos")
	return null


static func _count_old(stamps: PackedInt64Array, cutoff: int) -> int:
	var n: int = 0
	while n < stamps.size() and stamps[n] < cutoff:
		n += 1
	return n


func _text(now_us: int, window_us: int) -> String:
	var s: Dictionary = stats()
	var win_s: float = float(window_us) / 1000000.0
	var snaps_s: float = float(_snap_stamps_us.size()) / win_s
	var rx_total: int = 0
	for b: int in _rx_bytes:
		rx_total += b
	var rx_s: float = float(rx_total) / win_s
	var ents: int = 0
	var tick: int = _last_tick
	if _session != null and _session.get("store") != null:
		ents = (_session.get("store").call("ids") as PackedInt32Array).size()
	var input_text: String = "n/a" if _input_ms < 0.0 else "%.0f ms" % _input_ms
	var lines: PackedStringArray = [
		"frame ms  p50 %.1f  p95 %.1f  max %.1f  (%.0f fps, %d frames, %.1f s)" % [
			s["p50"], s["p95"], s["max"], DebugFrame.fps_from_ms(float(s["p95"])), s["n"], win_s],
		"tick %d   snaps/s %.1f   rx %.0f B/s   ents %d" % [tick, snaps_s, rx_s, ents],
		"input-to-screen %s   %s" % [input_text, recorder_state],
	]
	return "\n".join(lines)
