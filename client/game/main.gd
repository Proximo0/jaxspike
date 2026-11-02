extends Node
## Main: login screen -> NetSession over a WsLink -> GameView, plus the debug layer
## (F1 overlay, F2 tuning panel, F3 flight recorder, F4/F6-F9 render toggles, F10 frame stamp).
## Lead-owned. The login is sent only once the socket is open (WsLink drops sends before that).

const CONNECT_TIMEOUT_MS: int = 10000
const GAME_VIEW_SCENE: String = "res://game/world/game_view.tscn"
const OVERLAY_SCENE: String = "res://game/debug/overlay.tscn"
const TUNING_PANEL_SCENE: String = "res://game/debug/tuning_panel.tscn"

@onready var login: LoginScreen = $LoginScreen

var session: NetSession = null
var view: GameView = null
var overlay: Overlay = null
var recorder: FlightRecorder = null
var _pending_name: String = ""
var _connect_started_ms: int = 0


func _ready() -> void:
	add_child(FrameStamp.new())
	add_child(RenderToggles.new())
	overlay = (load(OVERLAY_SCENE) as PackedScene).instantiate() as Overlay
	add_child(overlay)
	add_child((load(TUNING_PANEL_SCENE) as PackedScene).instantiate())
	login.connect_requested.connect(_on_connect_requested)
	var url: String = Flags.get_arg("url", "")
	var player_name: String = Flags.get_arg("name", "")
	if url != "" and player_name != "":
		_on_connect_requested(player_name, url)


func _on_connect_requested(player_name: String, url: String) -> void:
	_drop_session()
	session = NetSession.new(WsLink.new())
	overlay.bind(session)
	session.connect_to(url)
	_pending_name = player_name
	_connect_started_ms = Time.get_ticks_msec()
	login.set_busy(true)


func _process(_delta: float) -> void:
	if session == null or _pending_name == "":
		return
	var state: int = session.link.state()
	if state == NetLink.STATE_OPEN:
		session.login(_pending_name)
		_pending_name = ""
		_open_view()
	elif state == NetLink.STATE_CLOSED or Time.get_ticks_msec() - _connect_started_ms > CONNECT_TIMEOUT_MS:
		_pending_name = ""
		_drop_session()
		login.show_error("could not connect")


func _open_view() -> void:
	view = (load(GAME_VIEW_SCENE) as PackedScene).instantiate() as GameView
	add_child(view)
	view.closed.connect(_on_view_closed)
	view.player_input.walk_sent.connect(overlay.note_walk_to)
	view.start(session)
	login.visible = false


func _on_view_closed(reason: String) -> void:
	if view != null:
		view.queue_free()
		view = null
	_drop_session()
	login.visible = true
	login.set_busy(false)
	if reason != "":
		login.show_error(reason)


func _drop_session() -> void:
	if session != null:
		session.close()
		session = null
	if recorder != null:
		recorder.stop()
		recorder = null
		overlay.recorder_state = "rec off"


func _unhandled_input(event: InputEvent) -> void:
	var key: InputEventKey = event as InputEventKey
	if key == null or not key.pressed or key.echo or key.physical_keycode != KEY_F3:
		return
	if session == null:
		return
	if recorder == null:
		recorder = FlightRecorder.new()
		DirAccess.make_dir_recursive_absolute("user://recordings")
		recorder.start("user://recordings/%d.jsonl" % Time.get_unix_time_from_system())
		session.recorder = recorder
		overlay.recorder_state = "rec on"
	else:
		recorder.stop()
		session.recorder = null
		recorder = null
		overlay.recorder_state = "rec off"
	get_viewport().set_input_as_handled()
