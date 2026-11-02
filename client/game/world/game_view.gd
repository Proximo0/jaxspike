class_name GameView
extends Node2D
## The world scene: map, entities, camera on the own player, input and HUD, driven by one NetSession.
## `start(session)` once the session exists and login was sent; the view polls the session every frame.

signal closed(reason: String)

@onready var map_renderer: MapRenderer = $MapRenderer
@onready var entity_layer: EntityLayer = $EntityLayer
@onready var camera: Camera2D = $Camera2D
@onready var player_input: PlayerInput = $PlayerInput
@onready var hud: Hud = $Hud

var session: NetSession = null
var map_drawn: bool = false
var welcome_to_drawn_ms: int = 0


func start(s: NetSession) -> void:
	session = s
	map_drawn = false
	entity_layer.store = s.store
	entity_layer.own_id = -1
	player_input.session = s
	player_input.camera = camera
	if not player_input.quit_requested.is_connected(_on_quit):
		player_input.quit_requested.connect(_on_quit)
	hud.bind(s)


func _process(_delta: float) -> void:
	if session == null:
		return
	for msg: Dictionary in session.poll():
		match String(msg["t"]):
			"welcome":
				_on_welcome()
			"error":
				_close(session.last_error)
				return
	if session.welcomed and session.link.state() == NetLink.STATE_CLOSED:
		_close("Disconnected.")
		return
	_follow()


func _on_welcome() -> void:
	if not session.welcomed:
		_close(session.last_error)
		return
	var t0: int = Time.get_ticks_msec()
	map_renderer.build(session.map)
	map_drawn = true
	welcome_to_drawn_ms = Time.get_ticks_msec() - t0
	entity_layer.own_id = session.slot
	if session.tick_ms > 0:
		entity_layer.tick_ms = float(session.tick_ms)


func _follow() -> void:
	var tile_px: float = ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)
	var z: float = ViewMath.tuned("camera_zoom", ViewMath.DEFAULT_CAMERA_ZOOM)
	camera.zoom = Vector2(z, z)
	var p: Vector2 = entity_layer.world_pos_of(session.slot)
	if p == Vector2.INF and not session.store.you.is_empty():
		p = ViewMath.tile_center(ViewMath.to_vec(session.store.you.get("pos")), tile_px)
	if p != Vector2.INF:
		camera.position = p


func _on_quit() -> void:
	_close("")


func _close(reason: String) -> void:
	if session == null:
		return
	session.close()
	session = null
	player_input.session = null
	closed.emit(reason)
