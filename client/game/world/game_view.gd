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
var walk_marker: WalkMarker = null
var _marker_tick: int = -1
var ui_layer: CanvasLayer = null
var panels: Array[Node] = []  ## every RO panel: bind(session) / set_welcome(msg) / on_message(msg) where defined
var skill_bar: SkillBar = null
var target_ring: TargetRing = null
var fx_layer: FxLayer = null  ## cast circles, hit sparks, level-up and heal effects from the animation state
var audio: FeelAudio = null

## Placeholder layout on the 1920x1080 canvas: status left under the HUD, inventory right, others centre.
const PANEL_POS: Dictionary = {
	"ShopPanel": Vector2(1300, 250), "CardAlbum": Vector2(420, 120), "CheatPanel": Vector2(760, 160), "InventoryPanel": Vector2(1660, 250), "SkillPanel": Vector2(420, 140),
	"CraftPanel": Vector2(700, 140), "PartyPanel": Vector2(1000, 140), "ChatWindow": Vector2(8, 900),
}
const PANEL_SCENES: Array[String] = [
	"res://game/ui/shop_panel.tscn", "res://game/ui/card_album.tscn", "res://game/ui/cheat_panel.tscn", "res://game/ui/skill_bar.tscn", "res://game/ui/skill_panel.tscn",
	"res://game/ui/inventory_panel.tscn", "res://game/ui/craft_panel.tscn", "res://game/ui/party_panel.tscn",
	"res://game/ui/chat_window.tscn",
]


func _ready() -> void:
	# Characters walk behind trees and the fountain: the view y-sorts its children (MapRenderer's Props cells join
	# through nested y-sort with per-prop y_sort_origin; EntityLayer sprites sit at their feet). Art team, round 5.
	y_sort_enabled = true


func start(s: NetSession) -> void:
	session = s
	map_drawn = false
	entity_layer.store = s.store
	entity_layer.own_id = -1
	if fx_layer == null:
		fx_layer = FxLayer.new()
		fx_layer.z_index = 20  # above entities (z 2..10); damage numbers (z 50) are children of EntityLayer
		add_child(fx_layer)
		move_child(fx_layer, entity_layer.get_index() + 1)  # a sibling of EntityLayer: same coordinate space
	fx_layer.bind(s, entity_layer)
	player_input.session = s
	player_input.camera = camera
	if not player_input.quit_requested.is_connected(_on_quit):
		player_input.quit_requested.connect(_on_quit)
	if walk_marker == null:
		walk_marker = WalkMarker.new()
		add_child(walk_marker)
		player_input.walk_target.connect(_on_walk_target)
		player_input.walk_blocked.connect(_on_walk_blocked)
	walk_marker.hide_marker()
	walk_marker.map = s.map
	player_input.entity_layer = entity_layer
	hud.bind(s)
	_ensure_panels()
	for p: Node in panels:
		if p == skill_bar:
			skill_bar.bind(s, entity_layer)
		elif p.has_method("bind"):
			p.call("bind", s)
	if target_ring == null:
		target_ring = TargetRing.new()
		add_child(target_ring)
	target_ring.bind(entity_layer, skill_bar.targeting)
	if audio == null:
		audio = FeelAudio.new()
		add_child(audio)
	audio.bind(s)
	player_input.click_filter = Callable(self, "_click_world")


func _process(_delta: float) -> void:
	if session == null:
		return
	for msg: Dictionary in session.poll():
		for p: Node in panels:
			if p.has_method("on_message"):
				p.call("on_message", msg)
		match String(msg["t"]):
			"welcome":
				_on_welcome()
				for p: Node in panels:
					if p.has_method("set_welcome"):
						p.call("set_welcome", msg)
				if fx_layer != null:
					fx_layer.set_welcome(msg)  # ro6-fx2: skill elements for the bolt tint
			"error":
				_close(session.last_error)
				return
	if session.welcomed and session.link.state() == NetLink.STATE_CLOSED:
		_close("Disconnected.")
		return
	_follow()
	_track_marker()


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
	var p: Vector2 = entity_layer.world_pos_of(session.slot)
	if p == Vector2.INF and not session.store.you.is_empty():
		p = ViewMath.tile_center(ViewMath.to_vec(session.store.you.get("pos")), tile_px)
	if p != Vector2.INF:
		camera.position = p


func _tile_px() -> float:
	return ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)


func _own_tile() -> Vector2i:
	var p: Variant = session.store.you.get("pos")
	if p is Vector2i:
		return p
	return Vector2i.ZERO


func _on_walk_target(tile: Vector2i) -> void:
	walk_marker.show_target(tile, _own_tile(), _tile_px())


func _on_walk_blocked(tile: Vector2i) -> void:
	walk_marker.flash_blocked(tile, _tile_px())


func _track_marker() -> void:
	if session.store.tick != _marker_tick:
		_marker_tick = session.store.tick
		walk_marker.observe(_own_tile(), _marker_tick)


## Instances the RO panels once, under a CanvasLayer above the HUD; keys live inside the panels.
func _ensure_panels() -> void:
	if ui_layer != null:
		return
	ui_layer = CanvasLayer.new()
	ui_layer.layer = 2
	add_child(ui_layer)
	for path: String in PANEL_SCENES:
		var node: Node = (load(path) as PackedScene).instantiate()
		ui_layer.add_child(node)
		panels.append(node)
		if node is SkillBar:
			skill_bar = node as SkillBar
		var c: Control = node as Control
		if c != null and PANEL_POS.has(c.name):
			c.position = PANEL_POS[c.name]
	var cast_bar: Control = Control.new()
	cast_bar.set_script(load("res://game/ui/cast_bar.gd"))
	cast_bar.name = "CastBar"
	cast_bar.set_anchors_preset(Control.PRESET_CENTER_BOTTOM)
	cast_bar.offset_top = -130.0
	ui_layer.add_child(cast_bar)
	panels.append(cast_bar)
	var buff_row: HBoxContainer = HBoxContainer.new()
	buff_row.set_script(load("res://game/ui/buff_row.gd"))
	buff_row.name = "BuffRow"
	buff_row.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	buff_row.offset_left = -400.0
	buff_row.offset_top = 250.0  # under the minimap
	ui_layer.add_child(buff_row)
	panels.append(buff_row)
	panels.append(hud)  # the HUD gets set_welcome / on_message like a panel (client-feel)


## Left click in the world: entity select wins over walk-to (the skill bar owns the selection).
func _click_world(world_pos: Vector2) -> bool:
	var picked: bool = skill_bar != null and skill_bar.click_world(world_pos)
	if picked:
		hud.on_target_selected(skill_bar.targeting.selected_id)  # ro5-pvp2: a player becomes the duel target
	return picked


func _on_quit() -> void:
	_close("")


func _close(reason: String) -> void:
	if session == null:
		return
	session.close()
	session = null
	player_input.session = null
	closed.emit(reason)
