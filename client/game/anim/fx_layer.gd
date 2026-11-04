class_name FxLayer
extends Node2D
## Draws the spell and hit effects (cast circle, hit spark, level-up, heal) above the entities. A pooled set of 64
## Sprite2D (sheet row of 128 px frames, hframes) is driven by FxLogic; positions come from the EntityLayer's
## drawn sprites, so the interpolation is never repeated here. Place it after the entity layer (or z_index 20+).

const FX_JSON: String = "res://assets/anim/fx/fx.json"
const FRAME_PX: float = 128.0
const SCALE_BASE: float = 0.5  ## fx.json "scale": the sheets are drawn at half size at a 32 px tile
## y offset of the anchor from the tile centre, in tiles: feet for circles and level-up, body for sparks and heals.
const FEET_TILES: float = 0.28
const BODY_TILES: float = -0.2

var logic: FxLogic = FxLogic.new()
var auto_refresh: bool = true  ## _process drives it; tests call refresh(delta_ms) themselves

var _store: EntityStore = null
var _entities: EntityLayer = null
var _anchors: Array[Vector2] = []
var _sprites: Array[Sprite2D] = []
var _tex: Array[Texture2D] = []
var _loaded: bool = false


## Binds the session's store and the entity layer whose drawn positions and attack timing are reused.
func bind(session: NetSession, entity_layer: EntityLayer) -> void:
	bind_store(session.store, entity_layer)


func bind_store(store: EntityStore, entity_layer: EntityLayer) -> void:
	_store = store
	_entities = entity_layer
	_load()


func _process(delta: float) -> void:
	if auto_refresh:
		refresh(delta * 1000.0)


## One frame: reads new snaps, ages the effects by `delta_ms` and poses the sprites.
func refresh(delta_ms: float) -> void:
	if _store == null or _entities == null:
		return
	var render: float = _entities.last_render_ms
	logic.observe(_store, _entities.own_id, _hit_delay.bind(render))
	logic.advance(delta_ms)
	var tile_px: float = ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)
	var s: float = SCALE_BASE * tile_px / 32.0
	for i: int in FxLogic.MAX_LIVE:
		var sp: Sprite2D = _sprites[i]
		var frame: int = logic.frame_of(i)
		var kind: int = logic.kinds[i]
		if frame < 0 or _tex[kind] == null:
			sp.visible = false
			continue
		var at: Vector2 = _entities.world_pos_of(logic.targets[i])
		if at == Vector2.INF:
			at = logic.last_pos[i]
			if at == Vector2.INF:
				sp.visible = false
				continue
		else:
			logic.last_pos[i] = at
		var y_tiles: float = BODY_TILES if kind == FxLogic.HIT or kind == FxLogic.HEAL else FEET_TILES
		if sp.texture != _tex[kind]:
			sp.texture = _tex[kind]
			sp.hframes = int(logic.specs[kind]["frames"])
			sp.offset = -_anchors[kind]
		sp.scale = Vector2(s, s)
		sp.position = at + Vector2(0.0, y_tiles * tile_px)
		sp.frame = frame
		sp.visible = true


## Server ms a hit on `id` waits for the attacker's atk frame (EntityLayer.pop_time reused).
func _hit_delay(id: int, tick: int, render: float) -> float:
	return _entities.pop_time(id, tick, render) - render


func _load() -> void:
	if _loaded:
		return
	_loaded = true
	var effects: Dictionary = {}
	var f: FileAccess = FileAccess.open(FX_JSON, FileAccess.READ)
	if f != null:
		var parsed: Variant = JSON.parse_string(f.get_as_text())
		if typeof(parsed) == TYPE_DICTIONARY and typeof((parsed as Dictionary).get("effects")) == TYPE_DICTIONARY:
			effects = (parsed as Dictionary)["effects"]
	logic.configure(effects)
	var slots: AssetSlots = AssetSlots.load_default()
	_tex.clear()
	_anchors.clear()
	for i: int in FxLogic.KIND_NAMES.size():
		var fx_name: String = FxLogic.KIND_NAMES[i]
		var t: Texture2D = null
		if slots != null and slots.has("fx." + fx_name):
			var p: String = slots.path_of("fx." + fx_name)
			if ResourceLoader.exists(p):
				t = slots.texture("fx." + fx_name)
		_tex.append(t)
		var entry: Dictionary = effects.get(fx_name, {})
		var a: Array = entry.get("anchor", [FRAME_PX * 0.5, FRAME_PX * 0.5])
		_anchors.append(Vector2(float(a[0]), float(a[1])))
	for i: int in FxLogic.MAX_LIVE:
		var sp := Sprite2D.new()
		sp.centered = false
		sp.visible = false
		add_child(sp)
		_sprites.append(sp)
