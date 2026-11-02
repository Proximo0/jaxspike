class_name EntityLayer
extends Node2D
## Draws every entity in the EntityStore with a pooled EntitySprite per known id. Positions are interpolated from the
## previous to the newest tick position over interp_delay_ticks * tick_ms.

var store: EntityStore = null
var own_id: int = -1
var tick_ms: float = ViewMath.DEFAULT_TICK_MS

var _pool: SlotPool = SlotPool.new()
var _sprites: Array[EntitySprite] = []
var _tracks: Dictionary = {}  ## id -> EntityTrack


func active_count() -> int:
	return _pool.active_count()


func has_sprite(id: int) -> bool:
	return _pool.has_id(id)


func sprite_of(id: int) -> EntitySprite:
	var slot: int = _pool.slot_of(id)
	return _sprites[slot] if slot >= 0 else null


## Drawn world pixel position of an entity (tile centre), or Vector2.INF when unknown.
func world_pos_of(id: int) -> Vector2:
	var s: EntitySprite = sprite_of(id)
	return s.position if s != null else Vector2.INF


func _process(_delta: float) -> void:
	if store != null:
		refresh(Time.get_ticks_msec())


## Syncs the sprites with the store and positions them for time `now_ms`.
func refresh(now_ms: float) -> void:
	var tile_px: float = ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)
	var dur: float = ViewMath.tuned("interp_delay_ticks", ViewMath.DEFAULT_INTERP_DELAY_TICKS) * tick_ms
	var label_px: int = roundi(ViewMath.tuned("name_label_px", ViewMath.DEFAULT_NAME_LABEL_PX))
	var live: Dictionary = {}
	for id: int in store.ids():
		live[id] = true
		var e: Dictionary = store.get_ent(id)
		var slot: int = _pool.acquire(id)
		var sprite: EntitySprite = _sprite_at(slot)
		var track: EntityTrack = _tracks.get(id) as EntityTrack
		if track == null:
			track = EntityTrack.new()
			_tracks[id] = track
		track.set_target(ViewMath.to_vec(e.get("pos")), now_ms, dur)
		var kind: int = int(e.get("kind", 0))
		var label: String = String(e.get("name", "")) if kind == 0 else ""
		sprite.configure(kind, id == own_id, label, ViewMath.hp_fraction(int(e.get("hp", 0)), int(e.get("max_hp", 0))),
				tile_px, label_px)
		sprite.visible = true
		sprite.z_index = 10 if id == own_id else 2 - ViewMath.category(kind)
		sprite.position = ViewMath.tile_center(track.sample(now_ms), tile_px)
	for id: int in _pool.ids():
		if not live.has(id):
			var slot: int = _pool.release(id)
			_sprites[slot].visible = false
			_tracks.erase(id)


func _sprite_at(slot: int) -> EntitySprite:
	while _sprites.size() <= slot:
		var s := EntitySprite.new()
		s.visible = false
		add_child(s)
		_sprites.append(s)
	return _sprites[slot]
