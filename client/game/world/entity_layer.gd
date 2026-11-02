class_name EntityLayer
extends Node2D
## Draws every entity in the EntityStore with a pooled EntitySprite per known id. Positions are interpolated from the
## previous to the newest tick position over interp_delay_ticks * tick_ms. Only the ids the store reports as changed
## or removed are reconfigured each frame; every visible node is still re-sampled for the interpolation.

var store: EntityStore = null
var own_id: int = -1
var tick_ms: float = ViewMath.DEFAULT_TICK_MS

var configured_count: int = 0  ## Sprites configured in the last refresh (for tests and the overlay).

var _pool: SlotPool = SlotPool.new()
var _sprites: Array[EntitySprite] = []
var _tracks: Dictionary = {}  ## id -> EntityTrack
var _synced_store: EntityStore = null
var _synced_version: int = -1
var _synced_tile_px: float = -1.0
var _synced_label_px: int = -1
var _synced_own_id: int = -2


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
	configured_count = 0
	# A different store, a missed snap (version jumped by more than one) or a look change needs every id again.
	var jumped: bool = store.version != _synced_version and store.version != _synced_version + 1
	var full: bool = (store != _synced_store or jumped
			or tile_px != _synced_tile_px or label_px != _synced_label_px or own_id != _synced_own_id)
	if full:
		_sync_all(now_ms, dur, tile_px, label_px)
	elif store.version != _synced_version:
		for id: int in store.removed_ids():
			if not store.has_ent(id):
				_drop(id)
		for id: int in store.changed_ids():
			_update(id, now_ms, dur, tile_px, label_px)
	_synced_store = store
	_synced_version = store.version
	_synced_tile_px = tile_px
	_synced_label_px = label_px
	_synced_own_id = own_id
	for id: int in _tracks:
		var track: EntityTrack = _tracks[id]
		_sprites[_pool.slot_of(id)].position = ViewMath.tile_center(track.sample(now_ms), tile_px)


func _sync_all(now_ms: float, dur: float, tile_px: float, label_px: int) -> void:
	var live: Dictionary = {}
	for id: int in store.ids():
		live[id] = true
		_update(id, now_ms, dur, tile_px, label_px)
	for id: int in _pool.ids():
		if not live.has(id):
			_drop(id)


func _update(id: int, now_ms: float, dur: float, tile_px: float, label_px: int) -> void:
	if not store.has_ent(id):
		return
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
	configured_count += 1


func _drop(id: int) -> void:
	var slot: int = _pool.release(id)
	if slot >= 0:
		_sprites[slot].visible = false
	_tracks.erase(id)


func _sprite_at(slot: int) -> EntitySprite:
	while _sprites.size() <= slot:
		var s := EntitySprite.new()
		s.visible = false
		add_child(s)
		_sprites.append(s)
	return _sprites[slot]
