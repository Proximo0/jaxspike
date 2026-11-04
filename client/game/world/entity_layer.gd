class_name EntityLayer
extends Node2D
## Draws every entity in the EntityStore with a pooled EntitySprite per known id (round2 contract 1.3).
## A RenderClock turns snap arrivals into a server-time clock delayed by interp_delay_ticks; each entity's
## EntityTrack interpolates cell to cell over its speed_ms and holds its anim state (a/d/t/m/s), and every frame
## the clip frame is picked (art-anim's ActSheet, else a baked PlaceholderClips sheet) with ClipTiming. Damage
## numbers (`h`) float up at the attacker's atk frame when the attacker is visible, else at once.
## Only the ids the store reports as changed or removed are re-fed each snap; every visible node is re-sampled
## each frame. Rows without anim fields still walk (motion-driven clips) and move (speed = tick interval).

const ANIM_SHEET_DIR: String = "res://assets/anim"
## The server keeps a dead entity visible for 3 s with `a = 7` before it sends `gone` (docs/anim_wire.md), so a
## track already showing DEAD is dropped at once. Fallback only: an animated monster removed with hp 0 that never
## showed DEAD plays its dead clip here and keeps the corpse this long.
const CORPSE_MS: float = 1200.0

var store: EntityStore = null
var own_id: int = -1
var tick_ms: float = ViewMath.DEFAULT_TICK_MS

var slots: AssetSlots = null  ## Art registry; loaded on first use. Slots that do not resolve keep the coloured quads.
var use_placeholders: bool = true  ## players and monsters animate, with PlaceholderClips sheets until art exists (else quads)
var auto_refresh: bool = true  ## refresh from _process with the wall clock (the lab drives a virtual clock instead)

var configured_count: int = 0  ## Sprites configured in the last refresh (for tests and the overlay).
var clock: RenderClock = RenderClock.new()
var pops: DamagePops = DamagePops.new()
var last_render_ms: float = 0.0  ## server ms drawn by the last refresh
var shown_pops: Array[Dictionary] = []  ## pops drawn by the last refresh ({id, dmg, seq, rise, alpha})

var _pool: SlotPool = SlotPool.new()
var _sprites: Array[EntitySprite] = []
var _tracks: Dictionary = {}  ## id -> EntityTrack
var _synced_store: EntityStore = null
var _synced_version: int = -1
var _synced_tile_px: float = -1.0
var _synced_label_px: int = -1
var _synced_own_id: int = -2
var _tex_cache: Dictionary = {}  ## slot key -> Texture2D, or null when the slot has no usable art
var _frames_cache: Dictionary = {}  ## sheet json path -> SpriteFrames or null
var _sheet_cache: Dictionary = {}  ## job, or -1 - monster kind -> ActSheet or null
var _cats: Dictionary = {}  ## id -> category
var _looks: Dictionary = {}  ## id -> look (job or monster art index)
var _hits: Array[Dictionary] = []  ## hits fed this snap, resolved once every row is in
var _last_pos: Dictionary = {}  ## id -> last drawn px (pops of removed targets stay put)
var _damage: DamageLayer = null
var _recs: Dictionary = {}  ## id -> LayerEntityRec
var _rec_list: Array[LayerEntityRec] = []  ## _recs values, rebuilt when ids come or go
var _recs_dirty: bool = false


## What the per-frame loop needs about one entity, without dictionary lookups.
class LayerEntityRec:
	var id: int = 0
	var track: EntityTrack = null
	var sprite: EntitySprite = null
	var cat: int = ViewMath.CAT_DROP
	var animated: bool = false
	var is_player: bool = false
	var look: int = -1
	var sheet: ActSheet = null
	var info_key: int = -1  ## action * 8 + face of the cached clip info
	var info: Dictionary = {}
	var follow_key: int = -1
	var follow_info: Dictionary = {}
	var hp: int = 1
	var last_miss_tick: int = -1  ## store tick of the last miss (`h: 0`) queued
	var corpse_until: float = -1.0  ## render ms the dead clip of a removed monster ends (-1 = alive)
	# Configure cache: _update calls EntitySprite.configure only when one of these differs.
	var c_valid: bool = false
	var c_kind: int = -1
	var c_hp: int = 0
	var c_max_hp: int = 0
	var c_name: Variant = null
	var c_item: Variant = null
	var c_job: int = 0
	var c_mkind: int = 0
	var c_own: bool = false
	var c_tile_px: float = -1.0
	var c_label_px: int = -1
	var c_animated: bool = false
	# Clip cache of the newest key, read once per key change instead of per frame.
	var i_frames: int = 1
	var i_delay: float = 100.0
	var i_atk: int = -1
	var i_name: String = "idle"
	var i_anim: StringName = &""
	var i_flip: bool = false
	var f_key: int = -1
	var f_frames: int = 1
	var f_delay: float = 100.0
	var f_name: String = "idle"
	var f_anim: StringName = &""
	var f_flip: bool = false
	# Last values handed to set_clip (l_key -1 = nothing handed yet or must be resent).
	var l_key: int = -1
	var l_frame: int = -1
	var l_flash: float = 0.0
	var alpha: float = 1.0  ## modulate alpha of the sprite (dead fade)


func active_count() -> int:
	return _pool.active_count()


func has_sprite(id: int) -> bool:
	return _pool.has_id(id)


func sprite_of(id: int) -> EntitySprite:
	var slot: int = _pool.slot_of(id)
	return _sprites[slot] if slot >= 0 else null


func track_of(id: int) -> EntityTrack:
	return _tracks.get(id) as EntityTrack


## Drawn world pixel position of an entity (tile centre), or Vector2.INF when unknown.
func world_pos_of(id: int) -> Vector2:
	var s: EntitySprite = sprite_of(id)
	return s.position if s != null else Vector2.INF


func _process(_delta: float) -> void:
	if store != null and auto_refresh:
		refresh(Time.get_ticks_msec())


## Syncs the sprites with the store and draws them for local time `now_ms`.
func refresh(now_ms: float) -> void:
	var tile_px: float = ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)
	var label_px: int = roundi(ViewMath.tuned("name_label_px", ViewMath.DEFAULT_NAME_LABEL_PX))
	clock.tick_ms = tick_ms
	clock.delay_ticks = ViewMath.tuned("interp_delay_ticks", ViewMath.DEFAULT_INTERP_DELAY_TICKS)
	configured_count = 0
	# A different store, a missed snap (version jumped by more than one) or a look change needs every id again.
	var jumped: bool = store.version != _synced_version and store.version != _synced_version + 1
	var full: bool = (store != _synced_store or jumped
			or tile_px != _synced_tile_px or label_px != _synced_label_px or own_id != _synced_own_id)
	if store.version != _synced_version and store.snaps > 0:
		clock.observe(store.tick, now_ms)
	var render: float = clock.render_ms(now_ms)
	if full:
		_sync_all(render, tile_px, label_px)
	elif store.version != _synced_version:
		for id: int in store.removed_ids():
			if not store.has_ent(id):
				_remove(id, render)
		for id: int in store.changed_ids():
			_update(id, render, tile_px, label_px)
	if not _hits.is_empty():
		_resolve_hits(render)
	_synced_store = store
	_synced_version = store.version
	_synced_tile_px = tile_px
	_synced_label_px = label_px
	_synced_own_id = own_id
	last_render_ms = render
	if _recs_dirty:
		_rec_list.clear()
		for r: LayerEntityRec in _recs.values():
			_rec_list.append(r)
		_recs_dirty = false
	for rec: LayerEntityRec in _rec_list:
		if rec.corpse_until >= 0.0 and render >= rec.corpse_until:
			_drop(rec.id)
			continue
		rec.sprite.position = ViewMath.tile_center(rec.track.pos_at(render), tile_px)
		if rec.animated:
			_animate(rec, render)
	_draw_pops(render, tile_px)


func _sync_all(render: float, tile_px: float, label_px: int) -> void:
	var live: Dictionary = {}
	for id: int in store.ids():
		live[id] = true
		_update(id, render, tile_px, label_px)
	for id: int in _pool.ids():
		if not live.has(id):
			_remove(id, render)


func _update(id: int, render: float, tile_px: float, label_px: int) -> void:
	if not store.has_ent(id):
		return
	var e: Dictionary = store.get_ent(id)
	var slot: int = _pool.acquire(id)
	var sprite: EntitySprite = _sprite_at(slot)
	var rec: LayerEntityRec = _recs.get(id) as LayerEntityRec
	if rec == null:
		rec = LayerEntityRec.new()
		rec.id = id
		rec.track = EntityTrack.new()
		sprite.modulate = Color.WHITE  # a pooled sprite may still carry a dead fade
		_recs[id] = rec
		_tracks[id] = rec.track
		_recs_dirty = true
	rec.sprite = sprite
	rec.corpse_until = -1.0
	var track: EntityTrack = rec.track
	var dmg: int = track.feed(e, store.tick, tick_ms, render, tick_ms)
	var kind: int = int(e.get("kind", 0))
	var cat: int = ViewMath.entity_category(kind, e.has("item"))
	_cats[id] = cat
	if dmg != 0:  # a negative h is a critical hit (docs/anim_wire.md)
		var crit: bool = int(e.get("c", 0)) == 1 or dmg < 0
		_hits.append({"id": id, "dmg": absi(dmg), "crit": crit, "tick": store.tick})
	elif e.has("h") and int(e["h"]) == 0 and store.tick > rec.last_miss_tick:
		rec.last_miss_tick = store.tick
		_hits.append({"id": id, "dmg": 0, "crit": false, "tick": store.tick})
	var hp: int = int(e.get("hp", 0))
	rec.hp = hp
	var animated: bool = (cat == ViewMath.CAT_PLAYER or cat == ViewMath.CAT_MONSTER) and use_placeholders
	var look: int = 0
	if cat == ViewMath.CAT_PLAYER:
		look = int(e.get("job", 0))
	elif cat == ViewMath.CAT_MONSTER:
		look = ViewMath.monster_art_index(kind, int(e.get("mkind", -1)))
	_looks[id] = look
	if animated and (look != rec.look or cat != rec.cat or not rec.animated):
		rec.sheet = entity_sheet(cat == ViewMath.CAT_PLAYER, look)
		rec.info_key = -1
		rec.follow_key = -1
		rec.l_key = -1
	rec.cat = cat
	rec.look = look
	rec.animated = animated
	rec.is_player = cat == ViewMath.CAT_PLAYER
	if not animated:
		sprite.clear_clip()
		rec.l_key = -1
	var max_hp: int = int(e.get("max_hp", 0))
	var nm: Variant = e.get("name", "")
	var item: Variant = e.get("item", 0)
	var job: int = int(e.get("job", 0))
	var mkind: int = int(e.get("mkind", -1))
	var own: bool = id == own_id
	if not (rec.c_valid and rec.c_kind == kind and rec.c_hp == hp and rec.c_max_hp == max_hp
			and rec.c_own == own and rec.c_tile_px == tile_px and rec.c_label_px == label_px
			and rec.c_animated == animated and rec.c_job == job and rec.c_mkind == mkind
			and typeof(rec.c_name) == typeof(nm) and rec.c_name == nm
			and typeof(rec.c_item) == typeof(item) and rec.c_item == item):
		rec.c_valid = true
		rec.c_kind = kind
		rec.c_hp = hp
		rec.c_max_hp = max_hp
		rec.c_own = own
		rec.c_tile_px = tile_px
		rec.c_label_px = label_px
		rec.c_animated = animated
		rec.c_job = job
		rec.c_mkind = mkind
		rec.c_name = nm
		rec.c_item = item
		var label: String = String(nm) if cat == ViewMath.CAT_PLAYER else ""
		var tex: Texture2D = null if animated else entity_texture(e, cat)
		sprite.configure(kind, own, label, ViewMath.hp_fraction(hp, max_hp),
				tile_px, label_px, tex, cat == ViewMath.CAT_NODE and ViewMath.node_depleted(hp), cat)
	sprite.visible = true
	sprite.z_index = 10 if id == own_id else 2 - mini(cat, ViewMath.CAT_DROP)
	configured_count += 1


## Picks and sets the clip frame of one animated entity at server time `render`.
func _animate(rec: LayerEntityRec, render: float) -> void:
	var c: Dictionary = rec.track.clip_at(render, tick_ms, tick_ms)  # pooled: read now, never keep
	var action: int = c["action"]
	var face: int = c["face"]
	var elapsed: float = c["elapsed_ms"]
	var dur: float = c["dur_ms"]
	var spd: float = c["speed_ms"]
	var key: int = action * 8 + face
	if key != rec.info_key:
		var ci: Dictionary = _clip_info(rec.sheet, action, face, rec.is_player)
		rec.info_key = key
		rec.i_frames = ci["frames"]
		rec.i_delay = ci["delay_ms"]
		rec.i_atk = atk_frame_for(rec.i_frames, ci["atk_frame"])
		rec.i_name = ci["name"]
		rec.i_anim = ci.get("anim", &"") as StringName
		rec.i_flip = ci.get("flip", false)
		rec.l_key = -1
	var frames: int = rec.i_frames
	var frame: int = ClipTiming.frame_at(action, elapsed, frames, rec.i_delay, dur, spd, rec.i_atk)
	var clip_name: String = rec.i_name
	var clip_anim: StringName = rec.i_anim
	var flip: bool = rec.i_flip
	var drawn: int = action
	if frame < 0:  # one-shot over: standby after an attack, idle after a flinch
		var total: float = ClipTiming.one_shot_ms(action, frames, rec.i_delay, dur)
		drawn = ClipTiming.follow_up(action)
		var fkey: int = drawn * 8 + face
		if fkey != rec.follow_key:
			var fi: Dictionary = _clip_info(rec.sheet, drawn, face, rec.is_player)
			rec.follow_key = fkey
			rec.f_frames = fi["frames"]
			rec.f_delay = fi["delay_ms"]
			rec.f_name = fi["name"]
			rec.f_anim = fi.get("anim", &"") as StringName
			rec.f_flip = fi.get("flip", false)
			rec.l_key = -1
		frames = rec.f_frames
		frame = ClipTiming.frame_at(drawn, elapsed - total, frames, rec.f_delay, 0.0, spd)
		clip_name = rec.f_name
		clip_anim = rec.f_anim
		flip = rec.f_flip
	var a: float = 1.0
	if drawn == ClipSelect.DEAD:
		a = DeadFade.alpha(elapsed, ClipTiming.one_shot_ms(drawn, rec.i_frames, rec.i_delay, dur))
	if a != rec.alpha:
		rec.alpha = a
		rec.sprite.modulate = Color(1.0, 1.0, 1.0, a)
	var flash: float = 0.0
	if drawn == ClipSelect.HURT:
		flash = floorf((1.0 - clampf(elapsed / ClipTiming.HURT_MS, 0.0, 1.0)) * 8.0) / 8.0
	frame = maxi(frame, 0)
	var skey: int = drawn * 8 + face
	if rec.l_key == skey and rec.l_frame == frame and rec.l_flash == flash:
		return
	rec.l_key = skey
	rec.l_frame = frame
	rec.l_flash = flash
	rec.sprite.set_clip(rec.sheet, rec.look, clip_name, clip_anim, frame, frames, face, flip, flash)


## {name, frames, delay_ms, atk_frame[, anim, flip]} for an action from the sheet, else the placeholder table.
func _clip_info(sheet: ActSheet, action: int, face: int, is_player: bool) -> Dictionary:
	if sheet != null:
		var r: Dictionary = sheet.resolve(action, face, is_player)
		if not r.is_empty():
			r["atk_frame"] = sheet.atk_frame
			return r
	return PlaceholderClips.clip(action, is_player)


## The attack clip's atk frame: the sheet's, else the frame at ClipTiming.ATK_FRACTION of the clip.
static func atk_frame_for(frames: int, sheet_atk: int) -> int:
	if frames <= 1:
		return -1
	if sheet_atk > 0 and sheet_atk < frames:
		return sheet_atk
	return clampi(roundi(float(frames) * ClipTiming.ATK_FRACTION), 1, frames - 1)


## Server ms the damage of a hit on `target_id` at `hit_tick` should show: the visible attacker's atk frame, else
## `render` (at once).
func pop_time(target_id: int, hit_tick: int, render: float) -> float:
	var target: EntityTrack = _tracks.get(target_id) as EntityTrack
	if target == null:
		return render
	var cands: Array[Dictionary] = []
	for id: int in _tracks:
		var t: EntityTrack = _tracks[id]
		if id == target_id or not t.has_anim or t.action != ClipSelect.ATTACK:
			continue
		cands.append({"id": id, "cell": t.cells.cell(), "start_ms": t.action_start_ms(tick_ms), "dur_ms": t.dur_ms})
	var hit_ms: float = float(hit_tick) * tick_ms
	var best: Dictionary = DamagePops.pick_attacker(target.cells.cell(), hit_ms, cands)
	if best.is_empty():
		return render
	var att: EntityTrack = _tracks[int(best["id"])]
	var is_player: bool = int(_cats.get(int(best["id"]), ViewMath.CAT_MONSTER)) == ViewMath.CAT_PLAYER
	var sheet: ActSheet = entity_sheet(is_player, int(_looks.get(int(best["id"]), 0)))
	var info: Dictionary = _clip_info(sheet, ClipSelect.ATTACK, att.face, is_player)
	var total: float = ClipTiming.one_shot_ms(ClipSelect.ATTACK, int(info["frames"]), float(info["delay_ms"]), att.dur_ms)
	var atk: int = atk_frame_for(int(info["frames"]), int(info["atk_frame"]))
	var at: float = att.action_start_ms(tick_ms) + (total * ClipTiming.ATK_FRACTION if atk > 0 else 0.0)
	return maxf(at, render)


func _resolve_hits(render: float) -> void:
	for h: Dictionary in _hits:
		var at: float = pop_time(int(h["id"]), int(h["tick"]), render)
		pops.add(int(h["id"]), int(h["dmg"]), at, bool(h.get("crit", false)))
		var target: EntityTrack = _tracks.get(int(h["id"])) as EntityTrack
		if target != null and target.action == ClipSelect.HURT:
			target.flinch_ms = at  # RO: the flinch lands with the number, at the attacker's atk frame
	_hits.clear()


func _draw_pops(render: float, tile_px: float) -> void:
	shown_pops = pops.active(render)
	if shown_pops.is_empty() and _damage == null:
		return
	if _damage == null:
		_damage = DamageLayer.new()
		_damage.z_index = 50
		add_child(_damage)
	var items: Array[Dictionary] = []
	var s: float = tile_px / 32.0
	for p: Dictionary in shown_pops:
		var id: int = int(p["id"])
		var base: Vector2
		var r: LayerEntityRec = _recs.get(id) as LayerEntityRec
		if r != null:
			base = r.sprite.position
		elif _last_pos.has(id):
			base = _last_pos[id]
		else:
			continue
		var head: float = tile_px * EntitySprite.FEET_Y - EntitySprite.HEAD_PX * s - tile_px * 0.3  # clear of the HP bar
		items.append({"pos": base + Vector2(0.0, head - float(p["rise"]) * tile_px), "dmg": p["dmg"],
				"text": "Miss" if int(p["dmg"]) == 0 else str(int(p["dmg"])),
				"crit": p.get("crit", false), "seq": p.get("seq", 0),
				"alpha": p["alpha"], "own": id == own_id})
	_damage.show_items(items, s)
	if _last_pos.size() > 64 and pops.pending() == 0:  # forget removed ids once no number needs them
		_last_pos.clear()


## Act sheet for a job / engine monster kind: art-anim's (the act JSON of slot `anim.<kind>.idle`, else
## res://assets/anim/<kind>/act.json, kind names from AssetSlots), else a baked PlaceholderClips sheet (null when
## placeholders are off: the quad is drawn).
func entity_sheet(is_player: bool, look: int) -> ActSheet:
	var cache_key: int = look if is_player else -1 - look
	if _sheet_cache.has(cache_key):
		return _sheet_cache[cache_key] as ActSheet
	if slots == null:
		slots = AssetSlots.load_default()
	var names: Array[String] = AssetSlots.JOB_NAMES if is_player else AssetSlots.ANIM_MONSTER_KINDS
	var sheet: ActSheet = null
	if look >= 0 and look < names.size():
		var key: String = AssetSlots.anim_key(names[look], "idle")
		var path: String = ANIM_SHEET_DIR.path_join(names[look]).path_join("act.json")
		if slots != null and slots.has(key) and slots.is_sheet(key):
			path = slots.path_of(key)
		sheet = ActSheet.load_act(path)
	if sheet == null and use_placeholders:
		sheet = PlaceholderClips.make_sheet(is_player, look)
	_sheet_cache[cache_key] = sheet
	return sheet


## Art for an entity, or null while its slot is missing (players job.N, monsters monster.x, drops item.x, nodes node.N).
func entity_texture(e: Dictionary, cat: int) -> Texture2D:
	if slots == null:
		slots = AssetSlots.load_default()
	var key: String = ""
	match cat:
		ViewMath.CAT_PLAYER:
			key = AssetSlots.job_key(int(e.get("job", 0)), "idle")
		ViewMath.CAT_MONSTER:
			var kind: int = int(e.get("kind", 1))
			key = AssetSlots.monster_key(ViewMath.monster_art_index(kind, int(e.get("mkind", -1))))
		ViewMath.CAT_DROP:
			key = AssetSlots.item_key(int(e.get("item", 0)))
		ViewMath.CAT_NODE:
			key = "node.%d" % (int(e.get("kind", 0)) - ViewMath.NODE_KIND_BASE)
	return slot_texture(key)


## Texture for a registry key when it resolves to a real file, else null (cached).
func slot_texture(key: String) -> Texture2D:
	if _tex_cache.has(key):
		return _tex_cache[key] as Texture2D
	var found: Texture2D = null
	if slots != null and slots.has(key):
		var path: String = slots.path_of(key)
		if slots.is_sheet(key):
			found = _sheet_frame(path, "%s_down" % slots.anim_of(key))
		elif ResourceLoader.exists(path):
			found = slots.texture(key)
	_tex_cache[key] = found
	return found


func _sheet_frame(json_path: String, anim: String) -> Texture2D:
	if not _frames_cache.has(json_path):
		_frames_cache[json_path] = SpriteSheetLoader.load_sheet(json_path)
	var frames: SpriteFrames = _frames_cache[json_path] as SpriteFrames
	if frames == null or not frames.has_animation(anim) or frames.get_frame_count(anim) == 0:
		return null
	return frames.get_frame_texture(anim, 0)


## Removes an id the store no longer has; a dead animated monster first plays its dead clip as a corpse.
func _remove(id: int, render: float) -> void:
	var rec: LayerEntityRec = _recs.get(id) as LayerEntityRec
	if rec != null and rec.track.has_anim and rec.track.action == ClipSelect.DEAD:
		_drop(id)  # the server's 3 s corpse (a = 7 rows) was already drawn
		return
	if rec == null or not rec.animated or rec.cat != ViewMath.CAT_MONSTER or rec.hp > 0:
		_drop(id)
		return
	if rec.corpse_until >= 0.0:
		return
	rec.corpse_until = render + CORPSE_MS
	rec.track.feed({"pos": rec.track.cells.cell(), "a": ClipSelect.DEAD, "t": floori(render / tick_ms), "m": 0},
			floori(render / tick_ms), tick_ms, render, tick_ms)
	rec.track.prev_action = ClipSelect.DEAD
	rec.track.prev_start_tick = floori(render / tick_ms)


func _drop(id: int) -> void:
	var slot: int = _pool.release(id)
	if slot >= 0:
		_last_pos[id] = _sprites[slot].position
		_sprites[slot].visible = false
		_sprites[slot].clear_clip()
	_tracks.erase(id)
	if _recs.erase(id):
		_recs_dirty = true
	_cats.erase(id)
	_looks.erase(id)


func _sprite_at(slot: int) -> EntitySprite:
	while _sprites.size() <= slot:
		var s := EntitySprite.new()
		s.visible = false
		add_child(s)
		_sprites.append(s)
	return _sprites[slot]
