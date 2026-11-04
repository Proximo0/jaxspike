class_name FxLogic
extends RefCounted
## Pure logic of the effect layer (no nodes, no wall clock): watches an EntityStore and runs a fixed pool of
## effect slots. observe() turns a new store version into effects, advance() ages them by a delta in ms and
## frame_of() says which frame of the sheet a slot shows. FxLayer only draws the slots.
## Effects: cast_circle loops under a caster while its action is 8; hit_spark plays once on a hit (h != 0),
## delayed to the attacker's swing; level_up when `you.level` rises; heal when hp jumps by 10% of max or more.
## ro6-fx2: the store's `fx_events` (snap head "fx": a skill resolved at a tile this tick) start a tile-anchored
## effect chosen by skill_kind(): physical melee skills -> hit_spark, the bolt family -> bolt tinted by element,
## Arrow Shower and Double Strafe -> arrow_shower, Magnum Break -> magnum_break, heals -> heal, the rest -> a one-shot
## cast_circle burst. Tile effects have targets[i] = -1 and their tile in tiles[i].

const CAST: int = 0
const HIT: int = 1
const LEVEL_UP: int = 2
const HEAL: int = 3
const BOLT: int = 4
const ARROW_SHOWER: int = 5
const MAGNUM_BREAK: int = 6
const CAST_BURST: int = 7  ## one pass of the cast circle sheet (a skill with no sprite of its own)
const KIND_NAMES: Array[String] = ["cast_circle", "hit_spark", "level_up", "heal", "bolt", "arrow_shower",
		"magnum_break", "cast_burst"]
## fx.json entry and fx.* slot each kind draws from.
const SHEET_NAMES: Array[String] = ["cast_circle", "hit_spark", "level_up", "heal", "bolt", "arrow_shower",
		"magnum_break", "cast_circle"]
## Skill ids (game/data/skills.py, never renumbered) and the welcome.skills effect / element codes.
const SK_FIRST_AID: int = 2
const SK_BASH: int = 4
const SK_MAGNUM_BREAK: int = 5
const SK_FIRE_BOLT: int = 10
const SK_COLD_BOLT: int = 11
const SK_LIGHTNING_BOLT: int = 12
const SK_NAPALM_BEAT: int = 13
const SK_SOUL_STRIKE: int = 14
const SK_DOUBLE_STRAFE: int = 16
const SK_ARROW_SHOWER: int = 17
const SK_HEAL: int = 20
const SK_ENVENOM: int = 28
const SK_MAMMONITE: int = 29
const BOLT_SKILLS: Array[int] = [SK_FIRE_BOLT, SK_COLD_BOLT, SK_LIGHTNING_BOLT, SK_NAPALM_BEAT, SK_SOUL_STRIKE]
const MELEE_SKILLS: Array[int] = [SK_BASH, SK_ENVENOM, SK_MAMMONITE]  ## used before welcome.skills arrives
const HEAL_SKILLS: Array[int] = [SK_FIRST_AID, SK_HEAL]
const EFFECT_ATK: int = 1
const EFFECT_HEAL: int = 3
const ELEM_NEUTRAL: int = 0
const ELEM_WATER: int = 1
const ELEM_FIRE: int = 3
const ELEM_WIND: int = 4
const ELEM_GHOST: int = 8
## Bolt elements before welcome.skills arrives (Fire, Water, Wind, Ghost, Ghost).
const BOLT_ELEMENTS: Array[int] = [ELEM_FIRE, ELEM_WATER, ELEM_WIND, ELEM_GHOST, ELEM_GHOST]
const MAX_LIVE: int = 64
const ACTION_CAST: int = 8
const HEAL_FRACTION: float = 0.10

## Used when fx.json is missing or lacks an entry: {frames, delay_ms, loop}.
const DEFAULT_SPECS: Array[Dictionary] = [
	{"frames": 8, "delay_ms": 90.0, "loop": true},
	{"frames": 6, "delay_ms": 60.0, "loop": false},
	{"frames": 10, "delay_ms": 80.0, "loop": false},
	{"frames": 8, "delay_ms": 80.0, "loop": false},
	{"frames": 6, "delay_ms": 60.0, "loop": false},
	{"frames": 8, "delay_ms": 70.0, "loop": false},
	{"frames": 6, "delay_ms": 60.0, "loop": false},
	{"frames": 8, "delay_ms": 60.0, "loop": false},
]

# Slot arrays (fixed size, reused).
var used: PackedByteArray = PackedByteArray()
var kinds: PackedInt32Array = PackedInt32Array()
var targets: PackedInt32Array = PackedInt32Array()  ## entity id the effect follows
var ages: PackedFloat32Array = PackedFloat32Array()  ## ms since it started playing (negative = still waiting)
var last_pos: PackedVector2Array = PackedVector2Array()  ## last drawn world px, kept when the target is gone
var tiles: PackedVector2Array = PackedVector2Array()  ## ro6-fx2: tile of a tile effect (targets[i] = -1)
var tints: PackedColorArray = PackedColorArray()  ## ro6-fx2: sprite modulate (bolt element), white otherwise
var live_count: int = 0
var dropped: int = 0  ## effects refused because all 64 slots were busy
var specs: Array[Dictionary] = []

var _casting: Dictionary = {}  ## caster id -> slot
var _skill_effect: Dictionary = {}  ## ro6-fx2: skill id -> welcome.skills effect code
var _skill_element: Dictionary = {}  ## ro6-fx2: skill id -> welcome.skills element code
var _last_hp: Dictionary = {}  ## id -> hp seen
var _level: int = -1
var _store: EntityStore = null
var _version: int = -1


func _init() -> void:
	used.resize(MAX_LIVE)
	kinds.resize(MAX_LIVE)
	targets.resize(MAX_LIVE)
	ages.resize(MAX_LIVE)
	last_pos.resize(MAX_LIVE)
	tiles.resize(MAX_LIVE)
	tints.resize(MAX_LIVE)
	for i: int in KIND_NAMES.size():
		specs.append(DEFAULT_SPECS[i].duplicate())


## Reads {frames, delay_ms, loop} per effect from the parsed fx.json (`effects` dictionary).
func configure(effects: Dictionary) -> void:
	for i: int in KIND_NAMES.size():
		var e: Variant = effects.get(SHEET_NAMES[i])
		if typeof(e) != TYPE_DICTIONARY:
			continue
		var d: Dictionary = e
		specs[i] = {"frames": maxi(1, int(d.get("frames", 1))), "delay_ms": maxf(1.0, float(d.get("delay_ms", 80))),
				"loop": bool(d.get("loop", false)) and i != CAST_BURST}


## ro6-fx2: reads effect and element per skill id from welcome.skills.list ([{id, effect, element, ...}]).
func set_skills(rows: Array) -> void:
	_skill_effect.clear()
	_skill_element.clear()
	for r: Variant in rows:
		if typeof(r) != TYPE_DICTIONARY or not (r as Dictionary).has("id"):
			continue
		var d: Dictionary = r
		_skill_effect[int(d["id"])] = int(d.get("effect", -1))
		_skill_element[int(d["id"])] = int(d.get("element", ELEM_NEUTRAL))


## ro6-fx2: the effect kind a resolved skill plays.
func skill_kind(skill: int) -> int:
	if BOLT_SKILLS.has(skill):
		return BOLT
	if skill == SK_ARROW_SHOWER or skill == SK_DOUBLE_STRAFE:
		return ARROW_SHOWER
	if skill == SK_MAGNUM_BREAK:
		return MAGNUM_BREAK
	var eff: int = int(_skill_effect.get(skill, -1))
	if HEAL_SKILLS.has(skill) or eff == EFFECT_HEAL:
		return HEAL
	if eff == EFFECT_ATK or (eff < 0 and MELEE_SKILLS.has(skill)):
		return HIT
	return CAST_BURST


## ro6-fx2: the skill's element (welcome.skills, else the built-in bolt table, else Neutral).
func skill_element(skill: int) -> int:
	if _skill_element.has(skill):
		return int(_skill_element[skill])
	var i: int = BOLT_SKILLS.find(skill)
	return BOLT_ELEMENTS[i] if i >= 0 else ELEM_NEUTRAL


## ro6-fx2: bolt tint per element: blue cold, orange fire, yellow lightning, violet ghost, white otherwise.
static func element_tint(element: int) -> Color:
	match element:
		ELEM_FIRE:
			return Color(1.0, 0.55, 0.2)
		ELEM_WATER:
			return Color(0.45, 0.72, 1.0)
		ELEM_WIND:
			return Color(1.0, 0.95, 0.35)
		ELEM_GHOST:
			return Color(0.82, 0.62, 1.0)
	return Color.WHITE


## Reacts to the store's newest snap (call every frame; it returns at once when nothing changed).
## `own_id` is the entity id of the own character; `hit_delay` (Callable(id, tick) -> ms >= 0, may be invalid)
## says how long a hit on `id` waits for the attacker's swing. Returns the number of effects started.
func observe(store: EntityStore, own_id: int, hit_delay: Callable) -> int:
	var started: int = 0
	var rebase: bool = store != _store or (store.version != _version and store.version != _version + 1)
	if store == _store and store.version == _version:
		return 0
	if store != _store:
		_reset()
	_store = store
	_version = store.version
	# Level-up of the own character.
	var lv: int = int(store.you.get("level", -1))
	if lv >= 0:
		if _level >= 0 and lv > _level and not rebase:
			started += _start(LEVEL_UP, own_id, 0.0)
		_level = lv
	for id: int in store.removed_ids():
		if not store.has_ent(id):
			_stop_cast(id)
			_last_hp.erase(id)
	if rebase:
		_casting_rebuild(store)
		for id: int in store.ids():
			_last_hp[id] = int(store.get_ent(id).get("hp", 0))
		return started + _skill_fx(store, {})
	var hit_tiles: Dictionary = {}  ## tiles that already got an entity hit spark this snap
	for id: int in store.changed_ids():
		if not store.has_ent(id):
			continue
		var e: Dictionary = store.get_ent(id)
		var casting: bool = int(e.get("a", 0)) == ACTION_CAST
		if casting and not _casting.has(id):
			if _start(CAST, id, 0.0) > 0:
				started += 1
				_casting[id] = _slot_of_last
		elif not casting:
			_stop_cast(id)
		var h: int = int(e.get("h", 0)) if e.has("h") else 0
		if h != 0:
			var wait: float = 0.0
			if hit_delay.is_valid():
				wait = maxf(0.0, float(hit_delay.call(id, store.tick)))
			started += _start(HIT, id, wait)
			hit_tiles[e.get("pos", Vector2i.ZERO)] = true
		var hp: int = int(e.get("hp", 0))
		var max_hp: int = int(e.get("max_hp", 0))
		if _last_hp.has(id) and max_hp > 0 and hp - int(_last_hp[id]) >= ceili(float(max_hp) * HEAL_FRACTION):
			started += _start(HEAL, id, 0.0)
		_last_hp[id] = hp
	return started + _skill_fx(store, hit_tiles)


var _slot_of_last: int = -1


## ro6-fx2: one tile effect per store.fx_events entry; a melee spark is skipped where the hit already sparked.
func _skill_fx(store: EntityStore, hit_tiles: Dictionary) -> int:
	var started: int = 0
	for ev: Dictionary in store.fx_events:
		var skill: int = int(ev.get("skill", 0))
		var tile: Vector2i = ev.get("tile", Vector2i.ZERO)
		var kind: int = skill_kind(skill)
		if kind == HIT and hit_tiles.has(tile):
			continue
		if _start(kind, -1, 0.0) > 0:
			started += 1
			tiles[_slot_of_last] = Vector2(tile)
			if kind == BOLT:
				tints[_slot_of_last] = element_tint(skill_element(skill))
	return started


## Ages every live effect by `delta_ms`; one-shots whose last frame has passed are freed.
func advance(delta_ms: float) -> void:
	for i: int in MAX_LIVE:
		if used[i] == 0:
			continue
		ages[i] += delta_ms
		var sp: Dictionary = specs[kinds[i]]
		if not bool(sp["loop"]) and ages[i] >= float(sp["frames"]) * float(sp["delay_ms"]):
			_free(i)


## Frame to draw for slot `i`, or -1 while it waits for its start time (or is unused).
func frame_of(i: int) -> int:
	if i < 0 or i >= MAX_LIVE or used[i] == 0 or ages[i] < 0.0:
		return -1
	var sp: Dictionary = specs[kinds[i]]
	var f: int = floori(ages[i] / float(sp["delay_ms"]))
	var n: int = int(sp["frames"])
	return f % n if bool(sp["loop"]) else mini(f, n - 1)


func live_of(kind: int) -> int:
	var n: int = 0
	for i: int in MAX_LIVE:
		if used[i] != 0 and kinds[i] == kind:
			n += 1
	return n


func is_casting(id: int) -> bool:
	return _casting.has(id)


func _reset() -> void:
	for i: int in MAX_LIVE:
		if used[i] != 0:
			_free(i)
	_casting.clear()
	_last_hp.clear()
	_level = -1


func _casting_rebuild(store: EntityStore) -> void:
	for id: int in store.ids():
		var casting: bool = int(store.get_ent(id).get("a", 0)) == ACTION_CAST
		if casting and not _casting.has(id):
			if _start(CAST, id, 0.0) > 0:
				_casting[id] = _slot_of_last
		elif not casting:
			_stop_cast(id)
	for id: int in _casting.keys():
		if not store.has_ent(id):
			_stop_cast(id)


func _stop_cast(id: int) -> void:
	if _casting.has(id):
		_free(int(_casting[id]))
		_casting.erase(id)


func _start(kind: int, id: int, wait_ms: float) -> int:
	for i: int in MAX_LIVE:
		if used[i] == 0:
			used[i] = 1
			kinds[i] = kind
			targets[i] = id
			ages[i] = -wait_ms
			last_pos[i] = Vector2.INF
			tiles[i] = Vector2.ZERO
			tints[i] = Color.WHITE
			live_count += 1
			_slot_of_last = i
			return 1
	dropped += 1
	_slot_of_last = -1
	return 0


func _free(i: int) -> void:
	if used[i] != 0:
		used[i] = 0
		live_count -= 1
