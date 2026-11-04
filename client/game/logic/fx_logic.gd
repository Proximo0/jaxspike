class_name FxLogic
extends RefCounted
## Pure logic of the effect layer (no nodes, no wall clock): watches an EntityStore and runs a fixed pool of
## effect slots. observe() turns a new store version into effects, advance() ages them by a delta in ms and
## frame_of() says which frame of the sheet a slot shows. FxLayer only draws the slots.
## Effects: cast_circle loops under a caster while its action is 8; hit_spark plays once on a hit (h != 0),
## delayed to the attacker's swing; level_up when `you.level` rises; heal when hp jumps by 10% of max or more.

const CAST: int = 0
const HIT: int = 1
const LEVEL_UP: int = 2
const HEAL: int = 3
const KIND_NAMES: Array[String] = ["cast_circle", "hit_spark", "level_up", "heal"]
const MAX_LIVE: int = 64
const ACTION_CAST: int = 8
const HEAL_FRACTION: float = 0.10

## Used when fx.json is missing or lacks an entry: {frames, delay_ms, loop}.
const DEFAULT_SPECS: Array[Dictionary] = [
	{"frames": 8, "delay_ms": 90.0, "loop": true},
	{"frames": 6, "delay_ms": 60.0, "loop": false},
	{"frames": 10, "delay_ms": 80.0, "loop": false},
	{"frames": 8, "delay_ms": 80.0, "loop": false},
]

# Slot arrays (fixed size, reused).
var used: PackedByteArray = PackedByteArray()
var kinds: PackedInt32Array = PackedInt32Array()
var targets: PackedInt32Array = PackedInt32Array()  ## entity id the effect follows
var ages: PackedFloat32Array = PackedFloat32Array()  ## ms since it started playing (negative = still waiting)
var last_pos: PackedVector2Array = PackedVector2Array()  ## last drawn world px, kept when the target is gone
var live_count: int = 0
var dropped: int = 0  ## effects refused because all 64 slots were busy
var specs: Array[Dictionary] = []

var _casting: Dictionary = {}  ## caster id -> slot
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
	for i: int in KIND_NAMES.size():
		specs.append(DEFAULT_SPECS[i].duplicate())


## Reads {frames, delay_ms, loop} per effect from the parsed fx.json (`effects` dictionary).
func configure(effects: Dictionary) -> void:
	for i: int in KIND_NAMES.size():
		var e: Variant = effects.get(KIND_NAMES[i])
		if typeof(e) != TYPE_DICTIONARY:
			continue
		var d: Dictionary = e
		specs[i] = {"frames": maxi(1, int(d.get("frames", 1))), "delay_ms": maxf(1.0, float(d.get("delay_ms", 80))),
				"loop": bool(d.get("loop", false))}


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
		return started
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
		var hp: int = int(e.get("hp", 0))
		var max_hp: int = int(e.get("max_hp", 0))
		if _last_hp.has(id) and max_hp > 0 and hp - int(_last_hp[id]) >= ceili(float(max_hp) * HEAL_FRACTION):
			started += _start(HEAL, id, 0.0)
		_last_hp[id] = hp
	return started


var _slot_of_last: int = -1


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
