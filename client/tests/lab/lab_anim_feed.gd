class_name LabAnimFeed
extends RefCounted
## A tiny deterministic stand-in for the JAX tick's anim state (round2 contract 1.1/1.2) that writes snap
## dictionaries with the protocol's anim fields (a, d, t, m, s, h) for the lab's animation metrics:
##   WALKERS walking monsters on 8-direction legs with RO speed accumulators (straight speed_ms, diagonal x1.4);
##   every HIT_EVERY_TICKS one walking walker flinches (a=6, h) for HURT_TICKS while its position keeps stepping,
##   PAIRS attacking players (amotion AMOTION_MS, one swing per SWING_TICKS) with a monster target each that flinches
##   (hurt 300 ms) and carries `h` on the hit tick. Players with index % 5 == 1 sit SIT_TICKS per SIT_CYCLE_TICKS
##   instead of attacking; index % 5 == 2 play a pickup every PICKUP_EVERY_TICKS between swings. Every
##   DEATH_EVERY_TICKS one target (round-robin) dies (a=7, hp 0) for DEAD_TICKS, is `gone` for one snap, then respawns.
##   Everything stays inside a BOX_W x BOX_H cell box so it fits a 1920x1080 screen at 32 px tiles. Like the
##   server's changed_mask, a row is sent only when one of its fields changed (or it carries `h`); the first snap is full.

const WALKERS: int = 150
const PAIRS: int = 25
const FIRST_WALKER_ID: int = 512
const FIRST_PLAYER_ID: int = 1
const FIRST_TARGET_ID: int = 800
const BOX_W: int = 56
const BOX_H: int = 30
const SPEEDS: PackedInt32Array = [150, 150, 150, 200, 120, 300]
const AMOTION_MS: int = 500
const SWING_TICKS: int = 10
const HURT_MS: int = 300
const HIT_EVERY_TICKS: int = 15
const HURT_TICKS: int = 3
const HIT_MIN: int = 5
const HIT_MAX: int = 20
const SIT_TICKS: int = 40
const SIT_CYCLE_TICKS: int = 120
const PICKUP_EVERY_TICKS: int = 30
const PICKUP_MS: int = 400
const PICKUP_TICKS: int = 4
const PICKUP_OFFSET: int = 5  ## pickup starts this many ticks after a swing, so it sits between swings
const ATTACK_ROW_TICKS: int = 5  ## pickers show a=5 only for the amotion, then idle
const DEATH_EVERY_TICKS: int = 50
const DEAD_TICKS: int = 30
const DELTAS: Array[Vector2i] = [Vector2i(0, 1), Vector2i(-1, 1), Vector2i(-1, 0), Vector2i(-1, -1),
		Vector2i(0, -1), Vector2i(1, -1), Vector2i(1, 0), Vector2i(1, 1)]

var tick_ms: int = 100
var tick: int = 0
var walkers: Array[Dictionary] = []  ## {id, pos, speed, acc, dir, left, rest, a, t, hurt_until, hit_t, h}
var pairs: Array[Dictionary] = []  ## {pid, tid, ppos, tpos, phase}
var rows_sent: int = 0
var _last_rows: Dictionary = {}  ## id -> last row sent (without h)
var _rng := RandomNumberGenerator.new()
var _hit_cursor: int = 0
var _dead_idx: int = -1  ## index into pairs of the target dying (or -1)
var _dead_tick: int = 0
var _next_death: int = 0


func _init(seed_value: int, new_tick_ms: int = 100) -> void:
	tick_ms = new_tick_ms
	_rng.seed = seed_value
	for i: int in range(WALKERS):
		walkers.append({"id": FIRST_WALKER_ID + i, "pos": Vector2i(_rng.randi_range(2, BOX_W - 3), _rng.randi_range(2, BOX_H - 3)),
				"speed": SPEEDS[i % SPEEDS.size()], "acc": 0, "dir": _rng.randi_range(0, 7),
				"left": _rng.randi_range(2, 8), "rest": _rng.randi_range(0, 10), "a": 0, "t": 0,
				"hurt_until": 0, "hit_t": 0, "h": 0})
	_hit_cursor = _rng.randi_range(0, WALKERS - 1)
	for i: int in range(PAIRS):
		var p := Vector2i(2 + (i % 9) * 6, BOX_H - 2 - (i / 9) * 3)
		pairs.append({"pid": FIRST_PLAYER_ID + i, "tid": FIRST_TARGET_ID + i, "ppos": p, "tpos": p + Vector2i(1, 0),
				"phase": i % SWING_TICKS})


func is_walker(id: int) -> bool:
	return id >= FIRST_WALKER_ID and id < FIRST_WALKER_ID + WALKERS


func walker_speed(id: int) -> float:
	return float(walkers[id - FIRST_WALKER_ID]["speed"])


## Advances one tick and returns the snap dictionary (full on the first call).
func step() -> Dictionary:
	tick += 1
	var ents: Array = []
	var gone: Array = []
	_pick_hit()
	for w: Dictionary in walkers:
		_step_walker(w)
		var pos: Vector2i = w["pos"]
		var hurt: bool = tick < int(w["hurt_until"])
		if int(w["hurt_until"]) > 0 and not hurt:
			w["hurt_until"] = 0
			w["a"] = 1 if int(w["rest"]) == 0 else 0
			w["t"] = tick
		var row: Dictionary = {"id": w["id"], "kind": 1, "mkind": int(w["id"]) % 4, "pos": [pos.x, pos.y], "hp": 50, "max_hp": 50,
				"a": 6 if hurt else w["a"], "d": w["dir"], "t": w["hit_t"] if hurt else w["t"], "m": HURT_MS if hurt else 0,
				"s": w["speed"]}
		if hurt and int(w["hit_t"]) == tick:
			row["h"] = int(w["h"])
		ents.append(row)
	_step_death(gone)
	for i: int in range(PAIRS):
		var p: Dictionary = pairs[i]
		var phase: int = int(p["phase"])
		var sit_r: int = (tick + i * 7) % SIT_CYCLE_TICKS
		var sitting: bool = i % 5 == 1 and sit_r < SIT_TICKS
		var swing: bool = (tick + phase) % SWING_TICKS == 0 and not sitting
		if swing:
			p["t"] = tick
		var t0: int = int(p.get("t", 0))
		var pp: Vector2i = p["ppos"]
		var tp: Vector2i = p["tpos"]
		var attacking: bool = tick - t0 < SWING_TICKS and p.has("t")
		var pa: int = 5 if attacking else 0
		var pt: int = t0
		var pm: int = AMOTION_MS
		if i % 5 == 2:
			attacking = attacking and tick - t0 < ATTACK_ROW_TICKS
			pa = 5 if attacking else 0
			var pick_r: int = (tick + phase) % SWING_TICKS
			var cycle: int = PICKUP_EVERY_TICKS / SWING_TICKS
			if pick_r >= PICKUP_OFFSET and pick_r < PICKUP_OFFSET + PICKUP_TICKS and ((tick + phase) / SWING_TICKS) % cycle == 0:
				pa = 3
				pt = maxi(tick - (pick_r - PICKUP_OFFSET), 0)
				pm = PICKUP_MS
		elif sitting:
			pa = 2
			pt = maxi(tick - sit_r, 0)
			pm = 0
		ents.append({"id": p["pid"], "kind": 0, "job": int(p["pid"]) % 7, "name": "P%d" % int(p["pid"]),
				"pos": [pp.x, pp.y], "hp": 100, "max_hp": 100, "a": pa, "d": 6, "t": pt,
				"m": pm, "s": 150})
		if i == _dead_idx and tick == _dead_tick + DEAD_TICKS:
			continue  # the `gone` snap
		var row: Dictionary = {"id": p["tid"], "kind": 1, "mkind": 0, "pos": [tp.x, tp.y], "hp": 40, "max_hp": 50,
				"a": 6 if p.has("t") else 0, "d": 2, "t": t0, "m": HURT_MS, "s": 150}
		if i == _dead_idx and tick < _dead_tick + DEAD_TICKS:
			row["a"] = 7
			row["t"] = _dead_tick
			row["m"] = 0
			row["hp"] = 0
		elif i == _dead_idx and tick == _dead_tick + DEAD_TICKS + 1:
			row["a"] = 0
			row["t"] = tick
		elif swing:
			row["h"] = 7 + (tick % 13)
		ents.append(row)
	var changed: Array = []
	for e: Variant in ents:
		var row: Dictionary = e
		var key: Dictionary = row.duplicate()
		key.erase("h")
		var id: int = int(row["id"])
		if tick == 1 or row.has("h") or _last_rows.get(id, {}) != key:
			changed.append(row)
			_last_rows[id] = key
	rows_sent += changed.size()
	return {"t": "snap", "tick": tick, "full": tick == 1, "ents": changed, "gone": gone, "notice": "",
			"you": {"pos": [0, 0], "hp": 100, "max_hp": 100, "kills": 0, "deaths": 0, "inv": [], "town": 0, "safe": true}}


## Every HIT_EVERY_TICKS the next walking (not already hurt) walker, scanning round-robin from a seeded start, is hit.
func _pick_hit() -> void:
	if tick % HIT_EVERY_TICKS != 0:
		return
	for k: int in range(WALKERS):
		var w: Dictionary = walkers[(_hit_cursor + k) % WALKERS]
		if int(w["a"]) == 1 and int(w["rest"]) == 0 and int(w["hurt_until"]) == 0:
			w["hurt_until"] = tick + HURT_TICKS
			w["hit_t"] = tick
			w["h"] = _rng.randi_range(HIT_MIN, HIT_MAX)
			_hit_cursor = (_hit_cursor + k + 1) % WALKERS
			return


## Starts the round-robin death on its tick; on the `gone` tick lists the id and forgets its last row so the respawn is full.
func _step_death(gone: Array) -> void:
	if tick % DEATH_EVERY_TICKS == 0:
		_dead_idx = _next_death % PAIRS
		_next_death += 1
		_dead_tick = tick
	if _dead_idx >= 0 and tick == _dead_tick + DEAD_TICKS:
		var tid: int = int(pairs[_dead_idx]["tid"])
		gone.append(tid)
		_last_rows.erase(tid)


func _step_walker(w: Dictionary) -> void:
	if int(w["rest"]) > 0:
		w["rest"] = int(w["rest"]) - 1
		if int(w["a"]) != 0:
			w["a"] = 0
			w["t"] = tick
		w["acc"] = 0
		return
	var d: int = int(w["dir"])
	var cost: int = int(w["speed"]) * (14 if d % 2 == 1 else 10) / 10
	w["acc"] = int(w["acc"]) + tick_ms
	if int(w["acc"]) < cost:
		return
	w["acc"] = int(w["acc"]) - cost
	var nxt: Vector2i = (w["pos"] as Vector2i) + DELTAS[d]
	if nxt.x < 1 or nxt.y < 1 or nxt.x > BOX_W - 2 or nxt.y > BOX_H - 6:
		w["dir"] = (d + 4) % 8
		return
	w["pos"] = nxt
	if int(w["a"]) != 1:
		w["a"] = 1
		w["t"] = tick
	w["left"] = int(w["left"]) - 1
	if int(w["left"]) <= 0:
		w["left"] = _rng.randi_range(2, 8)
		w["rest"] = _rng.randi_range(0, 12)
		w["dir"] = _rng.randi_range(0, 7)
