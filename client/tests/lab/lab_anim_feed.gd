class_name LabAnimFeed
extends RefCounted
## A tiny deterministic stand-in for the JAX tick's anim state (round2 contract 1.1/1.2) that writes snap
## dictionaries with the protocol's anim fields (a, d, t, m, s, h) for the lab's animation metrics:
##   WALKERS walking monsters on 8-direction legs with RO speed accumulators (straight speed_ms, diagonal x1.4),
##   PAIRS attacking players (amotion AMOTION_MS, one swing per SWING_TICKS) with a monster target each that flinches
##   (hurt 300 ms) and carries `h` on the hit tick. Everything stays inside a BOX_W x BOX_H cell box so it fits a
##   1920x1080 screen at 32 px tiles. Like the server's changed_mask, a row is sent only when one of its fields
##   changed (or it carries `h`); the first snap is full.

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
const DELTAS: Array[Vector2i] = [Vector2i(0, 1), Vector2i(-1, 1), Vector2i(-1, 0), Vector2i(-1, -1),
		Vector2i(0, -1), Vector2i(1, -1), Vector2i(1, 0), Vector2i(1, 1)]

var tick_ms: int = 100
var tick: int = 0
var walkers: Array[Dictionary] = []  ## {id, pos, speed, acc, dir, left, rest, a, t}
var pairs: Array[Dictionary] = []  ## {pid, tid, ppos, tpos, phase}
var rows_sent: int = 0
var _last_rows: Dictionary = {}  ## id -> last row sent (without h)
var _rng := RandomNumberGenerator.new()


func _init(seed_value: int, new_tick_ms: int = 100) -> void:
	tick_ms = new_tick_ms
	_rng.seed = seed_value
	for i: int in range(WALKERS):
		walkers.append({"id": FIRST_WALKER_ID + i, "pos": Vector2i(_rng.randi_range(2, BOX_W - 3), _rng.randi_range(2, BOX_H - 3)),
				"speed": SPEEDS[i % SPEEDS.size()], "acc": 0, "dir": _rng.randi_range(0, 7),
				"left": _rng.randi_range(2, 8), "rest": _rng.randi_range(0, 10), "a": 0, "t": 0})
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
	for w: Dictionary in walkers:
		_step_walker(w)
		var pos: Vector2i = w["pos"]
		ents.append({"id": w["id"], "kind": 1, "mkind": int(w["id"]) % 4, "pos": [pos.x, pos.y], "hp": 50, "max_hp": 50,
				"a": w["a"], "d": w["dir"], "t": w["t"], "m": 0, "s": w["speed"]})
	for p: Dictionary in pairs:
		var swing: bool = (tick + int(p["phase"])) % SWING_TICKS == 0
		if swing:
			p["t"] = tick
		var t0: int = int(p.get("t", 0))
		var pp: Vector2i = p["ppos"]
		var tp: Vector2i = p["tpos"]
		var attacking: bool = tick - t0 < SWING_TICKS and p.has("t")
		ents.append({"id": p["pid"], "kind": 0, "job": int(p["pid"]) % 7, "name": "P%d" % int(p["pid"]),
				"pos": [pp.x, pp.y], "hp": 100, "max_hp": 100, "a": 5 if attacking else 0, "d": 6, "t": t0,
				"m": AMOTION_MS, "s": 150})
		var row: Dictionary = {"id": p["tid"], "kind": 1, "mkind": 0, "pos": [tp.x, tp.y], "hp": 40, "max_hp": 50,
				"a": 6 if p.has("t") else 0, "d": 2, "t": t0, "m": HURT_MS, "s": 150}
		if swing:
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
	return {"t": "snap", "tick": tick, "full": tick == 1, "ents": changed, "gone": [], "notice": "",
			"you": {"pos": [0, 0], "hp": 100, "max_hp": 100, "kills": 0, "deaths": 0, "inv": [], "town": 0, "safe": true}}


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
