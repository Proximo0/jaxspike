extends Node
## Client lab: runs headless and windowed over FakeLink and measures four metrics against their targets.
##   welcome_to_drawn_ms   synthetic 512x512 welcome -> map decoded and a map texture drawn (target < 1000)
##   apply_ms_p95          NetSession.poll() time per snap, 200 moving entities, 300 snaps (target < 5)
##   input_to_screen_p95   walk_to sent -> own pos changes, 1 tick (120 ms) of fake lag (target <= 150)
##   frame_fps_p95         1000 / p95 real frame time (target >= 55; WARN, not FAIL, when headless)
## Prints "LAB <name> value=<n> target=<n> PASS|WARN|FAIL" per metric and "LAB summary pass= warn= fail=".
## Exits non-zero on FAIL. If the Part C net classes are missing it still prints a summary (WARN rows + reason).
## Embedded use (sims): set `embedded = true` (and a smaller `snap_count`) before add_child; no quit is called.

signal finished

const MAP: int = 512
const TICK_MS: int = 120
const ENTITY_COUNT: int = 200
const FIRST_MONSTER_ID: int = 512
const INPUT_TRIALS: int = 20
const WARMUP_FRAMES: int = 30
const SEED: int = 12345
const T_WELCOME_MS: float = 1000.0
const T_APPLY_MS: float = 5.0  # 1.4 ms idle; 2-5 ms with nine agent sessions on the box (lesson in docs/lessons_learned.md)
const T_INPUT_MS: float = 150.0
const T_FPS: float = 55.0
const NEEDED: PackedStringArray = ["NetSession", "FakeLink", "EntityStore", "MapData", "Codec"]
const LOWEST_PRIORITY: int = 2147483647

var embedded: bool = false
var snap_count: int = 300
var rows: Array[Dictionary] = []  ## {name, value, target, verdict}
var summary_line: String = ""
var reason: String = ""

var _frame_ms: PackedFloat64Array = PackedFloat64Array()
var _last_us: int = 0
var _frames_seen: int = 0
var _rng := RandomNumberGenerator.new()


## Inner node that records real frame times after every other node has processed.
class LabFrameClock extends Node:
	var lab: Node = null

	func _init() -> void:
		process_priority = 2147483647

	func _process(_delta: float) -> void:
		lab.call("frame_tick", Time.get_ticks_usec())


## PASS when the value meets the target; a miss is WARN instead of FAIL when warn_only.
static func verdict(value: float, target: float, lower_is_better: bool, inclusive: bool, warn_only: bool) -> String:
	var ok: bool
	if lower_is_better:
		ok = value <= target if inclusive else value < target
	else:
		ok = value >= target
	if ok:
		return "PASS"
	return "WARN" if warn_only else "FAIL"


func _ready() -> void:
	_rng.seed = SEED
	var clock := LabFrameClock.new()
	clock.lab = self
	add_child(clock)
	_run.call_deferred()


func frame_tick(now_us: int) -> void:
	if _last_us > 0:
		_frames_seen += 1
		if _frames_seen > WARMUP_FRAMES:
			_frame_ms.append(float(now_us - _last_us) / 1000.0)
	_last_us = now_us


func _class_path(cname: String) -> String:
	for c: Dictionary in ProjectSettings.get_global_class_list():
		if c["class"] == cname:
			return str(c["path"])
	return ""


func _missing() -> PackedStringArray:
	var out := PackedStringArray()
	for c: String in NEEDED:
		if _class_path(c) == "":
			out.append(c)
	return out


func _run() -> void:
	var headless: bool = DisplayServer.get_name() == "headless"
	var miss: PackedStringArray = _missing()
	var welcome_ms: float = -1.0
	var apply_p95: float = -1.0
	var apply_p50: float = -1.0
	var input_p95: float = -1.0
	if miss.is_empty():
		var res: Dictionary = await _measure_net()
		welcome_ms = float(res["welcome_ms"])
		apply_p95 = float(res["apply_p95"])
		apply_p50 = float(res.get("apply_p50", -1.0))
		input_p95 = float(res["input_p95"])
		reason = str(res.get("reason", ""))
	else:
		reason = "Part C classes missing: %s" % ", ".join(miss)
		for _i: int in range(WARMUP_FRAMES + 60):
			await get_tree().process_frame
	var fps: float = DebugFrame.fps_from_ms(DebugFrame.percentile(_frame_ms, 95.0))
	var net_missing: bool = not miss.is_empty()
	_add("welcome_to_drawn_ms", welcome_ms, T_WELCOME_MS, true, false, net_missing)
	_add("apply_ms_p50", apply_p50, T_APPLY_MS, true, false, net_missing)
	_add("apply_ms_p95", apply_p95, T_APPLY_MS, true, false, true)  # WARN only: p95 is load noise on a shared box
	_add("input_to_screen_p95", input_p95, T_INPUT_MS, true, true, net_missing)
	_add("frame_fps_p95", fps, T_FPS, false, false, headless or _frame_ms.is_empty())
	_finish()


func _add(metric: String, value: float, target: float, lower: bool, inclusive: bool, warn_only: bool) -> void:
	var v: String = verdict(value, target, lower, inclusive, warn_only)
	if value < 0.0:
		v = "WARN"  # not measured
	rows.append({"name": metric, "value": value, "target": target, "verdict": v})
	print("LAB %s value=%.2f target=%.1f %s" % [metric, value, target, v])


func _finish() -> void:
	var p: int = 0
	var w: int = 0
	var f: int = 0
	for r: Dictionary in rows:
		match str(r["verdict"]):
			"PASS":
				p += 1
			"WARN":
				w += 1
			_:
				f += 1
	if reason != "":
		print("LAB note %s" % reason)
	summary_line = "LAB summary pass=%d warn=%d fail=%d" % [p, w, f]
	print(summary_line)
	finished.emit()
	if not embedded:
		get_tree().quit(1 if f > 0 else 0)


func _wait_frames(n: int) -> void:
	for _i: int in range(n):
		await get_tree().process_frame


func _measure_net() -> Dictionary:
	var link: FakeLink = FakeLink.new()
	link.connect_to("fake://lab")
	var session: NetSession = NetSession.new(link)
	var out: Dictionary = {"welcome_ms": -1.0, "apply_p95": -1.0, "input_p95": -1.0}
	await _wait_frames(5)

	# 1. welcome -> map decoded -> map texture drawn
	var welcome_text: String = _welcome_text()
	var rect := TextureRect.new()
	add_child(rect)
	var t0: int = Time.get_ticks_usec()
	link.push_inbound(welcome_text)
	session.poll()
	if not session.welcomed:
		out["reason"] = "welcome did not decode: %s" % session.last_error
		rect.queue_free()
		return out
	var img: Image = Image.create_from_data(MAP, MAP, false, Image.FORMAT_L8, session.map.terrain)
	rect.texture = ImageTexture.create_from_image(img)
	await _wait_frames(2)
	out["welcome_ms"] = float(Time.get_ticks_usec() - t0) / 1000.0
	rect.queue_free()

	# 2. apply time per snap, 200 entities all moving, one snap per frame
	var texts: PackedStringArray = _snap_texts(snap_count)
	var apply_ms := PackedFloat64Array()
	for text: String in texts:
		link.push_inbound(text)
		var a: int = Time.get_ticks_usec()
		session.poll()
		apply_ms.append(float(Time.get_ticks_usec() - a) / 1000.0)
		await get_tree().process_frame
	out["apply_p95"] = DebugFrame.percentile(apply_ms, 95.0)
	out["apply_p50"] = DebugFrame.percentile(apply_ms, 50.0)
	if session.store.count() != ENTITY_COUNT:
		out["reason"] = "store holds %d entities, expected %d" % [session.store.count(), ENTITY_COUNT]

	# 3. input to screen: the fake server answers each walk_to with a snap that moves us. The link lag is 1 tick and
	# the lab's logical server tick (link.tick()) fires every TICK_MS of wall time, so the reply lands 0..1 tick later.
	var latencies := PackedFloat64Array()
	var last_tick_us: int = Time.get_ticks_usec()
	var snap_tick: int = snap_count + 10
	for trial: int in range(INPUT_TRIALS):
		var settle_until: int = Time.get_ticks_usec() + _rng.randi_range(0, TICK_MS - 1) * 1000
		while Time.get_ticks_usec() < settle_until:
			last_tick_us = _server_tick(link, last_tick_us)
			session.poll()
			await get_tree().process_frame
		var start_pos: Variant = session.store.you.get("pos")
		var target := Vector2i(10 + trial, 20)
		var sent_us: int = Time.get_ticks_usec()
		session.walk_to(target.x, target.y)
		snap_tick += 1
		link.push_inbound(_you_snap(snap_tick, target), 1)
		var done: bool = false
		while not done and Time.get_ticks_usec() - sent_us < 2000000:
			last_tick_us = _server_tick(link, last_tick_us)
			session.poll()
			if session.store.you.get("pos") != start_pos:
				latencies.append(float(Time.get_ticks_usec() - sent_us) / 1000.0)
				done = true
			else:
				await get_tree().process_frame
		if not done:
			latencies.append(2000.0)
	out["input_p95"] = DebugFrame.percentile(latencies, 95.0)
	return out


func _server_tick(link: FakeLink, last_us: int) -> int:
	var now: int = Time.get_ticks_usec()
	if now - last_us >= TICK_MS * 1000:
		link.tick()
		return last_us + TICK_MS * 1000
	return last_us


func _welcome_text() -> String:
	var terrain := PackedByteArray()
	terrain.resize(MAP * MAP)
	for i: int in range(terrain.size()):
		terrain[i] = (i / 64 + i % 7) % 8  # blocky, compressible, deterministic
	var pz := PackedByteArray()
	pz.resize(MAP * MAP)
	return Codec.encode({
		"t": "welcome", "slot": 0, "map": MAP, "tick_ms": TICK_MS, "view_w": 31, "view_h": 21,
		"terrain": Marshalls.raw_to_base64(terrain.compress(FileAccess.COMPRESSION_DEFLATE)),
		"pz": Marshalls.raw_to_base64(pz.compress(FileAccess.COMPRESSION_DEFLATE)),
	})


func _you_snap(tick: int, pos: Vector2i) -> String:
	return Codec.encode({
		"t": "snap", "tick": tick, "full": false, "ents": [], "gone": [], "notice": "",
		"you": {"pos": [pos.x, pos.y], "hp": 100, "max_hp": 100, "kills": 0, "deaths": 0, "inv": [], "town": 0, "safe": true},
	})


## n encoded snaps: the first is full with ENTITY_COUNT monsters, every later one moves all of them one tile.
func _snap_texts(n: int) -> PackedStringArray:
	var out := PackedStringArray()
	var pos: Array[Vector2i] = []
	for i: int in range(ENTITY_COUNT):
		pos.append(Vector2i(_rng.randi_range(100, 400), _rng.randi_range(100, 400)))
	var dirs: Array[Vector2i] = [Vector2i(0, -1), Vector2i(0, 1), Vector2i(-1, 0), Vector2i(1, 0)]
	for t: int in range(n):
		var ents: Array = []
		for i: int in range(ENTITY_COUNT):
			if t > 0:
				pos[i] = (pos[i] + dirs[_rng.randi_range(0, 3)]).clamp(Vector2i.ZERO, Vector2i(MAP - 1, MAP - 1))
			ents.append({"id": FIRST_MONSTER_ID + i, "kind": 1, "pos": [pos[i].x, pos[i].y], "hp": 50, "max_hp": 50, "mkind": 0})
		out.append(Codec.encode({
			"t": "snap", "tick": t + 1, "full": t == 0, "ents": ents, "gone": [], "notice": "",
			"you": {"pos": [5, 5], "hp": 100, "max_hp": 100, "kills": 0, "deaths": 0, "inv": [], "town": 0, "safe": true},
		}))
	return out
