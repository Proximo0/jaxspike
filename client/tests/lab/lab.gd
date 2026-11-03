extends Node
## Client lab: runs headless and windowed over FakeLink and measures four metrics against their targets.
##   welcome_to_drawn_ms   synthetic 512x512 welcome -> map decoded and a map texture drawn (target < 1000)
##   apply_ms_p95          NetSession.poll() time per snap, 200 moving entities, 300 snaps (target < 5)
##   input_to_screen_p95   walk_to sent -> own pos changes, 1 tick (120 ms) of fake lag (target <= 150)
##   frame_fps_p95         1000 / p95 real frame time (target >= 55; WARN, not FAIL, when headless)
## Animation (round2 contract 1.3), over FakeLink snaps from LabAnimFeed (200 animated entities with a/d/t/m/s/h,
## 100 ms ticks, 0..30 ms arrival jitter) drawn by an EntityLayer on a virtual 60 fps clock:
##   anim_walk_jump_p95    p95 of per-frame position jump of walking entities / (frame_dt / speed_ms) (<= 1.2)
##   anim_walk_stall_pct   % of walking frames that moved under half the expected distance (WARN only, < 5)
##   anim_clock_drift_p95  |render clock - (newest tick - delay)| in ticks, p95 (< 1)
##   anim_atk_sync_frames  max |frame the damage number appears - frame the attacker's atk frame shows| (<= 1)
##   anim_fps_p95          real fps while the 200 animated entities draw (>= 55; WARN when headless)
## Prints "LAB <name> value=<n> target=<n> PASS|WARN|FAIL" per metric and "LAB summary pass= warn= fail=".
## Exits non-zero on FAIL. If the Part C net classes are missing it still prints a summary (WARN rows + reason).
## Embedded use (sims): set `embedded = true` (and a smaller `snap_count`) before add_child; no quit is called.
## Embedded runs skip the animation rows unless `anim_rows = true` (sim_lab_smoke counts five rows).

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
const ANIM_TICK_MS: int = 100
const ANIM_FRAMES: int = 900  # 15 s of virtual time at 60 fps
const ANIM_FRAME_MS: float = 1000.0 / 60.0
const ANIM_JITTER_MS: int = 30
const ANIM_WARMUP_FRAMES: int = 30
const T_WALK_JUMP: float = 1.2
const T_STALL_PCT: float = 5.0
const T_DRIFT_TICKS: float = 1.0
const T_ATK_FRAMES: float = 1.0
const LOWEST_PRIORITY: int = 2147483647

var embedded: bool = false
var anim_rows: bool = false  ## embedded runs add the animation rows only when set
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
	if miss.is_empty() and (anim_rows or not embedded):
		var an: Dictionary = await _measure_anim()
		_add("anim_walk_jump_p95", float(an["jump_p95"]), T_WALK_JUMP, true, true, false)
		_add("anim_walk_stall_pct", float(an["stall_pct"]), T_STALL_PCT, true, false, true)
		_add("anim_clock_drift_p95", float(an["drift_p95"]), T_DRIFT_TICKS, true, false, false)
		_add("anim_atk_sync_frames", float(an["atk_max"]), T_ATK_FRAMES, true, true, false)
		_add("anim_fps_p95", float(an["fps"]), T_FPS, false, false, headless)
		if str(an.get("note", "")) != "":
			reason = (reason + "; " if reason != "" else "") + str(an["note"])
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


## Animation metrics: LabAnimFeed snaps -> FakeLink (jittered arrivals) -> NetSession -> EntityLayer on a virtual
## 60 fps clock, measured per frame. See the header for the rows.
func _measure_anim() -> Dictionary:
	var out: Dictionary = {"jump_p95": -1.0, "stall_pct": -1.0, "drift_p95": -1.0, "atk_max": -1.0, "fps": -1.0}
	var feed := LabAnimFeed.new(SEED, ANIM_TICK_MS)
	var link := FakeLink.new()
	link.connect_to("fake://lab-anim")
	var session := NetSession.new(link)
	link.push_inbound(_anim_welcome_text())
	session.poll()
	if not session.welcomed:
		out["note"] = "anim welcome did not decode: %s" % session.last_error
		return out
	var layer := EntityLayer.new()
	layer.auto_refresh = false
	layer.tick_ms = float(ANIM_TICK_MS)
	layer.store = session.store
	layer.position = Vector2(48.0, 40.0)
	add_child(layer)
	var tile_px: float = ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)
	var pending: Array[Dictionary] = []
	var now: float = 0.0
	var next_tick_ms: float = 0.0
	var last_arrival: float = 0.0
	var patched: int = 0
	var ratios := PackedFloat64Array()
	var stalls: int = 0
	var drifts := PackedFloat64Array()
	var last_pos: Dictionary = {}
	var was_moving: Dictionary = {}
	var atk_frame_of: Dictionary = {}  ## "pid:start_tick" -> frame number the atk frame first showed
	var pop_frame_of: Dictionary = {}  ## "pid:start_tick" -> frame number the number first showed
	var seen_pops: Dictionary = {}
	var real_ms := PackedFloat64Array()
	var split_us: PackedInt64Array = [0, 0, 0]  ## net, layer, metrics
	var last_us: int = Time.get_ticks_usec()
	var atk_frame: int = EntityLayer.atk_frame_for(int(PlaceholderClips.clip(ClipSelect.ATTACK, true)["frames"]),
			PlaceholderClips.ATK_FRAME)
	for f: int in range(ANIM_FRAMES):
		now += ANIM_FRAME_MS
		while next_tick_ms <= now:
			var arrival: float = maxf(next_tick_ms + float(_rng.randi_range(0, ANIM_JITTER_MS)), last_arrival)
			last_arrival = arrival
			pending.append({"at": arrival, "snap": feed.step()})
			next_tick_ms += float(ANIM_TICK_MS)
		var arrived: bool = false
		var s0: int = Time.get_ticks_usec()
		while not pending.is_empty() and float(pending[0]["at"]) <= now:
			var snap: Dictionary = pending.pop_front()["snap"]
			link.push_inbound(Codec.encode(snap))
			session.poll()
			patched += _lab_patch_anim(session.store, snap)
			arrived = true
		var s1: int = Time.get_ticks_usec()
		layer.refresh(now)
		var s2: int = Time.get_ticks_usec()
		var render: float = layer.last_render_ms
		if f >= ANIM_WARMUP_FRAMES and arrived:
			drifts.append(absf(layer.clock.drift_ticks(now)))
		for w: Dictionary in feed.walkers:
			var id: int = int(w["id"])
			var tr: EntityTrack = layer.track_of(id)
			var sp: EntitySprite = layer.sprite_of(id)
			if tr == null or sp == null:
				continue
			var pos: Vector2 = sp.position / tile_px
			var moving: bool = tr.cells.moving(render)
			if f >= ANIM_WARMUP_FRAMES and last_pos.has(id) and moving and bool(was_moving.get(id, false)):
				var ratio: float = pos.distance_to(last_pos[id]) / (ANIM_FRAME_MS / feed.walker_speed(id))
				ratios.append(ratio)
				if ratio < 0.5:
					stalls += 1
			last_pos[id] = pos
			was_moving[id] = moving
		for p: Dictionary in feed.pairs:
			var pid: int = int(p["pid"])
			var sp: EntitySprite = layer.sprite_of(pid)
			var tr: EntityTrack = layer.track_of(pid)
			if sp == null or tr == null:
				continue
			var key: String = "%d:%d" % [pid, tr.start_tick]
			if sp.clip_name == "attack" and sp.frame >= atk_frame and not atk_frame_of.has(key):
				atk_frame_of[key] = f
		for pop: Dictionary in layer.shown_pops:
			var seq: int = int(pop["seq"])
			if seen_pops.has(seq):
				continue
			seen_pops[seq] = true
			var pid: int = int(pop["id"]) - LabAnimFeed.FIRST_TARGET_ID + LabAnimFeed.FIRST_PLAYER_ID
			var atr: EntityTrack = layer.track_of(pid)
			if atr != null:
				pop_frame_of["%d:%d" % [pid, atr.start_tick]] = f
		split_us[0] += s1 - s0
		split_us[1] += s2 - s1
		split_us[2] += Time.get_ticks_usec() - s2
		await get_tree().process_frame
		var us: int = Time.get_ticks_usec()
		if f >= ANIM_WARMUP_FRAMES:
			real_ms.append(float(us - last_us) / 1000.0)
		last_us = us
	var worst: float = -1.0
	var matched: int = 0
	for key: String in pop_frame_of:
		if atk_frame_of.has(key):
			matched += 1
			worst = maxf(worst, absf(float(int(pop_frame_of[key]) - int(atk_frame_of[key]))))
	if not ratios.is_empty():
		out["jump_p95"] = DebugFrame.percentile(ratios, 95.0)
		out["stall_pct"] = 100.0 * float(stalls) / float(ratios.size())
	if not drifts.is_empty():
		out["drift_p95"] = DebugFrame.percentile(drifts, 95.0)
	out["atk_max"] = worst
	if not real_ms.is_empty():
		out["fps"] = DebugFrame.fps_from_ms(DebugFrame.percentile(real_ms, 95.0))
	var notes := PackedStringArray()
	notes.append("anim: %d walk samples, %d swings matched of %d numbers" % [ratios.size(), matched, seen_pops.size()])
	var total_ms: float = 0.0
	for m: float in real_ms:
		total_ms += m
	notes.append("anim ms/frame: net %.2f layer %.2f metrics %.2f frame %.2f" % [split_us[0] / 1000.0 / ANIM_FRAMES,
			split_us[1] / 1000.0 / ANIM_FRAMES, split_us[2] / 1000.0 / ANIM_FRAMES, total_ms / maxf(real_ms.size(), 1.0)])
	if patched > 0:
		notes.append("EntityStore drops anim fields a/d/t/m/s/h/job, lab copied them back into %d rows (lead line pending)" % patched)
	out["note"] = "; ".join(notes)
	layer.queue_free()
	return out


## Copies anim fields the store did not keep back into its rows (store rows are shared dictionaries).
## Returns the rows patched (0 once EntityStore keeps a/d/t/m/s/h itself).
func _lab_patch_anim(store: EntityStore, snap: Dictionary) -> int:
	var n: int = 0
	for e: Variant in snap.get("ents", []):
		var src: Dictionary = e
		var row: Dictionary = store.get_ent(int(src["id"]))
		if row.is_empty():
			continue
		var did: bool = false
		for k: String in ["a", "d", "t", "m", "s", "h", "job"]:
			if src.has(k) and not row.has(k):
				row[k] = src[k]
				did = true
		if did:
			n += 1
	return n


func _anim_welcome_text() -> String:
	var terrain := PackedByteArray()
	terrain.resize(64 * 64)
	var packed: String = Marshalls.raw_to_base64(terrain.compress(FileAccess.COMPRESSION_DEFLATE))
	return Codec.encode({"t": "welcome", "slot": 0, "map": 64, "tick_ms": ANIM_TICK_MS, "view_w": 61, "view_h": 33,
			"terrain": packed, "pz": packed})
