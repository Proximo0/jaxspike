extends Node
## Integration bot: logs in to a live server over WsLink, walks to random tiles, attacks adjacent monsters and
## prints "BOT score=<snaps> errors=<n> stalls=<n> moved=<tiles>" plus "BOT <what> FAIL <detail>" lines.
## Args after "--": --url=ws://host:port (absent: prints "BOT skipped no --url", exit 0), --seconds=30,
## --shard=0 --nshards=1, --seed=N. Run with --fixed-fps 60.

const WELCOME_TIMEOUT_S: float = 10.0
const FIRST_SNAP_TIMEOUT_S: float = 5.0
const WALK_INTERVAL_S: float = 2.0
const WALK_RADIUS: int = 20
const MAX_WALKABLE_TERRAIN: int = 3
const STALL_TICKS: int = 50
const MAX_STALLS: int = 3
const MONSTER_KIND_MIN: int = 1
const MONSTER_KIND_MAX: int = 99
const PICK_TRIES: int = 30
const ARRIVE_TILES: int = 1
const WALK_TIMEOUT_TICKS: int = 120
const HIT_WINDOW_TICKS: int = 10
const MIN_ATTACKS_FOR_HIT_CHECK: int = 5
const MAX_PATH_MISS_RATIO: float = 0.5

var _session: NetSession = NetSession.new(WsLink.new())
var _rng: RandomNumberGenerator = RandomNumberGenerator.new()
var _seconds: float = 30.0
var _shard: int = 0
var _start_ms: int = 0
var _logged_in: bool = false
var _welcome_ms: int = -1
var _next_walk_ms: int = 0
var _target: Vector2i = Vector2i(-1, -1)
var _walk_pending: bool = false
var _last_pos: Vector2i = Vector2i(-1, -1)
var _still_since_tick: int = -1
var _start_pos: Vector2i = Vector2i(-1, -1)
var _moved: int = 0
var _stalls: int = 0
var _errors: int = 0
var _reported_error: String = ""
var _fails: Array[String] = []
var _done: bool = false
var _walks: int = 0
var _path_miss: int = 0
var _walk_tick: int = 0
var _walk_notice: bool = false
var _attacks: int = 0
var _hits: int = 0
var _attack_miss: int = 0
var _probe_id: int = -1
var _probe_hp: int = 0
var _probe_tick: int = 0
var _kills: int = 0
var _kills_base: int = -1


func _ready() -> void:
	var url: String = Flags.get_arg("url")
	if url.is_empty():
		print("BOT skipped no --url")
		print("BOT score=0 errors=0 stalls=0 moved=0 walks=0 path_miss=0 attacks=0 hits=0 attack_miss=0 kills=0")
		_done = true
		get_tree().quit(0)
		return
	_seconds = float(Flags.get_arg("seconds", "30"))
	_shard = int(Flags.get_arg("shard", "0"))
	_rng.seed = Flags.seed_value() * 1000 + _shard
	_start_ms = Time.get_ticks_msec()
	_session.connect_to(url)
	if _session.link.state() == NetLink.STATE_CLOSED:
		_fail("connect could not start for " + url)
		_finish(1)


func _physics_process(_delta: float) -> void:
	if _done:
		return
	var now_ms: int = Time.get_ticks_msec()
	var elapsed_s: float = float(now_ms - _start_ms) / 1000.0
	var msgs: Array[Dictionary] = _session.poll()
	for m: Dictionary in msgs:
		if String(m["t"]) == "error":
			_errors += 1
			_reported_error = _session.last_error
		elif String(m["t"]) == "snap":
			var n: Variant = m.get("notice")
			if typeof(n) == TYPE_STRING and (n == "too far" or n == "blocked"):
				_walk_notice = true
	if not _logged_in and _session.link.state() == NetLink.STATE_OPEN:
		_session.login("bot%d" % _shard)
		_logged_in = true
	if not _session.welcomed:
		if elapsed_s > WELCOME_TIMEOUT_S:
			_fail("no welcome after %d s (last_error=%s)" % [int(WELCOME_TIMEOUT_S), _session.last_error])
			_finish(1)
		return
	if _welcome_ms < 0:
		_welcome_ms = now_ms
	if _session.store.snaps == 0:
		if float(now_ms - _welcome_ms) / 1000.0 > FIRST_SNAP_TIMEOUT_S:
			_fail("zero snaps %d s after welcome" % int(FIRST_SNAP_TIMEOUT_S))
			_finish(1)
		return
	_track_position()
	_check_walk()
	if now_ms >= _next_walk_ms and not _walk_pending:
		_next_walk_ms = now_ms + int(WALK_INTERVAL_S * 1000.0)
		_pick_and_walk()
	var adj: int = _adjacent_monster()
	if adj >= 0:
		_session.send_input(0, true, false)
	_track_attack(adj)
	_track_kills()
	if elapsed_s >= _seconds:
		_finish(_report())


func _track_position() -> void:
	var pos: Vector2i = _session.store.you.get("pos", _last_pos)
	var tick: int = _session.store.tick
	if _start_pos.x < 0:
		_start_pos = pos
		_last_pos = pos
		_still_since_tick = tick
		return
	if pos != _last_pos:
		_moved += absi(pos.x - _last_pos.x) + absi(pos.y - _last_pos.y)
		_last_pos = pos
		_still_since_tick = tick
		return
	if _walk_pending and _target != pos and tick - _still_since_tick >= STALL_TICKS:
		_stalls += 1
		_walk_pending = false
		if not _walk_notice:
			_path_miss += 1
		_still_since_tick = tick


func _pick_and_walk() -> void:
	var map: MapData = _session.map
	if not map.is_loaded() or _last_pos.x < 0:
		return
	for i: int in range(PICK_TRIES):
		var t := Vector2i(
			_last_pos.x + _rng.randi_range(-WALK_RADIUS, WALK_RADIUS),
			_last_pos.y + _rng.randi_range(-WALK_RADIUS, WALK_RADIUS))
		var terr: int = map.at(t.x, t.y)
		if terr >= 0 and terr <= MAX_WALKABLE_TERRAIN and t != _last_pos:
			_target = t
			_walk_pending = true
			_still_since_tick = _session.store.tick
			_walks += 1
			_walk_tick = _session.store.tick
			_walk_notice = false
			_session.walk_to(t.x, t.y)
			return


func _adjacent_monster() -> int:
	for id: int in _session.store.ids():
		var e: Dictionary = _session.store.get_ent(id)
		var kind: int = int(e.get("kind", 0))
		if kind < MONSTER_KIND_MIN or kind > MONSTER_KIND_MAX:
			continue
		var p: Vector2i = e["pos"]
		if absi(p.x - _last_pos.x) + absi(p.y - _last_pos.y) == 1:
			return id
	return -1


## Scores a walk once: arrived within ARRIVE_TILES or a too-far/blocked notice is fine, a timeout is a path_miss.
func _check_walk() -> void:
	if not _walk_pending:
		return
	var d: int = maxi(absi(_target.x - _last_pos.x), absi(_target.y - _last_pos.y))
	if d <= ARRIVE_TILES or _walk_notice:
		_walk_pending = false
	elif _session.store.tick - _walk_tick >= WALK_TIMEOUT_TICKS:
		_walk_pending = false
		_path_miss += 1


## Probes whether attacking an adjacent monster lowers its hp (or removes it) within HIT_WINDOW_TICKS.
func _track_attack(adj: int) -> void:
	var tick: int = _session.store.tick
	if _probe_id >= 0:
		var e: Dictionary = _session.store.get_ent(_probe_id)
		if e.is_empty() or int(e.get("hp", 0)) < _probe_hp:
			_hits += 1
			_probe_id = -1
		elif tick - _probe_tick >= HIT_WINDOW_TICKS:
			_attack_miss += 1
			_probe_id = -1
	if _probe_id < 0 and adj >= 0:
		_probe_id = adj
		_probe_hp = int(_session.store.get_ent(adj).get("hp", 0))
		_probe_tick = tick
		_attacks += 1


func _track_kills() -> void:
	var k: int = int(_session.store.you.get("kills", 0))
	if _kills_base < 0:
		_kills_base = k
	_kills = k - _kills_base


func _report() -> int:
	if _errors > 0:
		_fail("%d error messages, last: %s" % [_errors, _reported_error])
	if _session.bad_messages > 0:
		_fail("%d undecodable messages" % _session.bad_messages)
	if _moved == 0:
		_fail("zero movement over the whole run")
	if _stalls > MAX_STALLS:
		_fail("%d stalls (max %d)" % [_stalls, MAX_STALLS])
	if _walks > 0 and float(_path_miss) > MAX_PATH_MISS_RATIO * float(_walks):
		_fail("path_miss %d of %d walks" % [_path_miss, _walks])
	if _attacks >= MIN_ATTACKS_FOR_HIT_CHECK and _hits == 0:
		_fail("0 hits in %d attacks (%d attack_miss)" % [_attacks, _attack_miss])
	return 1 if not _fails.is_empty() else 0


func _fail(detail: String) -> void:
	_fails.append(detail)
	print("BOT run FAIL %s" % detail)


func _finish(code: int) -> void:
	if _done:
		return
	_done = true
	print("BOT score=%d errors=%d stalls=%d moved=%d walks=%d path_miss=%d attacks=%d hits=%d attack_miss=%d kills=%d" % [
		_session.store.snaps, _errors, _stalls, _moved, _walks, _path_miss, _attacks, _hits, _attack_miss, _kills])
	_session.close()
	get_tree().quit(code)
