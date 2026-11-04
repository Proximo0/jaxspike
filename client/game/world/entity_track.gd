class_name EntityTrack
extends RefCounted
## Per-entity playback state (round2 contract 1.3). Pure: callers pass times.
## `feed` takes an entity row from the store at server tick `tick` and the render clock's server time; the
## position runs through a CellLerp (cell to cell over speed_ms), the clip state comes from the row's anim fields
## (a action, d face, t start_tick, m dur_ms, s speed_ms). Missing fields keep their last value; a row that never
## carried `a` is animated from its motion (walk while a step is in flight, else idle; face from the step).
##
## The older tick-interval interpolation (`set_target` / `sample` / `settled`) stays for callers without a clock.

var prev: Vector2 = Vector2.ZERO
var target: Vector2 = Vector2.ZERO
var start_ms: float = 0.0
var duration_ms: float = 0.0
var _seen: bool = false

var cells: CellLerp = CellLerp.new()
var has_anim: bool = false  ## the server sent `a` at least once
var action: int = ClipSelect.IDLE
var face: int = 0
var start_tick: int = 0
var dur_ms: float = 0.0
var speed_ms: float = 0.0  ## 0 = unknown (the caller's fallback is used)
var hit_dmg: int = 0  ## `h` of the newest feed (0 = no hit in that row)
var flinch_ms: float = -INF  ## server ms a hurt clip really starts (the attacker's atk frame), set by the layer
var prev_action: int = ClipSelect.IDLE  ## action before the newest one: drawn until the render clock reaches it
var prev_start_tick: int = 0
var prev_dur_ms: float = 0.0
var _last_hit_tick: int = -1


## Feeds a store row seen at server tick `tick`. Returns the damage number to show (0 = none): `h` when present
## on a newer tick than the last one shown.
func feed(e: Dictionary, tick: int, tick_ms: float, now_ms: float, fallback_speed_ms: float) -> int:
	if e.has("s"):
		speed_ms = float(e["s"])
	var new_action: int = ClipSelect.clamp_action(int(e["a"])) if e.has("a") else action
	var new_start: int = int(e["t"]) if e.has("t") else start_tick
	if has_anim and (new_action != action or new_start != start_tick):
		prev_action = action
		prev_start_tick = start_tick
		prev_dur_ms = dur_ms
		flinch_ms = -INF
	if e.has("a"):
		has_anim = true
	action = new_action
	start_tick = new_start
	if e.has("d"):
		face = ClipSelect.clamp_face(int(e["d"]))
	if e.has("m"):
		dur_ms = float(e["m"])
	var spd: float = speed_ms if speed_ms > 0.0 else fallback_speed_ms
	var before: Vector2 = cells.cell()
	cells.push(ViewMath.to_vec(e.get("pos")), tick, spd, tick_ms, now_ms)
	if not e.has("d"):
		face = ClipSelect.face_of(cells.cell() - before, face)
	hit_dmg = 0
	if e.has("h") and int(e["h"]) != 0 and tick > _last_hit_tick:
		_last_hit_tick = tick
		hit_dmg = int(e["h"])
	return hit_dmg


## Drawn position (cells) at server time `now_ms`.
func pos_at(now_ms: float) -> Vector2:
	return cells.sample(now_ms)


var _clip: Dictionary = {"action": 0, "face": 0, "elapsed_ms": 0.0, "dur_ms": 0.0, "speed_ms": 0.0}  ## pooled result of clip_at

## Clip to draw at server time `now_ms`: {action, face, elapsed_ms, dur_ms, speed_ms}. Without anim fields the
## clip follows the motion; with them, a walk whose steps ran out shows idle (the server's idle row is a tick late).
## The returned Dictionary is pooled and overwritten by the next call: callers must read it at once, never keep it.
func clip_at(now_ms: float, tick_ms: float, fallback_speed_ms: float) -> Dictionary:
	var spd: float = speed_ms if speed_ms > 0.0 else fallback_speed_ms
	var a: int = action
	var begin: float = float(start_tick) * tick_ms
	if a == ClipSelect.HURT and flinch_ms > begin:
		begin = flinch_ms
	var elapsed: float = now_ms - begin
	var dur: float = dur_ms
	if has_anim and now_ms < begin:  # the render clock has not reached the newest action yet
		a = prev_action
		elapsed = now_ms - float(prev_start_tick) * tick_ms
		dur = prev_dur_ms
	var f: int = face
	var mv: bool = false
	if not has_anim or a == ClipSelect.WALK or a == ClipSelect.IDLE or a == ClipSelect.STANDBY:
		mv = cells.moving(now_ms)
	if not has_anim:
		a = ClipSelect.WALK if mv else ClipSelect.IDLE
		elapsed = now_ms
	elif a == ClipSelect.WALK or ((a == ClipSelect.IDLE or a == ClipSelect.STANDBY) and mv):
		# Walk is one continuous cycle across steps, phase-locked to server time; the drawn position lags the
		# server by the render delay, so legs keep moving while the last step is still being drawn.
		elapsed = now_ms
		a = ClipSelect.WALK if mv else ClipSelect.IDLE
	if a == ClipSelect.WALK:
		var sd: Vector2 = cells.step_delta(now_ms)
		if sd != Vector2.ZERO and not has_anim:
			f = ClipSelect.face_of(sd, face)
	_clip["action"] = a
	_clip["face"] = f
	_clip["elapsed_ms"] = elapsed
	_clip["dur_ms"] = dur
	_clip["speed_ms"] = spd
	return _clip


## Server ms the current action began.
func action_start_ms(tick_ms: float) -> float:
	return float(start_tick) * tick_ms


## Feed the newest tick position (tile units). The first call, a zero duration or a jump over `snap_dist` tiles
## (teleport, respawn) places the entity at once; an unchanged target keeps the running motion.
func set_target(pos: Vector2, now_ms: float, dur_ms_in: float, snap_dist: float = 3.0) -> void:
	if not _seen:
		_seen = true
		prev = pos
		target = pos
		start_ms = now_ms
		duration_ms = 0.0
		return
	if pos == target:
		return
	var here: Vector2 = sample(now_ms)
	prev = pos if (dur_ms_in <= 0.0 or here.distance_to(pos) > snap_dist) else here
	target = pos
	start_ms = now_ms
	duration_ms = dur_ms_in


func sample(now_ms: float) -> Vector2:
	if duration_ms <= 0.0:
		return target
	var t: float = clampf((now_ms - start_ms) / duration_ms, 0.0, 1.0)
	return prev.lerp(target, t)


func settled(now_ms: float) -> bool:
	return duration_ms <= 0.0 or now_ms - start_ms >= duration_ms
