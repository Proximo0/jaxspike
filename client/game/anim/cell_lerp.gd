class_name CellLerp
extends RefCounted
## Cell-to-cell interpolation of one entity by its walk speed (round2 contract 1.3): a step takes `speed_ms`
## (x DIAGONAL for a diagonal step), not the tick interval, so 150 ms/cell walking is smooth at a 100 ms tick.
## Pure, in server milliseconds (RenderClock.render_ms): the caller passes the time.
##
## A step seen at tick T happened during (T-1, T], so it is scheduled to start at (T-1) * tick_ms, or when the
## previous step ends if that is later (steps chain without overlapping). A step scheduled before `now` starts at
## `now` (no jump), and a queue that falls more than CATCH_UP_TICKS behind its schedule is shortened so walking
## never lags further and further. A jump of more than SNAP_CELLS (teleport, respawn) places the entity at once.

const DIAGONAL: float = 1.4
const SNAP_CELLS: float = 3.0
const CATCH_UP_TICKS: float = 1.0
const MAX_SEGMENTS: int = 8

var _from: Array[Vector2] = []
var _to: Array[Vector2] = []
var _start: PackedFloat64Array = PackedFloat64Array()
var _end: PackedFloat64Array = PackedFloat64Array()
var _pos: Vector2 = Vector2.ZERO  ## newest cell
var _seen: bool = false


## Duration of one step from `a` to `b` at `speed_ms` (a multi-cell gap costs one step per cell of Chebyshev distance).
static func step_ms(a: Vector2, b: Vector2, speed_ms: float) -> float:
	var d: Vector2 = (b - a).abs()
	var straight: float = absf(d.x - d.y)
	var diag: float = minf(d.x, d.y)
	return speed_ms * (straight + diag * DIAGONAL)


## Feeds the newest cell, seen at server tick `tick`. `now_ms` is the server time being drawn now.
func push(cell: Vector2, tick: int, speed_ms: float, tick_ms: float, now_ms: float) -> void:
	if not _seen:
		_seen = true
		_pos = cell
		return
	if cell == _pos:
		return
	var here: Vector2 = sample(now_ms)
	if here.distance_to(cell) > SNAP_CELLS or speed_ms <= 0.0:
		_clear()
		_pos = cell
		return
	_prune(now_ms)
	var sched: float = float(tick - 1) * tick_ms
	var prev_end: float = _end[_end.size() - 1] if _end.size() > 0 else -INF
	var start: float = maxf(maxf(sched, prev_end), now_ms)
	var dur: float = step_ms(_pos, cell, speed_ms)
	var behind: float = start - sched
	if behind > CATCH_UP_TICKS * tick_ms:
		dur *= clampf(1.0 - (behind - CATCH_UP_TICKS * tick_ms) / maxf(dur, 1.0), 0.5, 1.0)
	_from.append(_pos)
	_to.append(cell)
	_start.append(start)
	_end.append(start + dur)
	_pos = cell
	if _end.size() > MAX_SEGMENTS:
		_drop_first()


## Drawn position (cells) at server time `now_ms`.
func sample(now_ms: float) -> Vector2:
	var n: int = _end.size()
	if n == 0:
		return _pos
	if now_ms <= _start[0]:
		return _from[0]
	for i: int in range(n):
		if now_ms < _end[i]:
			if now_ms <= _start[i]:
				return _from[i]
			var t: float = (now_ms - _start[i]) / maxf(_end[i] - _start[i], 0.001)
			return _from[i].lerp(_to[i], t)
	return _pos


## True while a step is in flight or queued at `now_ms`.
func moving(now_ms: float) -> bool:
	return _end.size() > 0 and now_ms < _end[_end.size() - 1]


## Newest cell fed.
func cell() -> Vector2:
	return _pos


## Cell delta of the step being drawn at `now_ms` (Vector2.ZERO when still), for facing fallbacks.
func step_delta(now_ms: float) -> Vector2:
	for i: int in range(_end.size()):
		if now_ms < _end[i]:
			return _to[i] - _from[i]
	return Vector2.ZERO


func pending() -> int:
	return _end.size()


func _prune(now_ms: float) -> void:
	while _end.size() > 0 and _end[0] <= now_ms:
		_drop_first()


func _drop_first() -> void:
	_from.remove_at(0)
	_to.remove_at(0)
	_start.remove_at(0)
	_end.remove_at(0)


func _clear() -> void:
	_from.clear()
	_to.clear()
	_start = PackedFloat64Array()
	_end = PackedFloat64Array()
