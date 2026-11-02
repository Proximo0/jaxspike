class_name EntityTrack
extends RefCounted
## Interpolation of one entity between ticks: from where it is drawn now to the newest tick position, over
## `duration_ms`. Pure: the caller passes `now_ms`.

var prev: Vector2 = Vector2.ZERO
var target: Vector2 = Vector2.ZERO
var start_ms: float = 0.0
var duration_ms: float = 0.0
var _seen: bool = false


## Feed the newest tick position (tile units). The first call, a zero duration or a jump over `snap_dist` tiles
## (teleport, respawn) places the entity at once; an unchanged target keeps the running motion.
func set_target(pos: Vector2, now_ms: float, dur_ms: float, snap_dist: float = 3.0) -> void:
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
	prev = pos if (dur_ms <= 0.0 or here.distance_to(pos) > snap_dist) else here
	target = pos
	start_ms = now_ms
	duration_ms = dur_ms


func sample(now_ms: float) -> Vector2:
	if duration_ms <= 0.0:
		return target
	var t: float = clampf((now_ms - start_ms) / duration_ms, 0.0, 1.0)
	return prev.lerp(target, t)


func settled(now_ms: float) -> bool:
	return duration_ms <= 0.0 or now_ms - start_ms >= duration_ms
