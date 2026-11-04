class_name UiCastTrack
extends RefCounted
## Remembers the first `cast_left` seen for a cast, so the bar can fill: total = that value, filled = 1 - left/total.
## An optional `now_ms` (fed by the Node) lets the bar glide between ticks: `left` counts down by the time since the
## last change, at most one tick.

var skill: int = 0
var total: int = 0
var left: int = 0
var shown_left: float = 0.0  ## `left` minus the interpolated part of the current tick.
var changed_ms: float = 0.0


## Feed `you.casting` and `you.cast_left` every snapshot (or frame). `now_ms` < 0 means no interpolation.
func update(casting: int, cast_left: int, now_ms: float = -1.0, tick_ms: int = 100) -> void:
	if casting <= 0 or cast_left <= 0:
		skill = 0
		total = 0
		left = 0
		shown_left = 0.0
		return
	if casting != skill or cast_left > total:
		skill = casting
		total = cast_left
	if cast_left != left or now_ms < 0.0:
		changed_ms = now_ms
	left = cast_left
	var t: float = 0.0
	if now_ms >= 0.0 and tick_ms > 0:
		t = clampf((now_ms - changed_ms) / float(tick_ms), 0.0, 1.0)
	shown_left = maxf(float(left) - t, 0.0)


func active() -> bool:
	return skill > 0


## 0 at the start of the cast, 1 when it resolves.
func fraction() -> float:
	if total <= 0:
		return 0.0
	return clampf(1.0 - shown_left / float(total), 0.0, 1.0)


## Seconds of cast left (interpolated when `now_ms` was fed).
func seconds_left(tick_ms: int) -> float:
	return shown_left * float(tick_ms) / 1000.0
