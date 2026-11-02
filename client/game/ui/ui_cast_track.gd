class_name UiCastTrack
extends RefCounted
## Remembers the first `cast_left` seen for a cast, so the bar can fill: total = that value, filled = 1 - left/total.

var skill: int = 0
var total: int = 0
var left: int = 0


## Feed `you.casting` and `you.cast_left` every snapshot (or frame).
func update(casting: int, cast_left: int) -> void:
	if casting <= 0 or cast_left <= 0:
		skill = 0
		total = 0
		left = 0
		return
	if casting != skill or cast_left > total:
		skill = casting
		total = cast_left
	left = cast_left


func active() -> bool:
	return skill > 0


## 0 at the start of the cast, 1 when it resolves.
func fraction() -> float:
	if total <= 0:
		return 0.0
	return clampf(1.0 - float(left) / float(total), 0.0, 1.0)
