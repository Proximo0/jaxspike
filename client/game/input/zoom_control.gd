class_name ZoomControl
extends RefCounted
## Camera zoom with bounds and smooth wheel steps: each wheel notch multiplies the target by STEP (clamped to
## [min_zoom, max_zoom]) and `update(delta)` eases the current zoom toward it exponentially. Pure: takes delta.

const DEFAULT_MIN: float = 0.6
const DEFAULT_MAX: float = 2.0
const STEP: float = 1.12  ## Target factor per wheel notch.
const RATE: float = 12.0  ## Ease rate per second (about 90 % of the way in 0.2 s).
const SNAP_EPS: float = 0.001

var min_zoom: float = DEFAULT_MIN
var max_zoom: float = DEFAULT_MAX
var target: float = 1.0
var current: float = 1.0


func _init(start: float = 1.0, lo: float = DEFAULT_MIN, hi: float = DEFAULT_MAX) -> void:
	min_zoom = minf(lo, hi)
	max_zoom = maxf(lo, hi)
	target = clampf(start, min_zoom, max_zoom)
	current = target


## `notches` > 0 zooms in (bigger tiles), < 0 out. Returns the new target.
func wheel(notches: float) -> float:
	target = clampf(target * pow(STEP, notches), min_zoom, max_zoom)
	return target


## Eases `current` toward `target`; returns true while it is still moving.
func update(delta: float) -> bool:
	if is_equal_approx(current, target):
		current = target
		return false
	var k: float = 1.0 - exp(-RATE * maxf(delta, 0.0))
	current = lerpf(current, target, k)
	if absf(current - target) < SNAP_EPS:
		current = target
	return current != target


func snap(z: float) -> void:
	target = clampf(z, min_zoom, max_zoom)
	current = target


func vec() -> Vector2:
	return Vector2(current, current)
