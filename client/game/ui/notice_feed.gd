class_name NoticeFeed
extends RefCounted
## Turns the store's per-snap notice string into events: one per snap that carries text, plus a timed HUD line.

const HOLD_S: float = 4.0

var _last_snaps: int = -1
var _line: String = ""
var _left: float = 0.0


## Feed every frame. Returns the notice when this call sees a new server notice (a new store.snaps with a non-empty
## notice), else "". The same text in two different snaps counts twice; one snap seen on many frames counts once.
func observe(store_snaps: int, notice: String) -> String:
	if store_snaps == _last_snaps:
		return ""
	_last_snaps = store_snaps
	if notice == "":
		return ""
	_line = notice
	_left = HOLD_S
	return notice


## HUD line: the last notice for HOLD_S seconds of `delta` time, then "". Call once per frame after observe.
func hud_line(delta: float) -> String:
	if _left <= 0.0:
		return ""
	_left -= delta
	if _left <= 0.0:
		_line = ""
	return _line
