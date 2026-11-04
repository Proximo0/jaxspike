class_name CameraFollow
extends RefCounted
## Decides when the camera must sit on the own tile instead of following the lerped sprite: after a warp (the server
## flag, or a jump over WARP_TILES in one observed tick) until the drawn sprite is within ARRIVE_TILES or HOLD_S ran out.
## Pure: takes delta and the observed tick, no clock.

const WARP_TILES: float = 2.0  ## A jump over this many tiles between two observed ticks is a warp.
const ARRIVE_TILES: float = 0.6
const HOLD_S: float = 1.0

var snaps: int = 0  ## Warps seen.
var _active: bool = false
var _held: float = 0.0
var _have_tick: bool = false
var _last_tick: int = 0
var _last_tile: Vector2i = Vector2i.ZERO


func observe(tile: Vector2i, tick: int, warped: bool, drawn_tile: Vector2, delta: float) -> bool:
	if not _have_tick or tick != _last_tick:
		var jumped: bool = _have_tick and Vector2(tile - _last_tile).length() > WARP_TILES
		if _have_tick and (warped or jumped):
			snaps += 1
			_active = true
			_held = 0.0
		_have_tick = true
		_last_tick = tick
		_last_tile = tile
	if not _active:
		return false
	_held += maxf(delta, 0.0)
	if _held >= HOLD_S or drawn_tile.distance_to(Vector2(tile)) <= ARRIVE_TILES:
		_active = false
		return false
	return true
