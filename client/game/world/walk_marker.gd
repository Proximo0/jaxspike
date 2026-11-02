class_name WalkMarker
extends Node2D
## Click-to-walk feedback: a hollow square on the clicked tile. `show_target` while a walk is under way (hidden when
## the own player arrives or makes no progress for MARKER_STALL_TICKS server ticks); `flash_blocked` shows it red
## for a moment on a blocked tile. Call `observe(own_pos, tick)` with every new server tick.

var target: Vector2i = Vector2i.ZERO
var active: bool = false
var blocked_flash: bool = false

var _tile_px: float = ViewMath.DEFAULT_TILE_PX
var _stalled: int = 0
var _last_pos: Vector2i = Vector2i.ZERO
var _last_tick: int = -1
var _flash_left_ms: float = 0.0


func show_target(tile: Vector2i, own_pos: Vector2i, tile_px: float) -> void:
	target = tile
	active = true
	blocked_flash = false
	_flash_left_ms = 0.0
	_stalled = 0
	_last_pos = own_pos
	_place(tile_px)


func flash_blocked(tile: Vector2i, tile_px: float) -> void:
	target = tile
	active = true
	blocked_flash = true
	_flash_left_ms = ViewMath.MARKER_FLASH_MS
	_place(tile_px)


## Feeds the own position at server tick `tick`; counts ticks without progress and hides the marker when done.
func observe(own_pos: Vector2i, tick: int) -> void:
	if not active or blocked_flash or tick == _last_tick:
		_last_tick = tick
		return
	_last_tick = tick
	if own_pos != _last_pos:
		_last_pos = own_pos
		_stalled = 0
	else:
		_stalled += 1
	if ViewMath.marker_done(target, own_pos, _stalled):
		hide_marker()


func hide_marker() -> void:
	active = false
	blocked_flash = false
	visible = false


func _process(delta: float) -> void:
	if blocked_flash:
		_flash_left_ms -= delta * 1000.0
		if _flash_left_ms <= 0.0:
			hide_marker()


func _place(tile_px: float) -> void:
	_tile_px = tile_px
	position = Vector2(target) * tile_px
	visible = true
	z_index = 5
	queue_redraw()


func _draw() -> void:
	var col: Color = ViewMath.MARKER_BLOCKED_COLOR if blocked_flash else ViewMath.MARKER_COLOR
	var inset: float = maxf(2.0, _tile_px * 0.08)
	draw_rect(Rect2(Vector2(inset, inset), Vector2(_tile_px - inset * 2.0, _tile_px - inset * 2.0)), col, false, 2.0)
