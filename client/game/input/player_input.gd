class_name PlayerInput
extends Node
## Keyboard and mouse -> NetSession calls. WASD/arrows move, Space attack, E eat, left click walk_to, Esc quit.
## Reads physical keys directly so it needs no InputMap actions; the camera gives the click -> tile conversion.

signal quit_requested
signal walk_sent
signal walk_target(tile: Vector2i)  ## A walk_to was sent for this tile.
signal walk_blocked(tile: Vector2i)  ## A click on a blocked tile; nothing was sent.

var session: NetSession = null
var enabled: bool = true
var camera: Camera2D = null
var state: InputState = InputState.new()
## Optional: called with the click's world position before a walk; returning true consumes the click (entity select).
var click_filter: Callable = Callable()


func _process(_delta: float) -> void:
	if session == null or not enabled:
		return
	var changed: bool = state.update(
			_key(KEY_W) or _key(KEY_UP), _key(KEY_S) or _key(KEY_DOWN),
			_key(KEY_A) or _key(KEY_LEFT), _key(KEY_D) or _key(KEY_RIGHT),
			_key(KEY_SPACE), _key(KEY_E))
	if changed:
		session.send_input(state.move, state.attack, state.eat)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey:
		var k: InputEventKey = event
		if k.pressed and not k.echo and k.physical_keycode == KEY_ESCAPE:
			quit_requested.emit()
			get_viewport().set_input_as_handled()
		return
	if session == null or not enabled or camera == null:
		return
	if event is InputEventMouseButton:
		var m: InputEventMouseButton = event
		if m.pressed and m.button_index == MOUSE_BUTTON_LEFT:
			if get_viewport().gui_get_hovered_control() != null:
				return  # a panel is under the mouse
			if click_filter.is_valid() and bool(click_filter.call(camera.get_global_mouse_position())):
				return
			click_walk(click_tile(m.position))


## Tile under a viewport pixel, from the camera centre and zoom.
func click_tile(screen_px: Vector2) -> Vector2i:
	var tile_px: float = ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)
	return ViewMath.screen_to_tile(screen_px, camera.get_screen_center_position(), camera.zoom.x,
			get_viewport().get_visible_rect().size, tile_px)


func _key(code: Key) -> bool:
	return Input.is_physical_key_pressed(code)


## Handles a click on `tile`: blocked terrain only flashes (walk_blocked), other in-map tiles send walk_to.
## Returns true when walk_to was sent.
func click_walk(tile: Vector2i) -> bool:
	if session == null or not ViewMath.in_map(tile, session.map.size):
		return false
	if ViewMath.is_blocked_tile(session.map, tile):
		walk_blocked.emit(tile)
		return false
	session.walk_to(tile.x, tile.y)
	walk_sent.emit()
	walk_target.emit(tile)
	return true
