class_name PlayerInput
extends Node
## Keyboard and mouse -> NetSession calls. WASD/arrows move, Space attack, E eat, left click walk_to, Esc quit.
## Reads physical keys directly so it needs no InputMap actions; the camera gives the click -> tile conversion.
## Feel: the mouse wheel zooms the camera smoothly within bounds (ZoomControl, applied to `camera` every frame)
## and the cursor shows the RO state of what is under it (CursorState: walk/attack/talk/pickup/blocked), using
## `entity_layer` (optional) for the entity under the mouse.

signal quit_requested
signal walk_sent
signal walk_target(tile: Vector2i)  ## A walk_to was sent for this tile.
signal walk_blocked(tile: Vector2i)  ## A click on a blocked tile; nothing was sent.

const PICK_TILES: float = 0.75  ## Hover pick radius, in tiles (as the skill bar's click pick).

var session: NetSession = null
var enabled: bool = true
var camera: Camera2D = null
var state: InputState = InputState.new()
## Optional: called with the click's world position before a walk; returning true consumes the click (entity select).
var click_filter: Callable = Callable()
var entity_layer: EntityLayer = null  ## Optional: enables entity cursor states.
var zoom: ZoomControl = ZoomControl.new(ViewMath.tuned("camera_zoom", ViewMath.DEFAULT_CAMERA_ZOOM))
var cursor: int = CursorState.WALK
var cursor_changes: int = 0
var _cursor_tex: Dictionary = {}  ## state -> ImageTexture


func _process(delta: float) -> void:
	if camera != null:
		zoom.update(delta)
		camera.zoom = zoom.vec()
	if session == null or not enabled:
		return
	_update_cursor()
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
		if m.pressed and (m.button_index == MOUSE_BUTTON_WHEEL_UP or m.button_index == MOUSE_BUTTON_WHEEL_DOWN):
			if get_viewport().gui_get_hovered_control() == null:
				var f: float = m.factor if m.factor > 0.0 else 1.0
				zoom.wheel(f if m.button_index == MOUSE_BUTTON_WHEEL_UP else -f)
				get_viewport().set_input_as_handled()
			return
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


## Category of the entity drawn nearest to `world_pos` within PICK_TILES tiles (the own player skipped), or
## CursorState.NO_ENTITY.
func hover_category(world_pos: Vector2) -> int:
	if entity_layer == null or session == null:
		return CursorState.NO_ENTITY
	var tile_px: float = ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)
	var best_d: float = tile_px * PICK_TILES
	var best: int = CursorState.NO_ENTITY
	for id: int in session.store.ids():
		if id == session.slot:
			continue
		var p: Vector2 = entity_layer.world_pos_of(id)
		if p == Vector2.INF:
			continue
		var d: float = p.distance_to(world_pos)
		if d < best_d:
			best_d = d
			var e: Dictionary = session.store.get_ent(id)
			best = ViewMath.entity_category(int(e.get("kind", 0)), e.has("item"))
	return best


func _update_cursor() -> void:
	var state: int = CursorState.WALK
	if camera != null and get_viewport() != null and get_viewport().gui_get_hovered_control() == null:
		var wp: Vector2 = camera.get_global_mouse_position()
		var tile: Vector2i = click_tile(get_viewport().get_mouse_position())
		state = CursorState.pick(hover_category(wp), session.map.at(tile.x, tile.y))
	set_cursor(state)


## Swaps the arrow cursor image when the state changes.
func set_cursor(state: int) -> void:
	if state == cursor and cursor_changes > 0:
		return
	cursor = state
	cursor_changes += 1
	if DisplayServer.get_name() == "headless":
		return
	if not _cursor_tex.has(state):
		_cursor_tex[state] = ImageTexture.create_from_image(CursorState.make_image(state))
	Input.set_custom_mouse_cursor(_cursor_tex[state], Input.CURSOR_ARROW, CursorState.hotspot(state))


func _exit_tree() -> void:
	if cursor_changes > 0 and DisplayServer.get_name() != "headless":
		Input.set_custom_mouse_cursor(null)
