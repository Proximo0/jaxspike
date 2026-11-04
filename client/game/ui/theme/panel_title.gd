class_name PanelTitle
extends PanelContainer
## ro6-ui: the shared title bar of a window (inventory, shop, card album): the title on the left, optional extras
## (ryn), a close button on the right. Drag the bar to move the window; it stays at least KEEP_PX on screen.

const CLOSE_TEXT: String = "x"
const CLOSE_PX: Vector2 = Vector2(20, 18)
const KEEP_PX: float = 48.0
const BAR_PX: float = 22.0

var target: Control = null
var title_label: Label
var extras: HBoxContainer
var close_button: Button
var dragging: bool = false


## A title bar for `win`, added as the first child of `parent`; the close button calls `on_close`.
static func make(win: Control, parent: Node, title: String, on_close: Callable) -> PanelTitle:
	var t := PanelTitle.new()
	t.name = "TitleBar"
	t.target = win
	t.theme_type_variation = &"TitleBar"
	t.mouse_default_cursor_shape = Control.CURSOR_MOVE
	t.custom_minimum_size = Vector2(0, BAR_PX)
	var row := HBoxContainer.new()
	row.mouse_filter = Control.MOUSE_FILTER_IGNORE
	t.add_child(row)
	t.title_label = Label.new()
	t.title_label.text = title
	t.title_label.theme_type_variation = &"PanelTitle"
	t.title_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	t.title_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	row.add_child(t.title_label)
	t.extras = HBoxContainer.new()
	t.extras.mouse_filter = Control.MOUSE_FILTER_IGNORE
	row.add_child(t.extras)
	t.close_button = Button.new()
	t.close_button.name = "Close"
	t.close_button.text = CLOSE_TEXT
	t.close_button.theme_type_variation = &"CloseButton"
	t.close_button.custom_minimum_size = CLOSE_PX
	t.close_button.focus_mode = Control.FOCUS_NONE
	t.close_button.tooltip_text = "Close"
	t.close_button.pressed.connect(on_close)
	row.add_child(t.close_button)
	parent.add_child(t)
	parent.move_child(t, 0)
	return t


func _gui_input(event: InputEvent) -> void:
	var mb := event as InputEventMouseButton
	if mb != null and mb.button_index == MOUSE_BUTTON_LEFT:
		dragging = mb.pressed
		accept_event()
		return
	var mm := event as InputEventMouseMotion
	if mm != null and dragging and target != null:
		drag_by(mm.relative)
		accept_event()


## Moves the window by `delta` px, kept on screen.
func drag_by(delta: Vector2) -> void:
	if target == null:
		return
	target.position = clamp_position(target.position + delta, target.size, target.get_viewport_rect().size)


## A window position with at least KEEP_PX of the window and the whole title bar height inside `view`.
static func clamp_position(pos: Vector2, win_size: Vector2, view: Vector2) -> Vector2:
	var keep_x: float = minf(KEEP_PX, win_size.x)
	return Vector2(clampf(pos.x, keep_x - win_size.x, maxf(view.x - keep_x, 0.0)),
			clampf(pos.y, 0.0, maxf(view.y - BAR_PX, 0.0)))
