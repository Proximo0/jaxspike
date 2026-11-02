class_name TuningPanel
extends CanvasLayer
## F2 tuning panel: one row per Tuning.FIELDS entry (name, slider, value), live set_value on the shared Tuning,
## Reset, and Save/Load preset to user://tuning.json. Hidden by default.

const LAYER: int = 124
const PRESET_PATH: String = "user://tuning.json"
const FONT_SIZE: int = 14
const PLATE_COLOR := Color(0.05, 0.06, 0.09, 0.92)
const GROUP_COLOR := Color(0.95, 0.78, 0.35)
const SLIDER_W: float = 180.0

var tuning: Tuning = null
var preset_path: String = PRESET_PATH

var _rows: Dictionary = {}  ## field -> {slider: HSlider, value: Label}
var _status: Label = null
var _syncing: bool = false


func _init() -> void:
	layer = LAYER
	visible = false


func _ready() -> void:
	if tuning == null:
		tuning = Tuning.shared()
	_build()
	tuning.changed.connect(_on_tuning_changed)


func _unhandled_input(event: InputEvent) -> void:
	var key := event as InputEventKey
	if key != null and key.pressed and not key.echo and key.physical_keycode == KEY_F2:
		visible = not visible
		get_viewport().set_input_as_handled()


func row_count() -> int:
	return _rows.size()


## Slider position for a field (tests and tools).
func slider_value(field: String) -> float:
	if not _rows.has(field):
		return NAN
	return ((_rows[field] as Dictionary)["slider"] as HSlider).value


func reset_all() -> void:
	tuning.reset()
	_sync_all()
	_status.text = "reset to defaults"


func save_preset() -> void:
	var err: Error = tuning.save_json(preset_path)
	_status.text = "saved %s" % preset_path if err == OK else "save failed (%d)" % err


func load_preset() -> void:
	var ok: bool = tuning.load_json(preset_path)
	_sync_all()
	_status.text = "loaded %s" % preset_path if ok else "no preset at %s" % preset_path


func _build() -> void:
	var plate := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = PLATE_COLOR
	style.set_content_margin_all(8.0)
	plate.add_theme_stylebox_override("panel", style)
	plate.anchor_left = 1.0
	plate.anchor_right = 1.0
	plate.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	plate.offset_top = 8.0
	plate.offset_right = -8.0
	add_child(plate)
	var box := VBoxContainer.new()
	plate.add_child(box)
	var title := Label.new()
	title.text = "Tuning (F2)"
	title.add_theme_font_size_override("font_size", FONT_SIZE + 2)
	box.add_child(title)
	var grid := GridContainer.new()
	grid.columns = 3
	box.add_child(grid)
	var group: String = ""
	for f: Array in Tuning.FIELDS:
		if str(f[5]) != group:
			group = str(f[5])
			var g := Label.new()
			g.text = group
			g.add_theme_color_override("font_color", GROUP_COLOR)
			g.add_theme_font_size_override("font_size", FONT_SIZE)
			grid.add_child(g)
			grid.add_child(Control.new())
			grid.add_child(Control.new())
		_add_row(grid, f)
	var buttons := HBoxContainer.new()
	box.add_child(buttons)
	buttons.add_child(_button("Reset", reset_all))
	buttons.add_child(_button("Save", save_preset))
	buttons.add_child(_button("Load", load_preset))
	_status = Label.new()
	_status.add_theme_font_size_override("font_size", FONT_SIZE)
	box.add_child(_status)
	_sync_all()


func _add_row(grid: GridContainer, f: Array) -> void:
	var name_label := Label.new()
	name_label.text = str(f[0])
	name_label.tooltip_text = str(f[6])
	name_label.mouse_filter = Control.MOUSE_FILTER_PASS
	name_label.add_theme_font_size_override("font_size", FONT_SIZE)
	grid.add_child(name_label)
	var slider := HSlider.new()
	slider.min_value = float(f[2])
	slider.max_value = float(f[3])
	slider.step = float(f[4])
	slider.custom_minimum_size = Vector2(SLIDER_W, 0.0)
	slider.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	slider.value_changed.connect(_on_slider.bind(str(f[0])))
	grid.add_child(slider)
	var value_label := Label.new()
	value_label.custom_minimum_size = Vector2(56.0, 0.0)
	value_label.add_theme_font_size_override("font_size", FONT_SIZE)
	grid.add_child(value_label)
	_rows[str(f[0])] = {"slider": slider, "value": value_label}


func _button(text: String, action: Callable) -> Button:
	var b := Button.new()
	b.text = text
	b.pressed.connect(action)
	return b


func _on_slider(value: float, field: String) -> void:
	if _syncing:
		return
	tuning.set_value(field, value)


func _on_tuning_changed(field: String, _value: float) -> void:
	_sync_field(field)


func _sync_all() -> void:
	for field: String in _rows:
		_sync_field(field)


func _sync_field(field: String) -> void:
	if not _rows.has(field):
		return
	var row: Dictionary = _rows[field]
	var v: float = float(tuning.get(field))
	_syncing = true
	(row["slider"] as HSlider).value = v
	_syncing = false
	(row["value"] as Label).text = "%.2f" % v
