class_name RoThemeDemo
extends Control
## Demo of the RO theme: titled panel, buttons, item list, Hp/Sp/Exp bars, line edit.

const PANEL_POS := Vector2(40.0, 40.0)
const PANEL_SIZE := Vector2(480.0, 360.0)
const BAR_VALUES: Array[float] = [70.0, 40.0, 15.0]
const BAR_VARIATIONS: Array[String] = ["HpBar", "SpBar", "ExpBar"]
const BAR_NAMES: Array[String] = ["Hp", "Sp", "Exp"]

var focused_button: Button = null


func _ready() -> void:
	var panel := Panel.new()
	panel.position = PANEL_POS
	panel.size = PANEL_SIZE
	add_child(panel)
	var box := VBoxContainer.new()
	box.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT, Control.PRESET_MODE_MINSIZE, 8)
	panel.add_child(box)
	var title := Panel.new()
	title.theme_type_variation = &"TitleBar"
	title.custom_minimum_size = Vector2(0.0, 22.0)
	var title_label := Label.new()
	title_label.text = "Status"
	title_label.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT, Control.PRESET_MODE_MINSIZE, 4)
	title.add_child(title_label)
	box.add_child(title)
	var row := HBoxContainer.new()
	box.add_child(row)
	var names: Array[String] = ["Normal", "Disabled", "Focused"]
	for i: int in names.size():
		var b := Button.new()
		b.text = names[i]
		b.disabled = i == 1
		row.add_child(b)
		if i == 2:
			focused_button = b
	var list := ItemList.new()
	list.custom_minimum_size = Vector2(0.0, 130.0)
	var rows: Array[String] = ["Red Potion x5", "Blue Potion x2", "Knife", "Cotton Shirt", "Poring Card"]
	for r: String in rows:
		list.add_item(r)
	list.select(1)
	box.add_child(list)
	for i: int in BAR_VARIATIONS.size():
		var bar := ProgressBar.new()
		bar.theme_type_variation = StringName(BAR_VARIATIONS[i])
		bar.min_value = 0.0
		bar.max_value = 100.0
		bar.value = BAR_VALUES[i]
		bar.show_percentage = true
		bar.custom_minimum_size = Vector2(0.0, 16.0)
		bar.name = BAR_NAMES[i] + "Bar"
		box.add_child(bar)
	var edit := LineEdit.new()
	edit.placeholder_text = "Type a message"
	box.add_child(edit)
	focused_button.grab_focus.call_deferred()
