class_name CheatPanel
extends PanelContainer
## Playtest cheats (F5 toggles): level, job level, ryn, full heal, god mode, a gear kit, all skills, a teleport row
## with one button per town (welcome.world.towns) and an item id / count / refine row. Each button sends one
## {"t": "cheat", "op": ...} message (net/cheats.py); the server answers with a notice ("cheats are off" unless it
## runs with --cheats). The panel keeps no game state beyond the god-mode toggle it last sent.

signal message_sent(msg: Dictionary)

const TOGGLE_KEY: Key = KEY_F5
const TITLE: String = "Cheats (playtest)"
const LEVEL_STEP: int = 10
const LEVEL_TARGET: int = 50
const JOB_TARGET: int = 50
const MAX_BASE_LEVEL: int = 99
const RYN_GIFT: int = 100000
const MAX_ITEM_ID: int = 299
const MAX_STACK: int = 9999
const MAX_REFINE: int = 10

var session: NetSession = null
var godmode_on: bool = false
var title_bar: PanelTitle
var godmode_button: Button
var town_row: HFlowContainer
var item_spin: SpinBox
var count_spin: SpinBox
var refine_spin: SpinBox
var town_names: Array[String] = []


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	custom_minimum_size = Vector2(340, 0)
	var box := VBoxContainer.new()
	add_child(box)
	title_bar = PanelTitle.make(self, box, TITLE + " (F5)", close)
	var row1 := HFlowContainer.new()
	box.add_child(row1)
	_button(row1, "Level +%d" % LEVEL_STEP, func() -> void: send(level_message(current_level() + LEVEL_STEP, 0)))
	_button(row1, "Level %d" % LEVEL_TARGET, func() -> void: send(level_message(LEVEL_TARGET, 0)))
	_button(row1, "Job %d" % JOB_TARGET, func() -> void: send(level_message(0, JOB_TARGET)))
	var row2 := HFlowContainer.new()
	box.add_child(row2)
	_button(row2, "+%s Ryn" % group_digits(RYN_GIFT), func() -> void: send(ryn_message(RYN_GIFT)))
	_button(row2, "Full heal", func() -> void: send(simple_message("heal")))
	godmode_button = _button(row2, godmode_text(false), toggle_godmode)
	var row3 := HFlowContainer.new()
	box.add_child(row3)
	_button(row3, "Gear kit", func() -> void: send(simple_message("kit")))
	_button(row3, "All skills", func() -> void: send(simple_message("allskills")))
	var tl := Label.new()
	tl.text = "Teleport"
	box.add_child(tl)
	town_row = HFlowContainer.new()
	box.add_child(town_row)
	var il := Label.new()
	il.text = "Item id / count / refine"
	box.add_child(il)
	var row4 := HBoxContainer.new()
	box.add_child(row4)
	item_spin = _spin(row4, 1, MAX_ITEM_ID, 24)
	count_spin = _spin(row4, 1, MAX_STACK, 1)
	refine_spin = _spin(row4, 0, MAX_REFINE, 0)
	_button(row4, "Give", give_item)
	visible = false
	_rebuild_towns()


func _button(parent: Node, text: String, on_press: Callable) -> Button:
	var b := Button.new()
	b.text = text
	b.pressed.connect(on_press)
	parent.add_child(b)
	return b


func _spin(parent: Node, lo: int, hi: int, value: int) -> SpinBox:
	var s := SpinBox.new()
	s.min_value = lo
	s.max_value = hi
	s.step = 1
	s.value = value
	s.custom_minimum_size = Vector2(76, 0)
	parent.add_child(s)
	return s


func bind(s: NetSession) -> void:
	session = s


func set_welcome(msg: Dictionary) -> void:
	town_names = towns_from(msg)
	_rebuild_towns()


func on_message(_msg: Dictionary) -> void:
	pass


func close() -> void:
	visible = false


func _unhandled_key_input(event: InputEvent) -> void:
	var k := event as InputEventKey
	if k != null and k.pressed and not k.echo and k.keycode == TOGGLE_KEY:
		visible = not visible
		get_viewport().set_input_as_handled()


func _rebuild_towns() -> void:
	if town_row == null:
		return
	for c: Node in town_row.get_children():
		c.queue_free()
	for i: int in range(town_names.size()):
		_button(town_row, town_names[i], send.bind(teleport_message(i)))


## The player's base level from the last snap (1 before any).
func current_level() -> int:
	if session == null:
		return 1
	return int(session.store.you.get("level", 1))


func toggle_godmode() -> Dictionary:
	godmode_on = not godmode_on
	if godmode_button != null:
		godmode_button.text = godmode_text(godmode_on)
	return send(godmode_message(godmode_on))


func give_item() -> Dictionary:
	return send(item_message(int(item_spin.value), int(count_spin.value), int(refine_spin.value)))


## Sends one cheat message (nothing for an empty one) and returns it.
func send(msg: Dictionary) -> Dictionary:
	if msg.is_empty():
		return {}
	if session != null:
		session.send(msg)
	message_sent.emit(msg)
	return msg


# ---- pure helpers ----

static func level_message(base: int, job: int) -> Dictionary:
	return {"t": "cheat", "op": "level", "base": clampi(base, 0, MAX_BASE_LEVEL), "job": clampi(job, 0, JOB_TARGET)}


static func ryn_message(amount: int) -> Dictionary:
	return {"t": "cheat", "op": "ryn", "amount": amount}


static func simple_message(op: String) -> Dictionary:
	if not ["heal", "kit", "allskills"].has(op):
		return {}
	return {"t": "cheat", "op": op}


static func teleport_message(town: int) -> Dictionary:
	return {"t": "cheat", "op": "teleport", "town": town} if town >= 0 else {}


static func godmode_message(on: bool) -> Dictionary:
	return {"t": "cheat", "op": "godmode", "on": on}


static func item_message(item: int, count: int, refine: int) -> Dictionary:
	if item <= 0 or item > MAX_ITEM_ID:
		return {}
	return {"t": "cheat", "op": "item", "item": item, "count": clampi(count, 1, MAX_STACK),
		"refine": clampi(refine, 0, MAX_REFINE)}


static func godmode_text(on: bool) -> String:
	return "God mode: ON" if on else "God mode: off"


static func towns_from(welcome: Dictionary) -> Array[String]:
	var out: Array[String] = []
	var world: Variant = welcome.get("world", {})
	if typeof(world) != TYPE_DICTIONARY:
		return out
	var towns: Variant = (world as Dictionary).get("towns", [])
	if typeof(towns) != TYPE_ARRAY:
		return out
	for t: Variant in towns as Array:
		out.append(String((t as Dictionary).get("name", "?")) if typeof(t) == TYPE_DICTIONARY else "?")
	return out


static func group_digits(n: int) -> String:
	var s: String = str(absi(n))
	var out: String = ""
	while s.length() > 3:
		out = "," + s.substr(s.length() - 3) + out
		s = s.substr(0, s.length() - 3)
	return ("-" if n < 0 else "") + s + out
