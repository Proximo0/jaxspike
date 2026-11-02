class_name SkillPanel
extends PanelContainer
## RO skill learn window (key K toggles it): Novice plus own-job skills with level / max and a `+` that sends `learn`.

const TOGGLE_KEY: Key = KEY_K

var session: NetSession = null
var welcome: Dictionary = {}
var defs: Dictionary = {}

var points_label: Label
var rows: Dictionary = {}  ## skill id -> {"label": Label, "button": Button}
var _list: VBoxContainer
var _built_job: int = -1


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	custom_minimum_size = Vector2(220, 0)
	var box := VBoxContainer.new()
	add_child(box)
	var title := Label.new()
	title.text = "Skills (K)"
	box.add_child(title)
	points_label = Label.new()
	box.add_child(points_label)
	_list = VBoxContainer.new()
	box.add_child(_list)
	visible = false


func bind(s: NetSession) -> void:
	session = s


func set_welcome(msg: Dictionary) -> void:
	welcome = msg
	defs = UiStatsCalc.skill_index(msg)
	_built_job = -1


func _unhandled_key_input(event: InputEvent) -> void:
	var k: InputEventKey = event as InputEventKey
	if k != null and k.pressed and not k.echo and k.physical_keycode == TOGGLE_KEY:
		visible = not visible
		if visible:
			refresh()
		get_viewport().set_input_as_handled()


func _process(_delta: float) -> void:
	if visible:
		refresh()


func refresh() -> void:
	if _list == null or session == null:
		return
	var you: Dictionary = session.store.you
	var job: int = int(you.get("job", 0))
	var entries: Array[Dictionary] = UiStatsCalc.panel_skills(defs, you)
	if job != _built_job or entries.size() != rows.size():
		_rebuild(entries)
		_built_job = job
	points_label.text = "Skill points: %d" % int(you.get("skill_points", 0))
	for e: Dictionary in entries:
		var r: Dictionary = rows[int(e["id"])]
		(r["label"] as Label).text = "%s  %d / %d" % [e["name"], e["level"], e["max_level"]]
		(r["button"] as Button).disabled = not bool(e["can_learn"])


func _rebuild(entries: Array[Dictionary]) -> void:
	for c: Node in _list.get_children():
		c.queue_free()
	rows.clear()
	for e: Dictionary in entries:
		var row := HBoxContainer.new()
		_list.add_child(row)
		var l := Label.new()
		l.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		row.add_child(l)
		var b := Button.new()
		b.text = "+"
		b.pressed.connect(_on_learn.bind(int(e["id"])))
		row.add_child(b)
		rows[int(e["id"])] = {"label": l, "button": b}


func _on_learn(id: int) -> void:
	if session == null:
		return
	var you: Dictionary = session.store.you
	var def: Dictionary = defs.get(id, {})
	if def.is_empty() or not UiStatsCalc.can_learn(you, UiStatsCalc.learned_level(you, id), int(def.get("max_level", 1))):
		return
	session.send({"t": "learn", "id": id})
