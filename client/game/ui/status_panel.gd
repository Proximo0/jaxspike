class_name StatusPanel
extends PanelContainer
## RO status window (key C toggles it): name, job, levels, HP/SP/exp bars, six stats with `+` buttons, points left and
## the first-job change buttons. All decisions are in UiStatsCalc; this node only shows them and sends messages.

const TOGGLE_KEY: Key = KEY_C
const STAT_LABELS: Array[String] = ["STR", "AGI", "VIT", "INT", "DEX", "LUK"]

var session: NetSession = null
var welcome: Dictionary = {}
var player_name: String = ""

var name_label: Label
var job_label: Label
var level_label: Label
var hp_bar: ProgressBar
var sp_bar: ProgressBar
var base_bar: ProgressBar
var job_bar: ProgressBar
var points_label: Label
var stat_labels: Array[Label] = []
var stat_buttons: Array[Button] = []
var job_buttons: Array[Button] = []
var _job_row: HBoxContainer


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	custom_minimum_size = Vector2(220, 0)
	var box := VBoxContainer.new()
	add_child(box)
	var title := Label.new()
	title.text = "Status (C)"
	box.add_child(title)
	name_label = Label.new()
	box.add_child(name_label)
	job_label = Label.new()
	box.add_child(job_label)
	level_label = Label.new()
	box.add_child(level_label)
	hp_bar = _bar(box, "HP")
	sp_bar = _bar(box, "SP")
	base_bar = _bar(box, "Base")
	job_bar = _bar(box, "Job")
	for i: int in range(STAT_LABELS.size()):
		var row := HBoxContainer.new()
		box.add_child(row)
		var nm := Label.new()
		nm.text = STAT_LABELS[i]
		nm.custom_minimum_size = Vector2(40, 0)
		row.add_child(nm)
		var val := Label.new()
		val.custom_minimum_size = Vector2(40, 0)
		row.add_child(val)
		stat_labels.append(val)
		var plus := Button.new()
		plus.text = "+"
		plus.pressed.connect(_on_allot.bind(i))
		row.add_child(plus)
		stat_buttons.append(plus)
	points_label = Label.new()
	box.add_child(points_label)
	_job_row = HBoxContainer.new()
	box.add_child(_job_row)
	visible = false
	rebuild_jobs()


func bind(s: NetSession) -> void:
	session = s


## Feed the decoded `welcome` message (the session does not keep it).
func set_welcome(msg: Dictionary) -> void:
	welcome = msg
	if is_node_ready():
		rebuild_jobs()


func rebuild_jobs() -> void:
	for b: Button in job_buttons:
		b.queue_free()
	job_buttons.clear()
	for c: Dictionary in UiStatsCalc.job_choices(welcome):
		var b := Button.new()
		b.text = String(c["name"])
		b.pressed.connect(_on_job.bind(int(c["job"])))
		_job_row.add_child(b)
		job_buttons.append(b)


func _bar(box: VBoxContainer, caption: String) -> ProgressBar:
	var b := ProgressBar.new()
	b.custom_minimum_size = Vector2(0, 16)
	b.show_percentage = false
	b.max_value = 1.0
	b.tooltip_text = caption
	box.add_child(b)
	return b


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


func _you() -> Dictionary:
	return session.store.you if session != null else {}


func refresh() -> void:
	if name_label == null:
		return
	var you: Dictionary = _you()
	name_label.text = String(you.get("name", player_name))
	var job: int = int(you.get("job", 0))
	job_label.text = UiStatsCalc.job_name(welcome, job)
	level_label.text = "Base Lv %d   Job Lv %d" % [int(you.get("level", 1)), int(you.get("job_level", 1))]
	_set_bar(hp_bar, int(you.get("hp", 0)), int(you.get("max_hp", 0)), UiStatsCalc.bar_text(
			int(you.get("hp", 0)), int(you.get("max_hp", 0))))
	_set_bar(sp_bar, int(you.get("sp", 0)), int(you.get("max_sp", 0)), UiStatsCalc.bar_text(
			int(you.get("sp", 0)), int(you.get("max_sp", 0))))
	_exp_bar(base_bar, "Base", int(you.get("base_exp", 0)), int(you.get("base_exp_next", 0)))
	_exp_bar(job_bar, "Job", int(you.get("job_exp", 0)), int(you.get("job_exp_next", 0)))
	for i: int in range(STAT_LABELS.size()):
		stat_labels[i].text = str(UiStatsCalc.stat_value(you, i))
		stat_buttons[i].disabled = not UiStatsCalc.can_allot(you, welcome, i)
	points_label.text = "Stat points: %d   Skill points: %d" % [
			int(you.get("stat_points", 0)), int(you.get("skill_points", 0))]
	var can_job: bool = UiStatsCalc.can_change_job(you)
	for b: Button in job_buttons:
		b.disabled = not can_job


func _set_bar(bar: ProgressBar, cur: int, maxv: int, tip: String) -> void:
	bar.max_value = maxf(float(maxv), 1.0)
	bar.value = float(cur)
	bar.tooltip_text = tip


func _exp_bar(bar: ProgressBar, caption: String, cur: int, next: int) -> void:
	bar.max_value = 1.0
	bar.value = UiStatsCalc.exp_fraction(cur, next)
	bar.tooltip_text = "%s %s" % [caption, UiStatsCalc.exp_text(cur, next)]


func _on_allot(i: int) -> void:
	if session != null and UiStatsCalc.can_allot(_you(), welcome, i):
		session.send({"t": "allot", "stat": i})


func _on_job(j: int) -> void:
	if session != null and UiStatsCalc.can_change_job(_you()):
		session.send({"t": "job", "job": j})
