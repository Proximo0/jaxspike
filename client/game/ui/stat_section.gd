class_name StatSection
extends VBoxContainer
## The stat half of the RO status window: six stats with `+` buttons, points left and the first-job change buttons.
## Embedded under the basic info window. Decisions are in UiStatsCalc; this node shows them and sends messages.

const STAT_LABELS: Array[String] = ["STR", "AGI", "VIT", "INT", "DEX", "LUK"]
const NAME_W: float = 40.0

var session: NetSession = null
var welcome: Dictionary = {}
var points_label: Label
var stat_labels: Array[Label] = []
var stat_buttons: Array[Button] = []
var job_buttons: Array[Button] = []
var _job_row: HBoxContainer


func _ready() -> void:
	_build()


func _build() -> void:
	if _job_row != null:
		return
	add_child(HSeparator.new())
	for i: int in range(STAT_LABELS.size()):
		var row := HBoxContainer.new()
		add_child(row)
		var nm := Label.new()
		nm.text = STAT_LABELS[i]
		nm.custom_minimum_size = Vector2(NAME_W, 0)
		row.add_child(nm)
		var val := Label.new()
		val.custom_minimum_size = Vector2(NAME_W, 0)
		val.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		row.add_child(val)
		stat_labels.append(val)
		var plus := Button.new()
		plus.text = "+"
		plus.focus_mode = Control.FOCUS_NONE
		plus.pressed.connect(_on_allot.bind(i))
		row.add_child(plus)
		stat_buttons.append(plus)
	points_label = Label.new()
	add_child(points_label)
	_job_row = HBoxContainer.new()
	add_child(_job_row)
	rebuild_jobs()


func bind(s: NetSession) -> void:
	session = s


## Feed the decoded `welcome` message (the session does not keep it).
func set_welcome(msg: Dictionary) -> void:
	welcome = msg
	rebuild_jobs()


func rebuild_jobs() -> void:
	_build()
	for b: Button in job_buttons:
		_job_row.remove_child(b)
		b.queue_free()
	job_buttons.clear()
	for c: Dictionary in UiStatsCalc.job_choices(welcome):
		var b := Button.new()
		b.text = String(c["name"])
		b.focus_mode = Control.FOCUS_NONE
		b.pressed.connect(_on_job.bind(int(c["job"])))
		_job_row.add_child(b)
		job_buttons.append(b)


func _you() -> Dictionary:
	return session.store.you if session != null else {}


func refresh() -> void:
	_build()
	var you: Dictionary = _you()
	for i: int in range(STAT_LABELS.size()):
		stat_labels[i].text = str(UiStatsCalc.stat_value(you, i))
		stat_buttons[i].disabled = not UiStatsCalc.can_allot(you, welcome, i)
	points_label.text = "Stat points: %d   Skill points: %d" % [
			int(you.get("stat_points", 0)), int(you.get("skill_points", 0))]
	var can_job: bool = UiStatsCalc.can_change_job(you)
	for b: Button in job_buttons:
		b.disabled = not can_job


func _on_allot(i: int) -> void:
	if session != null and UiStatsCalc.can_allot(_you(), welcome, i):
		session.send({"t": "allot", "stat": i})


func _on_job(j: int) -> void:
	if session != null and UiStatsCalc.can_change_job(_you()):
		session.send({"t": "job", "job": j})
