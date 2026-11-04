class_name StatusPanel
extends PanelContainer
## Compatibility shell kept for tests/sim/sim_ui_stats.gd. The game shows the status through the Hud info window with
## its embedded StatSection (key C); this panel is no longer added to the game view and never toggles itself.

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
var section: StatSection
var points_label: Label
var stat_labels: Array[Label] = []
var stat_buttons: Array[Button] = []
var job_buttons: Array[Button] = []


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	custom_minimum_size = Vector2(220, 0)
	var box := VBoxContainer.new()
	add_child(box)
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
	section = StatSection.new()
	box.add_child(section)
	section.bind(session)
	section.set_welcome(welcome)
	_sync_refs()
	visible = false


func bind(s: NetSession) -> void:
	session = s
	if section != null:
		section.bind(s)


func set_welcome(msg: Dictionary) -> void:
	welcome = msg
	if section != null:
		section.set_welcome(msg)
		_sync_refs()


func _sync_refs() -> void:
	points_label = section.points_label
	stat_labels = section.stat_labels
	stat_buttons = section.stat_buttons
	job_buttons = section.job_buttons


func _bar(box: VBoxContainer, caption: String) -> ProgressBar:
	var b := ProgressBar.new()
	b.custom_minimum_size = Vector2(0, 16)
	b.show_percentage = false
	b.max_value = 1.0
	b.tooltip_text = caption
	box.add_child(b)
	return b


func refresh() -> void:
	if name_label == null:
		return
	var you: Dictionary = session.store.you if session != null else {}
	name_label.text = String(you.get("name", player_name))
	job_label.text = UiStatsCalc.job_name(welcome, int(you.get("job", 0)))
	level_label.text = "Base Lv %d   Job Lv %d" % [int(you.get("level", 1)), int(you.get("job_level", 1))]
	_set_bar(hp_bar, int(you.get("hp", 0)), int(you.get("max_hp", 0)))
	_set_bar(sp_bar, int(you.get("sp", 0)), int(you.get("max_sp", 0)))
	_exp_bar(base_bar, "Base", int(you.get("base_exp", 0)), int(you.get("base_exp_next", 0)))
	_exp_bar(job_bar, "Job", int(you.get("job_exp", 0)), int(you.get("job_exp_next", 0)))
	section.refresh()


func _set_bar(bar: ProgressBar, cur: int, maxv: int) -> void:
	bar.max_value = maxf(float(maxv), 1.0)
	bar.value = float(cur)
	bar.tooltip_text = UiStatsCalc.bar_text(cur, maxv)


func _exp_bar(bar: ProgressBar, caption: String, cur: int, next: int) -> void:
	bar.max_value = 1.0
	bar.value = UiStatsCalc.exp_fraction(cur, next)
	bar.tooltip_text = "%s %s" % [caption, UiStatsCalc.exp_text(cur, next)]
