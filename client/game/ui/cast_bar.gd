class_name CastBar
extends Control
## Cast bar from `you.casting` / `you.cast_left`; hidden while idle. Fills as the cast counts down.

var session: NetSession = null
var welcome: Dictionary = {}
var defs: Dictionary = {}
var track: UiCastTrack = UiCastTrack.new()
var bar: ProgressBar
var label: Label


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	custom_minimum_size = Vector2(200, 20)
	bar = ProgressBar.new()
	bar.show_percentage = false
	bar.max_value = 1.0
	bar.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(bar)
	label = Label.new()
	label.set_anchors_preset(Control.PRESET_FULL_RECT)
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	add_child(label)
	visible = false


func bind(s: NetSession) -> void:
	session = s


func set_welcome(msg: Dictionary) -> void:
	welcome = msg
	defs = UiStatsCalc.skill_index(msg)


func _process(_delta: float) -> void:
	refresh()


func refresh() -> void:
	if bar == null or session == null:
		return
	var you: Dictionary = session.store.you
	track.update(int(you.get("casting", 0)), int(you.get("cast_left", 0)))
	visible = track.active()
	if visible:
		bar.value = track.fraction()
		label.text = String((defs.get(track.skill, {}) as Dictionary).get("name", "Skill %d" % track.skill))
