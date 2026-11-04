class_name CastBar
extends Control
## Cast bar from `you.casting` / `you.cast_left`; hidden while idle. Fills as the cast counts down.

const DEFAULT_TICK_MS: int = 100

var session: NetSession = null
var welcome: Dictionary = {}
var defs: Dictionary = {}
var track: UiCastTrack = UiCastTrack.new()
var bar: ProgressBar
var label: Label
var time_label: Label  ## Seconds left, right-aligned ("0.4 s").


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
	time_label = Label.new()
	time_label.set_anchors_preset(Control.PRESET_FULL_RECT)
	time_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	add_child(time_label)
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
	var tick_ms: int = session.tick_ms if session.tick_ms > 0 else DEFAULT_TICK_MS
	track.update(int(you.get("casting", 0)), int(you.get("cast_left", 0)), float(Time.get_ticks_msec()), tick_ms)
	visible = track.active()
	if visible:
		bar.value = track.fraction()
		label.text = String((defs.get(track.skill, {}) as Dictionary).get("name", "Skill %d" % track.skill))
		time_label.text = "%.1f s" % track.seconds_left(tick_ms)
