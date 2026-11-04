class_name Hud
extends CanvasLayer
## RO basic info window (top left: name, job, base/job level, HP/SP bars with numbers, base/job exp bars with
## percent, ryn, weight), the zone name banner, the minimap (top right) and the notice line, read from the
## session's store. Feed set_welcome(msg) / on_message(msg) like the panels (the lead adds the Hud to the list).

const TOGGLE_KEY: Key = KEY_C
const INFO_POS: Vector2 = Vector2(8, 8)
const INFO_W: float = 260.0
const MINIMAP_MARGIN: float = 8.0
const BANNER_TOP: float = 110.0

var session: NetSession = null
var welcome: Dictionary = {}

var info: PanelContainer
var name_label: Label
var job_label: Label
var level_label: Label
var hp_bar: ProgressBar
var hp_label: Label
var sp_bar: ProgressBar
var sp_label: Label
var base_bar: ProgressBar
var base_label: Label
var job_bar: ProgressBar
var job_exp_label: Label
var ryn_label: Label
var weight_label: Label
var safe_badge: Label
var stats: StatSection
var stats_hint: Label
var _notice := NoticeFeed.new()
var banner: ZoneBanner
var minimap: Minimap
var _map_built_for: MapData = null
@onready var notice_label: Label = $Notice


func _ready() -> void:
	var th: Theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	info = PanelContainer.new()
	info.name = "InfoWindow"
	info.theme = th
	info.theme_type_variation = &"InfoWindow"
	info.position = INFO_POS
	info.custom_minimum_size = Vector2(INFO_W, 0)
	add_child(info)
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 2)
	info.add_child(box)
	var head := HBoxContainer.new()
	box.add_child(head)
	name_label = Label.new()
	name_label.theme_type_variation = &"InfoName"
	name_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	head.add_child(name_label)
	safe_badge = Label.new()
	safe_badge.text = "SAFE"
	safe_badge.theme_type_variation = &"InfoBadge"
	safe_badge.visible = false
	head.add_child(safe_badge)
	job_label = Label.new()
	box.add_child(job_label)
	level_label = Label.new()
	box.add_child(level_label)
	var hp: Array = _hud_bar(box, "HP", &"HpBar")
	hp_bar = hp[0]
	hp_label = hp[1]
	var sp: Array = _hud_bar(box, "SP", &"SpBar")
	sp_bar = sp[0]
	sp_label = sp[1]
	var be: Array = _hud_bar(box, "Base", &"ExpBar")
	base_bar = be[0]
	base_label = be[1]
	var je: Array = _hud_bar(box, "Job", &"ExpBar")
	job_bar = je[0]
	job_exp_label = je[1]
	var foot := HBoxContainer.new()
	box.add_child(foot)
	weight_label = Label.new()
	weight_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	foot.add_child(weight_label)
	ryn_label = Label.new()
	ryn_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	foot.add_child(ryn_label)
	stats = StatSection.new()
	stats.name = "StatSection"
	stats.visible = false
	stats.bind(session)
	stats.set_welcome(welcome)
	box.add_child(stats)
	stats_hint = Label.new()
	stats_hint.text = "(C) Stats"
	stats_hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	foot.add_child(stats_hint)
	banner = ZoneBanner.new()
	banner.name = "ZoneBanner"
	banner.set_anchors_preset(Control.PRESET_CENTER_TOP)
	banner.offset_left = -400.0
	banner.offset_right = 400.0
	banner.offset_top = BANNER_TOP
	banner.offset_bottom = BANNER_TOP + 80.0
	add_child(banner)
	minimap = Minimap.new()
	minimap.name = "Minimap"
	minimap.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	minimap.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	minimap.offset_left = -float(Minimap.VIEW_PX) - 16.0 - MINIMAP_MARGIN
	minimap.offset_right = -MINIMAP_MARGIN
	minimap.offset_top = MINIMAP_MARGIN
	add_child(minimap)
	refresh()


## Label + bar with the numbers drawn over it; returns [bar, overlay label].
func _hud_bar(parent: Control, caption: String, variation: StringName) -> Array:
	var row := HBoxContainer.new()
	parent.add_child(row)
	var cap := Label.new()
	cap.text = caption
	cap.custom_minimum_size = Vector2(36, 0)
	row.add_child(cap)
	var bar := ProgressBar.new()
	bar.theme_type_variation = variation
	bar.show_percentage = false
	bar.custom_minimum_size = Vector2(0, 14)
	bar.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	bar.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	row.add_child(bar)
	var over := Label.new()
	over.theme_type_variation = &"BarText"
	over.set_anchors_preset(Control.PRESET_FULL_RECT)
	over.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	over.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	bar.add_child(over)
	return [bar, over]


func bind(s: NetSession) -> void:
	session = s
	_map_built_for = null
	if stats != null:
		stats.bind(s)
	if minimap != null:
		minimap.bind(s)


func set_welcome(msg: Dictionary) -> void:
	welcome = msg
	if stats != null:
		stats.set_welcome(msg)
	if minimap != null:
		minimap.set_welcome(msg)
		_map_built_for = null


func on_message(msg: Dictionary) -> void:
	if String(msg.get("t", "")) == "welcome":
		set_welcome(msg)


func _unhandled_key_input(event: InputEvent) -> void:
	var k: InputEventKey = event as InputEventKey
	if k != null and k.pressed and not k.echo and k.physical_keycode == TOGGLE_KEY:
		stats.visible = not stats.visible
		stats_hint.visible = not stats.visible
		info.reset_size()
		if stats.visible:
			stats.refresh()
		get_viewport().set_input_as_handled()


func _process(delta: float) -> void:
	refresh(delta)


func refresh(delta: float = 0.0) -> void:
	if session == null or name_label == null:
		return
	var you: Dictionary = session.store.you
	var own: Dictionary = session.store.get_ent(session.slot)
	var r: Dictionary = info_rows(you, own, welcome)
	name_label.text = r["name"]
	job_label.text = r["job"]
	level_label.text = r["levels"]
	_hud_fill(hp_bar, hp_label, r["hp"], r["hp_text"])
	_hud_fill(sp_bar, sp_label, r["sp"], r["sp_text"])
	_hud_fill(base_bar, base_label, r["base"], r["base_text"])
	_hud_fill(job_bar, job_exp_label, r["job_exp"], r["job_text"])
	ryn_label.text = r["ryn"]
	weight_label.text = r["weight"]
	safe_badge.visible = bool(you.get("safe", false))
	_notice.observe(session.store.snaps, session.store.notice)
	notice_label.text = _notice.hud_line(delta)
	if stats.visible:
		stats.refresh()
	if session.map.is_loaded() and _map_built_for != session.map:
		minimap.build(session.map)
		_map_built_for = session.map
	if not you.is_empty() and session.map.is_loaded():
		var world: Variant = welcome.get("world")
		banner.observe(world if typeof(world) == TYPE_DICTIONARY else {}, session.map,
				Vector2i(ViewMath.to_vec(you.get("pos"))), int(you.get("town", 0)))


func _hud_fill(bar: ProgressBar, over: Label, fraction: float, text: String) -> void:
	bar.max_value = 1.0
	bar.step = 0.0
	bar.value = fraction
	over.text = text


# ---- pure helpers ----

## Every string and fraction of the basic info window from `you`, the own entity row and the welcome tables.
static func info_rows(you: Dictionary, own: Dictionary, welcome_msg: Dictionary) -> Dictionary:
	var nm: String = String(you.get("name", own.get("name", "")))
	var hp: int = int(you.get("hp", 0))
	var max_hp: int = int(you.get("max_hp", 0))
	var sp: int = int(you.get("sp", 0))
	var max_sp: int = int(you.get("max_sp", 0))
	var be: int = int(you.get("base_exp", 0))
	var bn: int = int(you.get("base_exp_next", 0))
	var je: int = int(you.get("job_exp", 0))
	var jn: int = int(you.get("job_exp_next", 0))
	return {
		"name": nm if nm != "" else "-",
		"job": UiStatsCalc.job_name(welcome_msg, int(you.get("job", 0))),
		"levels": "Base Lv. %d   Job Lv. %d" % [int(you.get("level", 1)), int(you.get("job_level", 1))],
		"hp": ViewMath.hp_fraction(hp, max_hp), "hp_text": "%d / %d" % [hp, max_hp],
		"sp": ViewMath.hp_fraction(sp, max_sp), "sp_text": "%d / %d" % [sp, max_sp],
		"base": UiStatsCalc.exp_fraction(be, bn), "base_text": exp_percent(be, bn),
		"job_exp": UiStatsCalc.exp_fraction(je, jn), "job_text": exp_percent(je, jn),
		"ryn": "Ryn %s" % ryn_text(int(you.get("ryn", 0))),
		"weight": weight_text(you),
	}


## RO shows exp as a percentage with one decimal; a maxed level (next 0) reads "MAX".
static func exp_percent(cur: int, next: int) -> String:
	if next <= 0:
		return "MAX"
	return "%.1f %%" % (100.0 * clampf(float(cur) / float(next), 0.0, 1.0))


## 1234567 -> "1,234,567".
static func ryn_text(z: int) -> String:
	var s: String = str(absi(z))
	var out: String = ""
	while s.length() > 3:
		out = "," + s.substr(s.length() - 3) + out
		s = s.substr(0, s.length() - 3)
	return ("-" if z < 0 else "") + s + out


## Weight is a placeholder until items carry weight: "Weight - / -" without `you.weight`.
static func weight_text(you: Dictionary) -> String:
	if not you.has("weight"):
		return "Weight - / -"
	return "Weight %d / %d" % [int(you.get("weight", 0)), int(you.get("max_weight", 0))]
