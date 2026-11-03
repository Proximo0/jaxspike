class_name SkillBar
extends PanelContainer
## Nine hotkey slots (keys 1..9) for learned castable skills: cooldown sweep from `you.cooldowns`, greyed when SP is
## short. A press sends {"t":"skill","id","target"}; the target comes from TargetSelect (left click selects).

const PICK_TILES: float = 0.75  ## Click pick radius, in tiles.

## Pie-shaped cooldown overlay: `fraction` of the circle (from 12 o'clock, clockwise) stays dark.
class Dial extends Control:
	var fraction: float = 0.0

	func _draw() -> void:
		if fraction <= 0.0:
			return
		var c: Vector2 = size * 0.5
		var r: float = maxf(size.x, size.y)
		var pts := PackedVector2Array([c])
		var steps: int = 24
		for i: int in range(steps + 1):
			var a: float = -PI * 0.5 + TAU * fraction * float(i) / float(steps)
			pts.append(c + Vector2(cos(a), sin(a)) * r)
		draw_colored_polygon(pts, Color(0.0, 0.0, 0.0, 0.55))

var session: NetSession = null
var welcome: Dictionary = {}
var defs: Dictionary = {}
var layer: EntityLayer = null  ## For click-to-select; optional.
var targeting: TargetSelect = TargetSelect.new()
var entries: Array[Dictionary] = []
var buttons: Array[Button] = []
var dials: Array[Dial] = []


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	theme_type_variation = &"HotkeyBar"
	var row := HBoxContainer.new()
	add_child(row)
	for i: int in range(UiStatsCalc.BAR_SLOTS):
		var b := Button.new()
		b.theme_type_variation = &"HotkeySlot"
		b.custom_minimum_size = Vector2(44, 44)
		b.clip_text = true
		b.focus_mode = Control.FOCUS_NONE
		b.pressed.connect(press_slot.bind(i))
		row.add_child(b)
		var d := Dial.new()
		d.set_anchors_preset(Control.PRESET_FULL_RECT)
		d.mouse_filter = Control.MOUSE_FILTER_IGNORE
		b.add_child(d)
		buttons.append(b)
		dials.append(d)
	refresh()


func bind(s: NetSession, entity_layer: EntityLayer = null) -> void:
	session = s
	layer = entity_layer


func set_welcome(msg: Dictionary) -> void:
	welcome = msg
	defs = UiStatsCalc.skill_index(msg)


func _unhandled_key_input(event: InputEvent) -> void:
	var k: InputEventKey = event as InputEventKey
	if k == null or not k.pressed or k.echo:
		return
	var slot: int = UiStatsCalc.slot_for_key(k.physical_keycode)
	if slot >= 0:
		press_slot(slot)
		get_viewport().set_input_as_handled()


func _process(_delta: float) -> void:
	refresh()


## Left click at a world position: selects the nearest non-own, non-drop entity (or clears). Returns true if one
## was selected, so the caller can skip the walk-to for that click.
func click_world(world_pos: Vector2) -> bool:
	if session == null or layer == null:
		return false
	var pos: Dictionary = {}
	for id: int in session.store.ids():
		if id == session.slot or int(session.store.get_ent(id).get("kind", 0)) >= EntityStore.KIND_DROP_BASE:
			continue
		var p: Vector2 = layer.world_pos_of(id)
		if p != Vector2.INF:
			pos[id] = p
	var tile_px: float = ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)
	return targeting.pick(world_pos, pos, tile_px * PICK_TILES) != TargetSelect.NO_TARGET


func refresh() -> void:
	if buttons.is_empty() or session == null:
		return
	var you: Dictionary = session.store.you
	targeting.prune(session.store.has_ent)
	entries = UiStatsCalc.bar_skills(defs, you)
	for i: int in range(buttons.size()):
		var b: Button = buttons[i]
		if i >= entries.size():
			b.text = "%d" % (i + 1)
			b.disabled = true
			b.tooltip_text = ""
			dials[i].fraction = 0.0
			dials[i].queue_redraw()
			continue
		var e: Dictionary = entries[i]
		b.text = "%d %s" % [i + 1, e["name"]]
		b.tooltip_text = "%s Lv%d  SP %d" % [e["name"], e["level"], e["sp"]]
		b.disabled = not UiStatsCalc.can_afford(you, int(e["sp"]))
		dials[i].fraction = UiStatsCalc.cooldown_fraction(UiStatsCalc.cooldown_left(you, int(e["id"])), int(e["cooldown"]))
		dials[i].queue_redraw()


## Casts the skill in slot `i` (0-based). Returns true when a message was sent.
func press_slot(i: int) -> bool:
	if session == null or i < 0 or i >= entries.size():
		return false
	var e: Dictionary = entries[i]
	var you: Dictionary = session.store.you
	if not UiStatsCalc.can_afford(you, int(e["sp"])) or UiStatsCalc.cooldown_left(you, int(e["id"])) > 0:
		return false
	var target: int = targeting.resolve(int(e["target"]), session.slot)
	if target == TargetSelect.NEEDS_TARGET:
		return false
	session.send({"t": "skill", "id": int(e["id"]), "target": target})
	return true
