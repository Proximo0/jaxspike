class_name CraftPanel
extends PanelContainer
## Gathering and crafting window (key R toggles it; key G gathers from an adjacent node, even while hidden): six
## craft skill levels with exp bars, the recipe list greyed when level, inputs or station are missing, a Craft button.
## Eligibility and exp maths are pure statics.

signal message_sent(msg: Dictionary)

const TOGGLE_KEY: Key = KEY_R
const GATHER_KEY: Key = KEY_G
const GATHER_RANGE: int = 1
const STATION_NONE: int = -1
const SKILL_NAMES: Array[String] = ["Mining", "Smithing", "Fishing", "Cooking", "Herbalism", "Woodcutting"]
const GREY: Color = Color(0.5, 0.5, 0.5)

var session: NetSession = null
var welcome: Dictionary = {}
var craft: Dictionary = {"level": [], "exp": []}  ## cached `you.craft` (sent only when it changes)
var selected: int = 0
var statuses: Dictionary = {}  ## recipe id -> recipe_status
var skill_bars: Array[ProgressBar] = []
var skill_labels: Array[Label] = []
var recipe_buttons: Dictionary = {}  ## recipe id -> Button
var craft_button: Button
var gather_button: Button
var _list: VBoxContainer
var _synced_version: int = -1
var _built_for: int = -1


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	custom_minimum_size = Vector2(260, 0)
	var box := VBoxContainer.new()
	add_child(box)
	var title := Label.new()
	title.text = "Crafting (R)"
	box.add_child(title)
	for i: int in range(SKILL_NAMES.size()):
		var row := HBoxContainer.new()
		box.add_child(row)
		var lab := Label.new()
		lab.custom_minimum_size = Vector2(110, 0)
		row.add_child(lab)
		skill_labels.append(lab)
		var bar := ProgressBar.new()
		bar.show_percentage = false
		bar.custom_minimum_size = Vector2(120, 14)
		bar.max_value = 1.0
		row.add_child(bar)
		skill_bars.append(bar)
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size = Vector2(0, 180)
	box.add_child(scroll)
	_list = VBoxContainer.new()
	_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(_list)
	craft_button = Button.new()
	craft_button.text = "Craft"
	craft_button.pressed.connect(craft_selected)
	box.add_child(craft_button)
	gather_button = Button.new()
	gather_button.text = "Gather (G)"
	gather_button.pressed.connect(gather)
	box.add_child(gather_button)
	visible = false
	refresh()


func bind(s: NetSession) -> void:
	session = s


## Feed the decoded `welcome` message (the session does not keep it).
func set_welcome(msg: Dictionary) -> void:
	welcome = msg
	_synced_version = -1
	_built_for = -1


func on_message(msg: Dictionary) -> void:
	if String(msg.get("t", "")) == "welcome":
		set_welcome(msg)


func _process(_delta: float) -> void:
	if session != null and session.store.version != _synced_version:
		refresh()


func _unhandled_key_input(event: InputEvent) -> void:
	var k := event as InputEventKey
	if k == null or not k.pressed or k.echo:
		return
	if k.keycode == TOGGLE_KEY:
		visible = not visible
		get_viewport().set_input_as_handled()
	elif k.keycode == GATHER_KEY:
		if not gather().is_empty():
			get_viewport().set_input_as_handled()


func refresh() -> void:
	if not is_node_ready():
		return
	var you: Dictionary = session.store.you if session != null else {}
	_synced_version = session.store.version if session != null else -1
	var c: Variant = you.get("craft")
	if typeof(c) == TYPE_DICTIONARY:
		craft = c
	var recipes: Array = recipe_list(welcome)
	var counts: Dictionary = inv_counts(you)
	var stations: Array[int] = []
	if session != null:
		stations = nearby_station_kinds(session.store, you.get("pos", Vector2i.ZERO))
	statuses.clear()
	for r: Dictionary in recipes:
		statuses[int(r["id"])] = recipe_status(r, craft, counts, stations)
	_rebuild_list(recipes)
	var lv: Array = craft_array(craft, "level")
	var ex: Array = craft_array(craft, "exp")
	var table: Array = _level_exp()
	for i: int in range(SKILL_NAMES.size()):
		var level: int = int(lv[i]) if i < lv.size() else 1
		var xp: int = int(ex[i]) if i < ex.size() else 0
		skill_labels[i].text = "%s Lv %d" % [SKILL_NAMES[i], maxi(level, 1)]
		skill_bars[i].value = level_fraction(level, xp, table)
	craft_button.disabled = not selected_ok()
	gather_button.disabled = session == null or not can_gather(session.store, you.get("pos", Vector2i.ZERO))


func _level_exp() -> Array:
	var v: Variant = welcome.get("craft_skills")
	if typeof(v) == TYPE_DICTIONARY:
		for key: String in ["LEVEL_EXP", "level_exp"]:
			if typeof((v as Dictionary).get(key)) == TYPE_ARRAY:
				return (v as Dictionary)[key]
	return []


func _rebuild_list(recipes: Array) -> void:
	if _built_for != recipes.size():
		_built_for = recipes.size()
		recipe_buttons.clear()
		for ch: Node in _list.get_children():
			_list.remove_child(ch)
			ch.queue_free()
		for r: Dictionary in recipes:
			var b := Button.new()
			b.alignment = HORIZONTAL_ALIGNMENT_LEFT
			b.pressed.connect(select_recipe.bind(int(r["id"])))
			_list.add_child(b)
			recipe_buttons[int(r["id"])] = b
	for r: Dictionary in recipes:
		var id: int = int(r["id"])
		var st: Dictionary = statuses[id]
		var b: Button = recipe_buttons[id]
		b.text = recipe_label(r, welcome)
		b.modulate = Color.WHITE if bool(st["ok"]) else GREY
		b.tooltip_text = ", ".join(PackedStringArray(st["reasons"])) if not bool(st["ok"]) else "Ready"


# ---- interaction ----

func select_recipe(id: int) -> void:
	selected = id
	craft_button.disabled = not selected_ok()


func selected_ok() -> bool:
	return statuses.has(selected) and bool((statuses[selected] as Dictionary)["ok"])


## Sends craft for the selected recipe when it is eligible. Returns the message ({} when not sent).
func craft_selected() -> Dictionary:
	if not selected_ok():
		return {}
	return _send({"t": "craft", "recipe": selected})


## Sends gather when a harvestable node is adjacent. Returns the message ({} when not sent).
func gather() -> Dictionary:
	if session == null or not can_gather(session.store, session.store.you.get("pos", Vector2i.ZERO)):
		return {}
	return _send({"t": "gather"})


func _send(msg: Dictionary) -> Dictionary:
	session.send(msg)
	message_sent.emit(msg)
	return msg


# ---- pure helpers ----

static func recipe_list(welcome_msg: Dictionary) -> Array:
	var v: Variant = welcome_msg.get("recipes")
	return v as Array if typeof(v) == TYPE_ARRAY else []


static func recipe_label(r: Dictionary, welcome_msg: Dictionary) -> String:
	var skill: int = int(r.get("skill", 0))
	var sk: String = SKILL_NAMES[skill] if skill >= 0 and skill < SKILL_NAMES.size() else "?"
	return "%s  (%s %d)" % [InventoryPanel.item_name(welcome_msg, int(r.get("output", 0))), sk, int(r.get("level", 1))]


static func craft_array(c: Dictionary, key: String) -> Array:
	var v: Variant = c.get(key)
	return v as Array if typeof(v) == TYPE_ARRAY else []


## item id -> total count over the inventory rows (a row with an item and count 0 counts as 1).
static func inv_counts(you: Dictionary) -> Dictionary:
	var out: Dictionary = {}
	var inv: Variant = you.get("inv")
	if typeof(inv) != TYPE_ARRAY:
		return out
	for raw: Variant in inv as Array:
		var item: int = 0
		var count: int = 0
		if typeof(raw) == TYPE_DICTIONARY:
			item = int((raw as Dictionary).get("item", 0))
			count = int((raw as Dictionary).get("count", 0))
		elif typeof(raw) == TYPE_INT or typeof(raw) == TYPE_FLOAT:
			item = int(raw)
			count = 1
		if item > 0:
			out[item] = int(out.get(item, 0)) + maxi(count, 1)
	return out


## Progress inside the current level, 0..1. `level_exp[L-1]` is the total exp needed for level L.
static func level_fraction(level: int, xp: int, level_exp: Array) -> float:
	if level_exp.is_empty():
		return 0.0
	var l: int = clampi(level, 1, level_exp.size())
	if l >= level_exp.size():
		return 1.0
	var lo: int = int(level_exp[l - 1])
	var hi: int = int(level_exp[l])
	if hi <= lo:
		return 1.0
	return clampf(float(xp - lo) / float(hi - lo), 0.0, 1.0)


## Whether a recipe can be crafted now. `stations` = node kinds (0..7) within GATHER_RANGE of the player.
## Returns {ok, reasons: Array[String], level_ok, inputs_ok, station_ok}.
static func recipe_status(r: Dictionary, craft_state: Dictionary, counts: Dictionary, stations: Array) -> Dictionary:
	var skill: int = int(r.get("skill", 0))
	var lv: Array = craft_array(craft_state, "level")
	var have_level: int = int(lv[skill]) if skill >= 0 and skill < lv.size() else 1
	var level_ok: bool = have_level >= int(r.get("level", 1))
	var need: Dictionary = {}
	var inputs: Variant = r.get("inputs")
	if typeof(inputs) == TYPE_ARRAY:
		for i: Variant in inputs as Array:
			if int(i) > 0:
				need[int(i)] = int(need.get(int(i), 0)) + 1
	var inputs_ok: bool = true
	for k: int in need:
		if int(counts.get(k, 0)) < int(need[k]):
			inputs_ok = false
	var station: int = int(r.get("station", STATION_NONE))
	var station_ok: bool = station < 0 or stations.has(station)
	var reasons: Array[String] = []
	if not level_ok:
		reasons.append("needs %s level %d" % [SKILL_NAMES[skill] if skill >= 0 and skill < SKILL_NAMES.size() else "skill", int(r.get("level", 1))])
	if not inputs_ok:
		reasons.append("missing ingredients")
	if not station_ok:
		reasons.append("needs a furnace" if station == 6 else "needs a cooking fire")
	return {"ok": level_ok and inputs_ok and station_ok, "reasons": reasons, "level_ok": level_ok,
			"inputs_ok": inputs_ok, "station_ok": station_ok}


## Node entities (kinds 200..207, no `item` key) within manhattan GATHER_RANGE of `pos`: [{kind, hp}].
static func nodes_near(store: EntityStore, pos: Variant) -> Array[Dictionary]:
	var p: Vector2i = pos if pos is Vector2i else Vector2i(ViewMath.to_vec(pos))
	var out: Array[Dictionary] = []
	for id: int in store.ids():
		var e: Dictionary = store.get_ent(id)
		var kind: int = int(e.get("kind", 0))
		if e.has("item") or not ViewMath.is_node_kind(kind):
			continue
		var d: Vector2i = (e["pos"] as Vector2i) - p
		if absi(d.x) + absi(d.y) <= GATHER_RANGE:
			out.append({"kind": kind - ViewMath.NODE_KIND_BASE, "hp": int(e.get("hp", 0))})
	return out


static func nearby_station_kinds(store: EntityStore, pos: Variant) -> Array[int]:
	var out: Array[int] = []
	for n: Dictionary in nodes_near(store, pos):
		if int(n["kind"]) >= ViewMath.NODE_GATHER_COUNT:
			out.append(int(n["kind"]))
	return out


## True when a harvestable (kinds 200..205), not depleted node is adjacent.
static func can_gather(store: EntityStore, pos: Variant) -> bool:
	for n: Dictionary in nodes_near(store, pos):
		if int(n["kind"]) < ViewMath.NODE_GATHER_COUNT and int(n["hp"]) > 0:
			return true
	return false
