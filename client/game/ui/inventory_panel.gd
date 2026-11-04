class_name InventoryPanel
extends PanelContainer
## RO inventory window (key I toggles it): 16 item slots, zeny, the 8 equipment slots and, once the server has sent
## `you.storage`, the kafra storage list. Left click = use or equip toggle, right click = menu (Equip/Use, Refine,
## Drop, Store). The static helpers are pure (slot rows, click rules, message builders) and unit tested.

signal message_sent(msg: Dictionary)

const TOGGLE_KEY: Key = KEY_I
const INV_SLOTS: int = 16
const EQUIP_NAMES: Array[String] = ["Weapon", "Head", "Body", "Shield", "Garment", "Shoes", "Acc 1", "Acc 2"]
const KIND_CONSUMABLE: int = 2
const KIND_WEAPON: int = 3
const KIND_ARMOR: int = 4
const KIND_CARD: int = 5
const ACT_USE: String = "Use"
const ACT_EQUIP: String = "Equip"
const ACT_UNEQUIP: String = "Unequip"
const ACT_REFINE: String = "Refine"
const ACT_DROP: String = "Drop"
const ACT_STORE: String = "Store"
const ACT_SELL: String = "Sell"
const ACT_DROP_ONE: String = "Drop 1"
const REFINE_ATK_BY_LEVEL: Array[int] = [0, 2, 3, 5, 7]  ## per refine, by weapon level (ro_items.REFINE_ATK)
const SHOP_NEAR_TILES: int = 1
const SLOT_PX: int = 52
const EQUIPPED_TINT: Color = Color(0.75, 1.0, 0.75)

var session: NetSession = null
var welcome: Dictionary = {}
var slots: AssetSlots = null

var rows: Array[Dictionary] = []  ## last computed slot rows (slot_rows)
var equip_rows: Array[Dictionary] = []
var storage_known: bool = false
var at_shop: bool = false  ## a shop tile is within 1 tile of the player (enables Sell)
var storage_items: Array = []  ## cached `you.storage` (the server sends it only on change)
var zeny_label: Label
var slot_buttons: Array[Button] = []
var equip_buttons: Array[Button] = []
var storage_box: VBoxContainer
var storage_list: VBoxContainer
var menu: PopupMenu
var menu_slot: int = -1
var menu_actions: Array[String] = []
var _synced_version: int = -1
var _storage_sig: String = ""


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	if slots == null:
		slots = AssetSlots.load_default()
	var box := VBoxContainer.new()
	add_child(box)
	var head := HBoxContainer.new()
	box.add_child(head)
	var title := Label.new()
	title.text = "Inventory (I)"
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	head.add_child(title)
	zeny_label = Label.new()
	head.add_child(zeny_label)
	var grid := GridContainer.new()
	grid.columns = 4
	box.add_child(grid)
	for i: int in range(INV_SLOTS):
		var b := Button.new()
		b.custom_minimum_size = Vector2(SLOT_PX, SLOT_PX)
		b.clip_text = true
		b.vertical_icon_alignment = VERTICAL_ALIGNMENT_TOP
		b.icon_alignment = HORIZONTAL_ALIGNMENT_CENTER
		b.gui_input.connect(_on_slot_gui_input.bind(i))
		grid.add_child(b)
		slot_buttons.append(b)
	var eq_title := Label.new()
	eq_title.text = "Equipment"
	box.add_child(eq_title)
	var eq_grid := GridContainer.new()
	eq_grid.columns = 2
	box.add_child(eq_grid)
	for i: int in range(EQUIP_NAMES.size()):
		var b := Button.new()
		b.custom_minimum_size = Vector2(SLOT_PX * 2, 28)
		b.alignment = HORIZONTAL_ALIGNMENT_LEFT
		b.clip_text = true
		b.pressed.connect(click_equip.bind(i))
		eq_grid.add_child(b)
		equip_buttons.append(b)
	storage_box = VBoxContainer.new()
	storage_box.visible = false
	box.add_child(storage_box)
	var st_title := Label.new()
	st_title.text = "Kafra storage (click = take out)"
	storage_box.add_child(st_title)
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size = Vector2(0, 120)
	storage_box.add_child(scroll)
	storage_list = VBoxContainer.new()
	storage_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(storage_list)
	menu = PopupMenu.new()
	menu.id_pressed.connect(_on_menu_id)
	add_child(menu)
	visible = false
	refresh()


func bind(s: NetSession) -> void:
	session = s


## Feed the decoded `welcome` message (the session does not keep it).
func set_welcome(msg: Dictionary) -> void:
	welcome = msg
	_synced_version = -1


## Convenience for a polling loop: forwards `welcome`, ignores everything else.
func on_message(msg: Dictionary) -> void:
	if String(msg.get("t", "")) == "welcome":
		set_welcome(msg)


func _process(_delta: float) -> void:
	if session != null and session.store.version != _synced_version:
		refresh()


func _unhandled_key_input(event: InputEvent) -> void:
	var k := event as InputEventKey
	if k != null and k.pressed and not k.echo and k.keycode == TOGGLE_KEY:
		visible = not visible
		get_viewport().set_input_as_handled()


## Recomputes the rows from the session's `you` and updates the buttons.
func refresh() -> void:
	if not is_node_ready():
		return
	var you: Dictionary = session.store.you if session != null else {}
	_synced_version = session.store.version if session != null else -1
	var items: Array = _items()
	at_shop = near_shop_tile(_pos_tile(you.get("pos")), _shops())
	rows = slot_rows(you, items, welcome)
	equip_rows = equip_slot_rows(you, items, welcome)
	zeny_label.text = "Zeny %d" % int(you.get("zeny", 0))
	for i: int in range(INV_SLOTS):
		_show_slot(slot_buttons[i], rows[i])
	for i: int in range(equip_rows.size()):
		var er: Dictionary = equip_rows[i]
		equip_buttons[i].text = "%s: %s" % [EQUIP_NAMES[i], String(er["name"]) if int(er["item"]) > 0 else "-"]
		equip_buttons[i].tooltip_text = String(er["tooltip"])
	if typeof(you.get("storage")) == TYPE_ARRAY:
		storage_items = (you["storage"] as Array).duplicate()
		storage_known = true
	storage_box.visible = storage_known
	if storage_known:
		_rebuild_storage()


func _items() -> Array:
	var v: Variant = welcome.get("items")
	return v as Array if typeof(v) == TYPE_ARRAY else []


func _shops() -> Array:
	var w: Variant = welcome.get("world")
	if typeof(w) == TYPE_DICTIONARY:
		var sh: Variant = (w as Dictionary).get("shops")
		if typeof(sh) == TYPE_ARRAY:
			return sh as Array
	return []


## A tile as Vector2i, or [x, y] after decode (anything else gives a far-away tile).
static func _pos_tile(v: Variant) -> Vector2i:
	if typeof(v) == TYPE_VECTOR2I:
		return v as Vector2i
	if typeof(v) == TYPE_ARRAY and (v as Array).size() >= 2:
		return Vector2i(int((v as Array)[0]), int((v as Array)[1]))
	return Vector2i(-100000, -100000)


func _show_slot(b: Button, r: Dictionary) -> void:
	if bool(r["empty"]):
		b.text = ""
		b.icon = null
		b.tooltip_text = ""
		b.modulate = Color.WHITE
		return
	b.icon = slots.texture(String(r["icon_key"])) if slots != null else null
	var cnt: int = int(r["count"])
	b.text = ("E " if bool(r["equipped"]) else "") + (str(cnt) if cnt > 1 else "")
	b.tooltip_text = String(r["tooltip"])
	b.modulate = EQUIPPED_TINT if bool(r["equipped"]) else Color.WHITE


func _rebuild_storage() -> void:
	var sig: String = str(storage_items.hash())
	if sig == _storage_sig:
		return
	_storage_sig = sig
	for c: Node in storage_list.get_children():
		storage_list.remove_child(c)
		c.queue_free()
	for r: Dictionary in storage_rows(storage_items, _items(), welcome):
		var b := Button.new()
		b.text = String(r["name"])
		b.alignment = HORIZONTAL_ALIGNMENT_LEFT
		b.icon = slots.texture(AssetSlots.item_key(int(r["item"]))) if slots != null else null
		b.pressed.connect(take_out.bind(int(r["slot"])))
		storage_list.add_child(b)


# ---- interaction (also called by the sims) ----

func _on_slot_gui_input(event: InputEvent, slot: int) -> void:
	var mb := event as InputEventMouseButton
	if mb == null or not mb.pressed:
		return
	if mb.button_index == MOUSE_BUTTON_LEFT:
		click_slot(slot)
	elif mb.button_index == MOUSE_BUTTON_RIGHT:
		open_menu(slot, Vector2i(mb.global_position))


## Left click: use a consumable or card, toggle gear. Returns the message sent ({} when the slot does nothing).
func click_slot(slot: int) -> Dictionary:
	if slot < 0 or slot >= rows.size():
		return {}
	var act: String = click_action(rows[slot])
	if act == "":
		return {}
	return send_action(act, slot)


## Click on an equipment slot: toggles the worn item off.
func click_equip(index: int) -> Dictionary:
	if index < 0 or index >= equip_rows.size():
		return {}
	var inv_slot: int = int(equip_rows[index]["slot"])
	if inv_slot < 0:
		return {}
	return send_action(ACT_UNEQUIP, inv_slot)


func open_menu(slot: int, at: Vector2i) -> void:
	if slot < 0 or slot >= rows.size():
		return
	menu_actions = menu_for(rows[slot], storage_known, at_shop)
	if menu_actions.is_empty():
		return
	menu_slot = slot
	menu.clear()
	for i: int in range(menu_actions.size()):
		menu.add_item(menu_actions[i], i)
	menu.position = at
	menu.popup()


func _on_menu_id(id: int) -> void:
	if id >= 0 and id < menu_actions.size():
		send_action(menu_actions[id], menu_slot)


## Sends the message for `action` on inventory slot `slot` and returns it.
func send_action(action: String, slot: int) -> Dictionary:
	var msg: Dictionary = action_message(action, slot)
	if msg.is_empty():
		return {}
	if session != null:
		session.send(msg)
	message_sent.emit(msg)
	return msg


func take_out(storage_slot: int) -> Dictionary:
	var msg: Dictionary = {"t": "storage", "op": "out", "slot": storage_slot}
	if session != null:
		session.send(msg)
	message_sent.emit(msg)
	return msg


# ---- pure helpers ----

static func item_info(items: Array, id: int) -> Dictionary:
	if id > 0 and id < items.size() and typeof(items[id]) == TYPE_DICTIONARY:
		return items[id]
	return {}


## Name from welcome.items, else welcome.craft_items (string or int keys), else "Item <id>".
static func item_name(welcome_msg: Dictionary, id: int) -> String:
	if id <= 0:
		return ""
	var items: Variant = welcome_msg.get("items")
	if typeof(items) == TYPE_ARRAY:
		var info: Dictionary = item_info(items as Array, id)
		if info.has("name") and String(info["name"]) != "":
			return String(info["name"])
	var ci: Variant = welcome_msg.get("craft_items")
	if typeof(ci) == TYPE_DICTIONARY:
		var d: Dictionary = ci
		if d.has(str(id)):
			return String(d[str(id)]).capitalize()
		if d.has(id):
			return String(d[id]).capitalize()
	return "Item %d" % id


static func item_kind(items: Array, id: int) -> int:
	return int(item_info(items, id).get("kind", 0))


## Exactly INV_SLOTS rows from `you.inv` (dict rows, or the legacy flat id list). Keys: slot, item, count, refine,
## cards, empty, name, kind, equipped, equip_index, icon_key, tooltip.
static func slot_rows(you: Dictionary, items: Array, welcome_msg: Dictionary = {}) -> Array[Dictionary]:
	var inv: Array = you["inv"] as Array if typeof(you.get("inv")) == TYPE_ARRAY else []
	var worn: Dictionary = _equip_map(you)
	var out: Array[Dictionary] = []
	for i: int in range(INV_SLOTS):
		var item: int = 0
		var count: int = 0
		var refine: int = 0
		var cards: Array[int] = [0, 0, 0, 0]
		if i < inv.size():
			var raw: Variant = inv[i]
			if typeof(raw) == TYPE_DICTIONARY:
				var d: Dictionary = raw
				item = int(d.get("item", 0))
				count = int(d.get("count", 0))
				refine = int(d.get("refine", 0))
				var cs: Variant = d.get("cards")
				if typeof(cs) == TYPE_ARRAY:
					for j: int in range(mini((cs as Array).size(), 4)):
						cards[j] = int((cs as Array)[j])
			elif typeof(raw) == TYPE_INT or typeof(raw) == TYPE_FLOAT:
				item = int(raw)
				count = 1 if item > 0 else 0
		var empty: bool = item <= 0
		var nm: String = item_name(welcome_msg, item)
		var eq_idx: int = int(worn.get(i, -1))
		out.append({
			"slot": i, "item": item, "count": count, "refine": refine, "cards": cards, "empty": empty,
			"name": nm, "kind": item_kind(items, item), "equipped": eq_idx >= 0 and not empty,
			"equip_index": eq_idx, "icon_key": AssetSlots.item_key(item),
			"tooltip": "" if empty else tooltip_text(nm, refine, cards, welcome_msg),
		})
	var eq: Variant = you.get("equip")
	for r: Dictionary in out:
		if bool(r["empty"]) or bool(r["equipped"]):
			continue
		var wi: int = equip_target(items, int(r["item"]))
		var wr: Dictionary = {}
		if wi >= 0 and typeof(eq) == TYPE_ARRAY and wi < (eq as Array).size():
			var ws: int = int((eq as Array)[wi])
			if ws >= 0 and ws < INV_SLOTS and not bool(out[ws]["empty"]):
				wr = out[ws]
		var ct: String = compare_text(r, wr, items)
		if ct != "":
			r["tooltip"] = String(r["tooltip"]) + "\n" + ct
	return out


## Equipment index an item would be worn in (weapon 0, armor by cls 1..6), -1 for anything else.
static func equip_target(items: Array, id: int) -> int:
	var info: Dictionary = item_info(items, id)
	match int(info.get("kind", 0)):
		KIND_WEAPON:
			return 0
		KIND_ARMOR:
			var c: int = int(info.get("cls", 0))
			return c if c >= 1 and c <= 6 else -1
	return -1


## ATK (weapon) or DEF (armor) including refine.
static func gear_stat(items: Array, id: int, refine: int) -> int:
	var info: Dictionary = item_info(items, id)
	if int(info.get("kind", 0)) == KIND_WEAPON:
		var wl: int = clampi(int(info.get("weapon_level", 1)), 0, REFINE_ATK_BY_LEVEL.size() - 1)
		return int(info.get("atk", 0)) + refine * REFINE_ATK_BY_LEVEL[wl]
	return int(info.get("def", 0)) + refine


## Compare lines for an unworn, single weapon/armor row against `worn` (the row in its equip slot, {} if none).
static func compare_text(row: Dictionary, worn: Dictionary, items: Array) -> String:
	if bool(row.get("empty", true)) or bool(row.get("equipped", false)) or int(row.get("count", 1)) > 1:
		return ""
	var kind: int = int(row.get("kind", 0))
	if kind != KIND_WEAPON and kind != KIND_ARMOR:
		return ""
	var label: String = "ATK" if kind == KIND_WEAPON else "DEF"
	var id: int = int(row.get("item", 0))
	var mine: int = gear_stat(items, id, int(row.get("refine", 0)))
	var my_slots: int = int(item_info(items, id).get("slots", 0))
	if worn.is_empty() or int(worn.get("item", 0)) <= 0:
		var t: String = "%s %d (nothing worn)" % [label, mine]
		return t + ("\nSlots %d" % my_slots if my_slots > 0 else "")
	var wid: int = int(worn["item"])
	var theirs: int = gear_stat(items, wid, int(worn.get("refine", 0)))
	var t2: String = "%s %d (worn %s %d, %+d)" % [label, mine, String(worn.get("name", "")), theirs, mine - theirs]
	var worn_slots: int = int(item_info(items, wid).get("slots", 0))
	if my_slots > 0 or worn_slots > 0:
		t2 += "\nSlots %d (worn %d)" % [my_slots, worn_slots]
	return t2


## True when any shop tile ({"tile": [x, y]}) is within SHOP_NEAR_TILES (Chebyshev) of `pos`.
static func near_shop_tile(pos: Vector2i, world_shops: Array) -> bool:
	for sh: Variant in world_shops:
		if typeof(sh) != TYPE_DICTIONARY:
			continue
		var t: Vector2i = _pos_tile((sh as Dictionary).get("tile"))
		if maxi(absi(t.x - pos.x), absi(t.y - pos.y)) <= SHOP_NEAR_TILES:
			return true
	return false


static func tooltip_text(nm: String, refine: int, cards: Array[int], welcome_msg: Dictionary) -> String:
	var t: String = ("+%d " % refine if refine > 0 else "") + nm
	var names: PackedStringArray = []
	for c: int in cards:
		if c > 0:
			names.append(item_name(welcome_msg, c))
	if not names.is_empty():
		t += "\nCards: " + ", ".join(names)
	return t


## inventory slot -> equipment index for every worn slot in `you.equip`.
static func _equip_map(you: Dictionary) -> Dictionary:
	var out: Dictionary = {}
	var eq: Variant = you.get("equip")
	if typeof(eq) == TYPE_ARRAY:
		for i: int in range((eq as Array).size()):
			var s: int = int((eq as Array)[i])
			if s >= 0:
				out[s] = i
	return out


## The 8 equipment rows: index, label, slot (inventory slot or -1), item, name, tooltip.
static func equip_slot_rows(you: Dictionary, items: Array, welcome_msg: Dictionary = {}) -> Array[Dictionary]:
	var inv_rows: Array[Dictionary] = slot_rows(you, items, welcome_msg)
	var eq: Array = you["equip"] as Array if typeof(you.get("equip")) == TYPE_ARRAY else []
	var out: Array[Dictionary] = []
	for i: int in range(EQUIP_NAMES.size()):
		var s: int = int(eq[i]) if i < eq.size() else -1
		var r: Dictionary = {"index": i, "label": EQUIP_NAMES[i], "slot": -1, "item": 0, "name": "", "tooltip": ""}
		if s >= 0 and s < INV_SLOTS and not bool(inv_rows[s]["empty"]):
			r["slot"] = s
			r["item"] = inv_rows[s]["item"]
			r["name"] = inv_rows[s]["name"]
			r["tooltip"] = inv_rows[s]["tooltip"]
		out.append(r)
	return out


## Non-empty kafra storage entries: {slot, item, name, kind}.
static func storage_rows(storage: Array, items: Array, welcome_msg: Dictionary = {}) -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	for i: int in range(storage.size()):
		var id: int = int(storage[i])
		if id > 0:
			out.append({"slot": i, "item": id, "name": item_name(welcome_msg, id), "kind": item_kind(items, id)})
	return out


## What a left click does: "" nothing, Use (consumable or card), Equip/Unequip (weapon, armor).
static func click_action(row: Dictionary) -> String:
	if bool(row.get("empty", true)):
		return ""
	match int(row.get("kind", 0)):
		KIND_CONSUMABLE, KIND_CARD:
			return ACT_USE
		KIND_WEAPON, KIND_ARMOR:
			return ACT_UNEQUIP if bool(row.get("equipped", false)) else ACT_EQUIP
	return ""


## Right click menu entries for a row: the click action, Refine (gear), Drop, Store (when storage is open).
static func menu_for(row: Dictionary, can_store: bool, can_sell: bool = false) -> Array[String]:
	var out: Array[String] = []
	if bool(row.get("empty", true)):
		return out
	var main: String = click_action(row)
	if main != "":
		out.append(main)
	var kind: int = int(row.get("kind", 0))
	if kind == KIND_WEAPON or kind == KIND_ARMOR:
		out.append(ACT_REFINE)
	out.append(ACT_DROP_ONE if int(row.get("count", 1)) > 1 else ACT_DROP)
	if can_sell and not bool(row.get("equipped", false)):
		out.append(ACT_SELL)
	if can_store:
		out.append(ACT_STORE)
	return out


## The wire message for an action on inventory slot `slot`, {} for an unknown action.
static func action_message(action: String, slot: int) -> Dictionary:
	match action:
		ACT_USE:
			return {"t": "use_item", "slot": slot}
		ACT_EQUIP, ACT_UNEQUIP:
			return {"t": "equip", "slot": slot}
		ACT_REFINE:
			return {"t": "refine", "slot": slot}
		ACT_DROP, ACT_DROP_ONE:
			return {"t": "drop", "slot": slot}
		ACT_SELL:
			return {"t": "sell", "slot": slot}
		ACT_STORE:
			return {"t": "storage", "op": "in", "slot": slot}
	return {}
