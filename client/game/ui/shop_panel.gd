class_name ShopPanel
extends PanelContainer
## RO shop window. No hotkey: it opens when the player stands within one tile of a shop tile (`welcome.world.shops`)
## and closes when they walk away. Buy rows list the town stock, sell rows the unworn inventory stacks. Prices
## follow the Discount (buy) and Overcharge (sell) skills. The static helpers are pure and unit tested.

signal message_sent(msg: Dictionary)

const NEAR_DIST: int = 1
const PRICE_BASE_PCT: int = 5
const PRICE_PER_LEVEL_PCT: int = 2
const PRICE_MAX_PCT: int = 24
const SKILL_DISCOUNT: String = "Discount"
const SKILL_OVERCHARGE: String = "Overcharge"
const ICON_PX: int = 32

var session: NetSession = null
var welcome: Dictionary = {}
var slots: AssetSlots = null

var town: int = -1
var dismissed_town: int = -1
var shop_name: String = ""
var rows_buy: Array[Dictionary] = []
var rows_sell: Array[Dictionary] = []
var title_label: Label
var ryn_label: Label
var buy_list: VBoxContainer
var sell_list: VBoxContainer
var _synced_version: int = -1
var _buy_sig: String = ""
var _sell_sig: String = ""


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	if slots == null:
		slots = AssetSlots.load_default()
	var box := VBoxContainer.new()
	add_child(box)
	var head := HBoxContainer.new()
	box.add_child(head)
	title_label = Label.new()
	title_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	head.add_child(title_label)
	ryn_label = Label.new()
	head.add_child(ryn_label)
	var close_btn := Button.new()
	close_btn.text = "Close"
	close_btn.pressed.connect(close)
	head.add_child(close_btn)
	var cols := HBoxContainer.new()
	box.add_child(cols)
	buy_list = _shop_column(cols, "Buy")
	sell_list = _shop_column(cols, "Sell")
	visible = false
	refresh()


func _shop_column(parent: Node, title: String) -> VBoxContainer:
	var col := VBoxContainer.new()
	col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	parent.add_child(col)
	var lbl := Label.new()
	lbl.text = title
	col.add_child(lbl)
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size = Vector2(220, 200)
	col.add_child(scroll)
	var list := VBoxContainer.new()
	list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(list)
	return list


func bind(s: NetSession) -> void:
	session = s


func set_welcome(msg: Dictionary) -> void:
	welcome = msg
	_synced_version = -1


func on_message(msg: Dictionary) -> void:
	if String(msg.get("t", "")) == "welcome":
		set_welcome(msg)


func _process(_delta: float) -> void:
	if session != null and session.store.version != _synced_version:
		refresh()


func refresh() -> void:
	if not is_node_ready():
		return
	var you: Dictionary = session.store.you if session != null else {}
	_synced_version = session.store.version if session != null else -1
	town = near_shop(shop_pos(you.get("pos")), shop_tiles(welcome)) if you.has("pos") else -1
	if town < 0:
		dismissed_town = -1
	visible = town >= 0 and town != dismissed_town
	ryn_label.text = "Ryn %d" % int(you.get("ryn", 0))
	if town < 0:
		return
	shop_name = shop_title(welcome, town)
	title_label.text = "%s shop" % shop_name
	rows_buy = buy_rows(welcome, town, you)
	rows_sell = sell_rows(welcome, you)
	_rebuild(buy_list, rows_buy, true)
	_rebuild(sell_list, rows_sell, false)


func _rebuild(list: VBoxContainer, rows: Array[Dictionary], is_buy: bool) -> void:
	var sig: String = str(rows.hash())
	if is_buy:
		if sig == _buy_sig:
			return
		_buy_sig = sig
	else:
		if sig == _sell_sig:
			return
		_sell_sig = sig
	for c: Node in list.get_children():
		list.remove_child(c)
		c.queue_free()
	for r: Dictionary in rows:
		var line := HBoxContainer.new()
		var icon := TextureRect.new()
		icon.custom_minimum_size = Vector2(ICON_PX, ICON_PX)
		icon.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		icon.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
		icon.texture = slots.item_texture(String(r["icon_key"])) if slots != null else null
		icon.tooltip_text = String(r["tooltip"])
		icon.mouse_filter = Control.MOUSE_FILTER_PASS
		line.add_child(icon)
		var nm := Label.new()
		nm.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		nm.text = String(r["label"])
		nm.tooltip_text = String(r["tooltip"])
		nm.mouse_filter = Control.MOUSE_FILTER_PASS
		line.add_child(nm)
		var pr := Label.new()
		pr.text = price_text(int(r["price"]))
		line.add_child(pr)
		var b := Button.new()
		b.text = "Buy" if is_buy else "Sell"
		if is_buy:
			b.pressed.connect(buy.bind(int(r["item"])))
		else:
			b.pressed.connect(sell.bind(int(r["slot"])))
		line.add_child(b)
		list.add_child(line)


## Hides the window until the player leaves this shop's range.
func close() -> void:
	dismissed_town = town
	visible = false


func buy(item: int) -> Dictionary:
	return _shop_send(buy_message(item))


func sell(slot: int) -> Dictionary:
	return _shop_send(sell_message(slot))


func _shop_send(msg: Dictionary) -> Dictionary:
	if town < 0:
		return {}
	if session != null:
		session.send(msg)
	message_sent.emit(msg)
	return msg


# ---- pure helpers ----

static func shop_pos(v: Variant) -> Vector2i:
	if typeof(v) == TYPE_VECTOR2I:
		return v
	if typeof(v) == TYPE_ARRAY and (v as Array).size() >= 2:
		return Vector2i(int((v as Array)[0]), int((v as Array)[1]))
	return Vector2i.ZERO


## `welcome.world.shops` or [].
static func shop_tiles(welcome_msg: Dictionary) -> Array:
	var w: Variant = welcome_msg.get("world")
	if typeof(w) == TYPE_DICTIONARY:
		var s: Variant = (w as Dictionary).get("shops")
		if typeof(s) == TYPE_ARRAY:
			return s
	return []


## Town of a shop tile within Chebyshev distance 1 of `pos`, else -1.
static func near_shop(pos: Vector2i, world_shops: Array) -> int:
	for s: Variant in world_shops:
		if typeof(s) != TYPE_DICTIONARY:
			continue
		var t: Vector2i = shop_pos((s as Dictionary).get("tile"))
		if maxi(absi(t.x - pos.x), absi(t.y - pos.y)) <= NEAR_DIST:
			return int((s as Dictionary).get("town", 0))
	return -1


## The `welcome.shops` entry of a town (matched by its `town` field, else by index) or {}.
static func town_shop(welcome_msg: Dictionary, town_id: int) -> Dictionary:
	var shops: Variant = welcome_msg.get("shops")
	if typeof(shops) != TYPE_ARRAY:
		return {}
	for s: Variant in shops:
		if typeof(s) == TYPE_DICTIONARY and (s as Dictionary).has("town") and int((s as Dictionary)["town"]) == town_id:
			return s
	if town_id >= 0 and town_id < (shops as Array).size() and typeof((shops as Array)[town_id]) == TYPE_DICTIONARY:
		return (shops as Array)[town_id]
	return {}


static func shop_title(welcome_msg: Dictionary, town_id: int) -> String:
	var nm: String = String(town_shop(welcome_msg, town_id).get("name", ""))
	return nm if nm != "" else "Town %d" % town_id


static func price_pct(level: int) -> int:
	if level <= 0:
		return 0
	return mini(PRICE_BASE_PCT + PRICE_PER_LEVEL_PCT * level, PRICE_MAX_PCT)


static func buy_price(base: int, discount_pct: int) -> int:
	return maxi(base * (100 - discount_pct) / 100, 1)


static func sell_price(base: int, overcharge_pct: int) -> int:
	return base * (100 + overcharge_pct) / 100


## Percent of the named skill (level from welcome.skills.names index and you.learned).
static func skill_pct(welcome_msg: Dictionary, you: Dictionary, skill_name: String) -> int:
	var sk: Variant = welcome_msg.get("skills")
	if typeof(sk) != TYPE_DICTIONARY:
		return 0
	var names: Variant = (sk as Dictionary).get("names")
	if typeof(names) != TYPE_ARRAY:
		return 0
	var idx: int = (names as Array).find(skill_name)
	if idx < 0:
		return 0
	return price_pct(UiStatsCalc.learned_level(you, idx))


static func _shop_item_info(welcome_msg: Dictionary, id: int) -> Dictionary:
	var items: Variant = welcome_msg.get("items")
	if typeof(items) == TYPE_ARRAY and id > 0 and id < (items as Array).size() and typeof((items as Array)[id]) == TYPE_DICTIONARY:
		return (items as Array)[id]
	return {}


static func buy_rows(welcome_msg: Dictionary, town_id: int, you: Dictionary) -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	var pct: int = skill_pct(welcome_msg, you, SKILL_DISCOUNT)
	var stock: Variant = town_shop(welcome_msg, town_id).get("items")
	if typeof(stock) != TYPE_ARRAY:
		return out
	for v: Variant in stock:
		var id: int = int(v)
		if id <= 0:
			continue
		var info: Dictionary = _shop_item_info(welcome_msg, id)
		var price: int = buy_price(int(info.get("buy", 0)), pct)
		out.append({"item": id, "name": InventoryPanel.item_name(welcome_msg, id),
				"label": ItemText.display_name(welcome_msg, id), "price": price,
				"icon_key": InventoryPanel.row_icon_key(welcome_msg, id),
				"tooltip": ItemText.tooltip(welcome_msg, id, 0, [], PackedStringArray(["Buy: " + price_text(price)]))})
	return out


static func sell_rows(welcome_msg: Dictionary, you: Dictionary) -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	var pct: int = skill_pct(welcome_msg, you, SKILL_OVERCHARGE)
	var inv: Variant = you.get("inv")
	if typeof(inv) != TYPE_ARRAY:
		return out
	var worn: Array = []
	var eq: Variant = you.get("equip")
	if typeof(eq) == TYPE_ARRAY:
		for e: Variant in eq:
			if int(e) >= 0:
				worn.append(int(e))
	for i: int in range((inv as Array).size()):
		var raw: Variant = (inv as Array)[i]
		if typeof(raw) != TYPE_DICTIONARY or worn.has(i):
			continue
		var d: Dictionary = raw
		var id: int = int(d.get("item", 0))
		var cnt: int = int(d.get("count", 0))
		if id <= 0 or cnt <= 0:
			continue
		var refine: int = int(d.get("refine", 0))
		var cards: Array = d["cards"] as Array if typeof(d.get("cards")) == TYPE_ARRAY else []
		var info: Dictionary = _shop_item_info(welcome_msg, id)
		var price: int = sell_price(int(info.get("sell", 0)), pct)
		var extra: PackedStringArray = PackedStringArray(["Sell: %s each" % price_text(price)])
		out.append({"slot": i, "item": id, "name": InventoryPanel.item_name(welcome_msg, id), "count": cnt,
				"label": ItemText.stack_name(welcome_msg, id, cnt, refine), "price": price,
				"icon_key": InventoryPanel.row_icon_key(welcome_msg, id),
				"tooltip": ItemText.tooltip(welcome_msg, id, refine, cards, extra)})
	return out


## "1,250 Ryn" (docs/naming.md 1.3).
static func price_text(ryn: int) -> String:
	var s: String = str(absi(ryn))
	var grouped: String = ""
	while s.length() > 3:
		grouped = "," + s.substr(s.length() - 3) + grouped
		s = s.substr(0, s.length() - 3)
	return "%s%s%s Ryn" % ["-" if ryn < 0 else "", s, grouped]


static func buy_message(item: int) -> Dictionary:
	return {"t": "buy", "item": item}


static func sell_message(slot: int) -> Dictionary:
	return {"t": "sell", "slot": slot}
