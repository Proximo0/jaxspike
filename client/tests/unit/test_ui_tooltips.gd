extends TestCase
## ro6-ui: the rich item tooltip (ItemText.tooltip_spec, ItemTooltip.build and its hosts), the gear bonus line, the
## shared window title bar (PanelTitle) and the HUD notice line de-duplicated against the chat window.

const ITEMS: Array = [
	{},
	{"name": "red potion", "display": "Red Potion", "kind": 2, "heal_hp": 45, "weight": 7, "icon": "item.potion_red"},
	{"name": "blade", "display": "Blade", "kind": 3, "cls": 2, "atk": 53, "weapon_level": 2, "slots": 3, "weight": 60,
		"icon": "item.sword"},
	{"name": "gildscarab card", "display": "Gildscarab Card", "kind": 5, "cls": 3, "weight": 1, "icon": "item.card"},
	{"name": "ring", "display": "Ring", "kind": 4, "cls": 6, "def": 0, "weight": 10, "icon": "item.ring",
		"bonus": 0, "bonus_value": 2, "bonus_text": "STR +2"},
]
const CARDS: Array = [
	{"id": 3, "name": "Gildscarab Card", "monster": "Gildscarab", "slot": 3, "slot_name": "Shield",
		"text": "DEF +10, Max HP +1000", "chance": 1, "mvp": true, "where": "Solmera Culvert", "icon": "card.gildscarab"},
]
const WELCOME: Dictionary = {"items": ITEMS, "cards": CARDS}


func test_refine_colors_by_tier() -> void:
	assert_eq(ItemText.refine_color(0), ItemText.NAME_COLOR)
	assert_eq(ItemText.refine_color(4), ItemText.REFINE_TIER_COLORS[0])
	assert_eq(ItemText.refine_color(5), ItemText.REFINE_TIER_COLORS[1])
	assert_eq(ItemText.refine_color(10), ItemText.REFINE_TIER_COLORS[2])


func test_bonus_text_in_plain_tooltip() -> void:
	assert_eq(ItemText.bonus_text(WELCOME, 4), "STR +2")
	assert_eq(ItemText.bonus_text(WELCOME, 2), "", "no bonus")
	assert_eq(ItemText.bonus_text({"items": [{}, {"kind": 4}]}, 1), "", "older server without the field")
	assert_eq(ItemText.tooltip(WELCOME, 4).split("\n")[2], "Bonus: STR +2")


func test_spec_of_refined_slotted_gear() -> void:
	var s: Dictionary = ItemText.tooltip_spec(WELCOME, 2, 4, [3, 0, 0, 0], PackedStringArray(["Buy: 10 Ryn"]))
	assert_eq(s["refine_text"], "+4")
	assert_eq(s["refine_color"], ItemText.REFINE_TIER_COLORS[0])
	assert_eq(s["name"], "Blade [3]", "refine drawn apart, in its colour")
	assert_eq(s["type_line"], "Sword, ATK 65")
	assert_eq(s["icon_key"], "item.sword")
	assert_eq(s["card_art"], "", "gear shows its item icon")
	assert_eq((s["cards"] as Array).size(), 1)
	assert_eq((s["cards"] as Array)[0]["art"], "card.gildscarab")
	assert_eq((s["cards"] as Array)[0]["text"], "DEF +10, Max HP +1000")
	assert_eq(s["empty_sockets"], 2)
	assert_eq(s["text"], ItemText.tooltip(WELCOME, 2, 4, [3, 0, 0, 0], PackedStringArray(["Buy: 10 Ryn"])))
	var p: Dictionary = ItemText.tooltip_spec(WELCOME, 1, 3)
	assert_eq(p["refine_text"], "", "refine only on gear")
	assert_eq(ItemText.tooltip_spec(WELCOME, 0), {})


func test_spec_of_card_uses_portrait() -> void:
	var s: Dictionary = ItemText.tooltip_spec(WELCOME, 3, 0, [], PackedStringArray(["Owned: 1"]))
	assert_eq(s["card_art"], "card.gildscarab")
	assert_eq((s["card_lines"] as PackedStringArray).size(), 2)


func test_built_tooltip_draws_every_line() -> void:
	var s: Dictionary = ItemText.tooltip_spec(WELCOME, 2, 7, [3, 0, 0, 0], PackedStringArray(["Sell: 5 Ryn each"]))
	var c: Control = ItemTooltip.build(s)
	assert_true(c != null)
	var t: PackedStringArray = ItemTooltip.texts(c)
	assert_eq(t[0], "+7")
	assert_eq(t[1], "Blade [3]")
	assert_true(t.has("Gildscarab Card") and t.has("DEF +10, Max HP +1000"), str(t))
	assert_true(t.has("2 empty sockets") and t.has("Weight 60") and t.has("Sell: 5 Ryn each"), str(t))
	var ref: Label = c.find_child("Refine", true, false) as Label
	assert_eq(ref.get_theme_color("font_color"), ItemText.REFINE_TIER_COLORS[1])
	var icon: TextureRect = c.find_child("Icon", true, false) as TextureRect
	assert_true(icon.texture != null, "item icon drawn")
	c.free()
	var ring: Control = ItemTooltip.build(ItemText.tooltip_spec(WELCOME, 4))
	assert_true(ItemTooltip.texts(ring).has("Bonus: STR +2"))
	ring.free()
	assert_true(ItemTooltip.build({}) == null, "no spec: Godot's plain tooltip")
	assert_true(ItemTooltip.build({"text": "plain"}) == null)


func test_hosts_make_the_rich_tooltip() -> void:
	var b := ItemTooltip.TipButton.new()
	b.set_tip(ItemText.tooltip_spec(WELCOME, 1))
	assert_true(b.tooltip_text.begins_with("Red Potion"))
	var tip: Object = b._make_custom_tooltip(b.tooltip_text)
	assert_true(tip is Control)
	(tip as Control).free()
	b.set_tip({})
	assert_eq(b.tooltip_text, "")
	b.free()
	var r := ItemTooltip.TipRect.new()
	r.set_tip(CardAlbum.detail_tip({"id": 3, "owned": 0}, WELCOME))
	assert_true(r.tooltip_text.ends_with("Not found yet"))
	var rt: Object = r._make_custom_tooltip(r.tooltip_text)
	assert_true(ItemTooltip.texts(rt as Control).has("Not found yet"))
	(rt as Control).free()
	r.free()


func test_inventory_and_shop_rows_carry_the_spec() -> void:
	var inv: Array = []
	for i: int in range(16):
		inv.append({"item": 0, "count": 0, "refine": 0, "cards": [0, 0, 0, 0]})
	inv[0] = {"item": 2, "count": 1, "refine": 5, "cards": [3, 0, 0, 0]}
	var rows: Array[Dictionary] = InventoryPanel.slot_rows({"inv": inv, "equip": [0, -1, -1, -1, -1, -1, -1, -1]},
			ITEMS, WELCOME)
	var tip: Dictionary = rows[0]["tip"]
	assert_eq(tip["refine_text"], "+5")
	assert_eq(rows[0]["tooltip"], tip["text"])
	assert_true(String(rows[0]["tooltip"]).contains("Worn (Weapon)"))
	assert_eq(rows[1]["tip"], {}, "empty slot")
	var sell: Array[Dictionary] = ShopPanel.sell_rows(WELCOME, {"inv": [{"item": 4, "count": 1, "refine": 0}]})
	assert_true((sell[0]["tip"] as Dictionary)["bonus"] == "STR +2")


func test_title_bar_keeps_window_on_screen() -> void:
	var view := Vector2(1920, 1080)
	var sz := Vector2(300, 400)
	assert_eq(PanelTitle.clamp_position(Vector2(100, 100), sz, view), Vector2(100, 100))
	assert_eq(PanelTitle.clamp_position(Vector2(-1000, -50), sz, view), Vector2(PanelTitle.KEEP_PX - 300.0, 0))
	assert_eq(PanelTitle.clamp_position(Vector2(5000, 5000), sz, view),
			Vector2(1920.0 - PanelTitle.KEEP_PX, 1080.0 - PanelTitle.BAR_PX))


func test_title_bar_close_and_drag() -> void:
	var win := PanelContainer.new()
	var box := VBoxContainer.new()
	win.add_child(box)
	box.add_child(Label.new())
	var hits: Array[int] = [0]
	var bar: PanelTitle = PanelTitle.make(win, box, "Inventory (I)", func() -> void: hits[0] += 1)
	assert_eq(box.get_child(0), bar, "title bar first")
	assert_eq(bar.title_label.text, "Inventory (I)")
	bar.close_button.pressed.emit()
	assert_eq(hits[0], 1)
	win.free()


func _tabs(a: Array) -> Array[int]:
	var out: Array[int] = []
	for v: Variant in a:
		out.append(int(v))
	return out


func test_notice_one_place() -> void:
	assert_true(Hud.notice_on_hud(_tabs([])), "no chat: the HUD line")
	assert_false(Hud.notice_on_hud(_tabs([ChatWindow.TAB_ALL])), "chat on All logs it")
	assert_false(Hud.notice_on_hud(_tabs([ChatWindow.TAB_SYSTEM])))
	assert_true(Hud.notice_on_hud(_tabs([ChatWindow.TAB_PARTY])), "Party tab hides System lines")
	assert_false(Hud.notice_on_hud(_tabs([ChatWindow.TAB_WHISPER, ChatWindow.TAB_ALL])))
