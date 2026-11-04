extends TestCase
## ro4-cards: InventoryPanel card tooltips (effects of socketed cards, sockets, compound target) and socket pips.

const ITEMS: Array = [
	{}, {"name": "Gold", "kind": 1}, {"name": "Red Potion", "kind": 2},
	{"name": "Sword", "kind": 3, "slots": 3, "atk": 25}, {"name": "Cotton Shirt", "kind": 4, "cls": 2, "slots": 1},
	{"name": "Blobbin Card", "kind": 5, "cls": 2}, {"name": "Antlet Card", "kind": 5, "cls": 0},
	{"name": "Capling Card", "kind": 5, "cls": 6}, {"name": "Ring", "kind": 4, "cls": 6, "slots": 0},
	{"name": "Clip", "kind": 4, "cls": 6, "slots": 1},
]
const CARDS: Array = [
	{"id": 5, "name": "Blobbin Card", "monster": "Blobbin", "slot": 2, "slot_name": "Armor",
		"text": "LUK +2, FLEE +1", "mvp": false, "icon": "card.blobbin"},
	{"id": 6, "name": "Antlet Card", "monster": "", "slot": 0, "slot_name": "Weapon", "text": "ATK +20", "mvp": false,
		"icon": "card.antlet"},
	{"id": 7, "name": "Capling Card", "monster": "Capling", "slot": 6, "slot_name": "Accessory", "text": "VIT +2, Max HP +100",
		"mvp": false, "icon": "card.capling"},
]
const WELCOME: Dictionary = {"items": ITEMS, "cards": CARDS}


func _you(sword_cards: Array, equip: Array) -> Dictionary:
	var inv: Array = []
	for i: int in range(16):
		inv.append({"item": 0, "count": 0, "refine": 0, "cards": [0, 0, 0, 0]})
	inv[0] = {"item": 3, "count": 1, "refine": 2, "cards": sword_cards}
	inv[1] = {"item": 6, "count": 1, "refine": 0, "cards": [0, 0, 0, 0]}
	inv[2] = {"item": 5, "count": 2, "refine": 0, "cards": [0, 0, 0, 0]}
	inv[3] = {"item": 7, "count": 1, "refine": 0, "cards": [0, 0, 0, 0]}
	inv[4] = {"item": 8, "count": 1, "refine": 0, "cards": [0, 0, 0, 0]}
	inv[5] = {"item": 9, "count": 1, "refine": 0, "cards": [0, 0, 0, 0]}
	return {"inv": inv, "equip": equip, "ryn": 0}


func test_socketed_cards_show_effects_and_socket_count() -> void:
	var rows: Array[Dictionary] = InventoryPanel.slot_rows(_you([6, 6, 0, 0], [0, -1, -1, -1, -1, -1, -1, -1]), ITEMS, WELCOME)
	var t: String = String(rows[0]["tooltip"])
	assert_true(t.contains("+2 Sword"), t)
	assert_true(t.contains("Cards: Antlet Card (ATK +20), Antlet Card (ATK +20)"), t)
	assert_true(t.contains("(2 / 3 sockets)"), t)


func test_card_row_tooltip_effect_and_source() -> void:
	var rows: Array[Dictionary] = InventoryPanel.slot_rows(_you([0, 0, 0, 0], [-1, -1, -1, -1, -1, -1, -1, -1]), ITEMS, WELCOME)
	var t: String = String(rows[2]["tooltip"])
	assert_true(t.contains("Armor card: LUK +2, FLEE +1"), t)
	assert_true(t.contains("Drops from Blobbin"), t)
	assert_true(t.contains("Click: no worn armor with a free socket"), t)


func test_compound_target_mirrors_the_engine() -> void:
	var you: Dictionary = _you([6, 0, 0, 0], [0, -1, -1, -1, -1, -1, -1, -1])
	var rows: Array[Dictionary] = InventoryPanel.slot_rows(you, ITEMS, WELCOME)
	var tgt: Dictionary = InventoryPanel.compound_target(rows[1], rows, you, WELCOME)
	assert_eq(int(tgt.get("equip_index", -1)), 0)
	assert_eq(int(tgt.get("free", 0)), 2)
	assert_true(String(rows[1]["tooltip"]).contains("Click: compound into worn Sword (2 free sockets)"))
	var full: Dictionary = _you([6, 6, 6, 0], [0, -1, -1, -1, -1, -1, -1, -1])
	var frows: Array[Dictionary] = InventoryPanel.slot_rows(full, ITEMS, WELCOME)
	assert_true(InventoryPanel.compound_target(frows[1], frows, full, WELCOME).is_empty(), "no free socket")
	# accessory: Acc 1 (a Ring, 0 sockets) is full, so it goes to Acc 2 (a Clip)
	var acc: Dictionary = _you([0, 0, 0, 0], [-1, -1, -1, -1, -1, -1, 4, 5])
	var arows: Array[Dictionary] = InventoryPanel.slot_rows(acc, ITEMS, WELCOME)
	assert_eq(int(InventoryPanel.compound_target(arows[3], arows, acc, WELCOME).get("equip_index", -1)), 7)


func test_socket_pips() -> void:
	var rows: Array[Dictionary] = InventoryPanel.slot_rows(_you([6, 0, 0, 0], [0, -1, -1, -1, -1, -1, -1, -1]), ITEMS, WELCOME)
	assert_eq(InventoryPanel.socket_pips(rows[0], WELCOME), [6, 0, 0] as Array[int])
	assert_eq(InventoryPanel.socket_pips(rows[2], WELCOME).size(), 0, "a card has no sockets")
	assert_eq(InventoryPanel.socket_count(WELCOME, 4), 1)
	assert_eq(InventoryPanel.card_effect(WELCOME, 7), "VIT +2, Max HP +100")
	assert_eq(InventoryPanel.card_effect({}, 7), "")


func test_panel_builds_pips_on_refresh() -> void:
	var p: InventoryPanel = (load("res://game/ui/inventory_panel.tscn") as PackedScene).instantiate() as InventoryPanel
	var root := Control.new()
	(Engine.get_main_loop() as SceneTree).root.add_child(root)
	root.add_child(p)
	p.set_welcome(WELCOME)
	assert_eq(p.slot_pips.size(), 16)
	root.free()
