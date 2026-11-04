extends TestCase

const ITEMS: Array = [
	{}, {"name": "Gold", "kind": 1}, {"name": "Red Potion", "kind": 2}, {"name": "Sword", "kind": 3},
	{"name": "Cap", "kind": 4}, {"name": "Blobbin Card", "kind": 5},
]
const WELCOME: Dictionary = {"items": ITEMS, "craft_items": {"100": "copper ore"}}


func _you() -> Dictionary:
	var inv: Array = []
	for i: int in range(16):
		inv.append({"item": 0, "count": 0, "refine": 0, "cards": [0, 0, 0, 0]})
	inv[0] = {"item": 2, "count": 5, "refine": 0, "cards": [0, 0, 0, 0]}
	inv[1] = {"item": 3, "count": 1, "refine": 4, "cards": [5.0, 0, 0, 0]}
	inv[2] = {"item": 100, "count": 3, "refine": 0, "cards": [0, 0, 0, 0]}
	var eq: Array = [1, -1, -1, -1, -1, -1, -1, -1]
	return {"inv": inv, "equip": eq, "ryn": 77}


func test_rows_always_16_and_fields() -> void:
	var rows: Array[Dictionary] = InventoryPanel.slot_rows(_you(), ITEMS, WELCOME)
	assert_eq(rows.size(), 16)
	assert_eq(rows[0]["name"], "Red Potion")
	assert_eq(rows[0]["count"], 5)
	assert_true(rows[1]["equipped"])
	assert_eq(rows[1]["equip_index"], 0)
	assert_eq(rows[1]["refine"], 4)
	assert_true(rows[5]["empty"])
	assert_eq(rows[2]["name"], "Copper Ore")
	assert_true(String(rows[1]["tooltip"]).contains("+4 Sword"))
	assert_true(String(rows[1]["tooltip"]).contains("Blobbin Card"))
	assert_eq(InventoryPanel.slot_rows({}, ITEMS).size(), 16)
	assert_eq(InventoryPanel.slot_rows({"inv": [3, 2]}, ITEMS)[0]["item"], 3, "legacy flat list")


func test_click_and_menu() -> void:
	var rows: Array[Dictionary] = InventoryPanel.slot_rows(_you(), ITEMS, WELCOME)
	assert_eq(InventoryPanel.click_action(rows[0]), "Use")
	assert_eq(InventoryPanel.click_action(rows[1]), "Unequip")
	assert_eq(InventoryPanel.click_action(rows[2]), "")
	assert_eq(InventoryPanel.click_action(rows[5]), "")
	assert_eq(InventoryPanel.menu_for(rows[0], false), ["Use", "Drop 1"] as Array[String])
	assert_eq(InventoryPanel.menu_for(rows[1], true), ["Unequip", "Refine", "Drop", "Store"] as Array[String])
	assert_eq(InventoryPanel.menu_for(rows[5], true).size(), 0)


func test_messages() -> void:
	assert_eq(InventoryPanel.action_message("Use", 3), {"t": "use_item", "slot": 3})
	assert_eq(InventoryPanel.action_message("Equip", 3), {"t": "equip", "slot": 3})
	assert_eq(InventoryPanel.action_message("Unequip", 3), {"t": "equip", "slot": 3})
	assert_eq(InventoryPanel.action_message("Refine", 3), {"t": "refine", "slot": 3})
	assert_eq(InventoryPanel.action_message("Drop", 3), {"t": "drop", "slot": 3})
	assert_eq(InventoryPanel.action_message("Store", 3), {"t": "storage", "op": "in", "slot": 3})
	assert_true(InventoryPanel.action_message("nope", 3).is_empty())


func test_equip_and_storage_rows() -> void:
	var er: Array[Dictionary] = InventoryPanel.equip_slot_rows(_you(), ITEMS, WELCOME)
	assert_eq(er.size(), 8)
	assert_eq(er[0]["slot"], 1)
	assert_eq(er[0]["name"], "Sword")
	assert_eq(er[1]["slot"], -1)
	var st: Array = []
	st.resize(100)
	st.fill(0)
	st[7] = 2
	var sr: Array[Dictionary] = InventoryPanel.storage_rows(st, ITEMS, WELCOME)
	assert_eq(sr.size(), 1)
	assert_eq(sr[0]["slot"], 7)
	assert_eq(sr[0]["name"], "Red Potion")


const GEAR: Array = [
	{}, {"name": "Knife", "kind": 3, "atk": 17, "weapon_level": 1, "slots": 1, "cls": 0},
	{"name": "Sword", "kind": 3, "atk": 25, "weapon_level": 1, "slots": 3, "cls": 0},
	{"name": "Hat", "kind": 4, "def": 2, "slots": 1, "cls": 1}, {"name": "Cap", "kind": 4, "def": 4, "slots": 3, "cls": 1},
	{"name": "Potion", "kind": 2},
]


func _gear_you(worn: bool) -> Dictionary:
	var inv: Array = []
	for i: int in range(16):
		inv.append({"item": 0, "count": 0, "refine": 0, "cards": [0, 0, 0, 0]})
	inv[0] = {"item": 1, "count": 1, "refine": 0, "cards": [0, 0, 0, 0]}
	inv[1] = {"item": 2, "count": 1, "refine": 0, "cards": [0, 0, 0, 0]}
	inv[2] = {"item": 4, "count": 1, "refine": 0, "cards": [0, 0, 0, 0]}
	inv[3] = {"item": 5, "count": 4, "refine": 0, "cards": [0, 0, 0, 0]}
	var eq: Array = [0 if worn else -1, -1, -1, -1, -1, -1, -1, -1]
	return {"inv": inv, "equip": eq}


func test_compare_text() -> void:
	var rows: Array[Dictionary] = InventoryPanel.slot_rows(_gear_you(true), GEAR, {"items": GEAR})
	assert_eq(InventoryPanel.compare_text(rows[1], rows[0], GEAR), "ATK 25 (worn Knife 17, +8)\nSlots 3 (worn 1)")
	assert_eq(InventoryPanel.compare_text(rows[0], rows[0], GEAR), "", "equipped row")
	assert_eq(rows[0]["tooltip"], "Knife")
	assert_true(String(rows[1]["tooltip"]).contains("Sword\nATK 25 (worn Knife 17, +8)"))
	assert_true(String(rows[2]["tooltip"]).contains("DEF 4 (nothing worn)"))
	assert_eq(InventoryPanel.compare_text(rows[3], {}, GEAR), "", "stack")
	var r2: Dictionary = rows[1].duplicate()
	r2["refine"] = 2
	assert_true(InventoryPanel.compare_text(r2, rows[0], GEAR).begins_with("ATK 29 (worn Knife 17, +12)"))


func test_near_shop_and_menu_sell() -> void:
	var shops: Array = [{"tile": [10, 10], "town": 0}]
	assert_true(InventoryPanel.near_shop_tile(Vector2i(11, 9), shops))
	assert_false(InventoryPanel.near_shop_tile(Vector2i(12, 10), shops))
	assert_false(InventoryPanel.near_shop_tile(Vector2i(10, 10), []))
	var rows: Array[Dictionary] = InventoryPanel.slot_rows(_you(), ITEMS, WELCOME)
	assert_eq(InventoryPanel.menu_for(rows[0], false, true), ["Use", "Drop 1", "Sell"] as Array[String])
	assert_eq(InventoryPanel.menu_for(rows[0], false, false), ["Use", "Drop 1"] as Array[String])
	assert_eq(InventoryPanel.menu_for(rows[1], false, true), ["Unequip", "Refine", "Drop"] as Array[String])
	assert_eq(InventoryPanel.action_message("Sell", 4), {"t": "sell", "slot": 4})
	assert_eq(InventoryPanel.action_message("Drop 1", 4), {"t": "drop", "slot": 4})
