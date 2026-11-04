extends TestCase
## ro5-items: ItemText, the shared item formatter (names, icon keys, the one tooltip) and the item icon slots.

const ITEMS: Array = [
	{},
	{"name": "gold coin", "kind": 1, "icon": "item.gold_coin"},
	{"name": "red potion", "display": "Red Potion", "kind": 2, "heal_hp": 45, "weight": 7, "icon": "item.potion_red"},
	{"name": "blade", "display": "Blade", "kind": 3, "cls": 2, "atk": 53, "weapon_level": 2, "slots": 3, "weight": 60,
		"icon": "item.sword"},
	{"name": "hat", "kind": 4, "cls": 1, "def": 2, "slots": 0},
	{"name": "gildscarab card", "display": "Gildscarab Card", "kind": 5, "cls": 3, "weight": 1, "icon": "item.card"},
	{"name": "bill of birds", "kind": 1},
	{"name": "", "kind": 0, "icon": ""},
	{"name": "knife", "kind": 3, "cls": 1, "atk": 17, "slots": 1},
]
const CARDS: Array = [
	{"id": 5, "name": "Gildscarab Card", "monster": "Gildscarab", "slot": 3, "slot_name": "Shield",
		"text": "DEF +10, Max HP +1000", "chance": 1, "mvp": true, "where": "Solmera Culvert", "icon": "card.gildscarab"},
]
const WELCOME: Dictionary = {"items": ITEMS, "cards": CARDS, "craft_items": {"100": "copper ore"}}
## Every icon key the engine's welcome.items uses (game/data/items.py _icon) plus the kind fallbacks.
const ENGINE_ICONS: Array[String] = [
	"item.gold_coin", "item.meat", "item.potion_red", "item.potion_blue", "item.herb", "item.fish", "item.ore",
	"item.etc", "item.card", "item.dagger", "item.sword", "item.bow", "item.staff", "item.mace", "item.axe",
	"item.hat", "item.armor", "item.shield", "item.garment", "item.boots", "item.ring",
]


func test_title_case() -> void:
	assert_eq(ItemText.title_case("gildscarab card"), "Gildscarab Card")
	assert_eq(ItemText.title_case("bill of birds"), "Bill of Birds")
	assert_eq(ItemText.title_case("of mice"), "Of Mice")
	assert_eq(ItemText.title_case(""), "")


func test_names() -> void:
	assert_eq(ItemText.base_name(WELCOME, 1), "Gold Coin", "name capitalised when no display")
	assert_eq(ItemText.base_name(WELCOME, 5), "Gildscarab Card")
	assert_eq(ItemText.base_name(WELCOME, 6), "Bill of Birds")
	assert_eq(ItemText.base_name(WELCOME, 100), "Copper Ore", "craft_items fallback")
	assert_eq(ItemText.base_name(WELCOME, 250), "Item 250")
	assert_eq(ItemText.base_name(WELCOME, 0), "")


func test_display_name_refine_and_slots() -> void:
	assert_eq(ItemText.display_name(WELCOME, 3, 4), "+4 Blade [3]")
	assert_eq(ItemText.display_name(WELCOME, 3), "Blade [3]")
	assert_eq(ItemText.display_name(WELCOME, 4, 2), "+2 Hat", "no slots, no brackets")
	assert_eq(ItemText.display_name(WELCOME, 5), "Gildscarab Card")
	assert_eq(ItemText.display_name(WELCOME, 2, 3), "Red Potion", "refine only on gear")
	assert_eq(ItemText.stack_name(WELCOME, 2, 5), "Red Potion x5")
	assert_eq(ItemText.stack_name(WELCOME, 8, 1, 7), "+7 Knife [1]")


func test_icon_keys_and_fallbacks() -> void:
	assert_eq(ItemText.icon_key(WELCOME, 3), "item.sword")
	assert_eq(ItemText.icon_key(WELCOME, 4), "item.hat", "armor fallback by slot")
	assert_eq(ItemText.icon_key(WELCOME, 8), "item.dagger", "weapon fallback by class")
	assert_eq(ItemText.icon_key(WELCOME, 6), "item.etc")
	assert_eq(ItemText.icon_key(WELCOME, 250), "item.etc", "unknown id")
	assert_eq(ItemText.icon_key(WELCOME, 0), "")
	assert_eq(ItemText.icon_by_kind(5, 0), "item.card")
	assert_eq(ItemText.icon_by_kind(3, 99), "item.sword")
	var tab: PackedStringArray = ItemText.icon_table(WELCOME)
	assert_eq(tab.size(), ITEMS.size())
	assert_eq(tab[0], "")
	assert_eq(tab[7], "", "reserved row")
	assert_eq(tab[4], "item.hat")


func test_asset_slots_take_the_welcome_icons() -> void:
	AssetSlots.set_item_icons(ItemText.icon_table(WELCOME))
	assert_eq(AssetSlots.item_key(8), "item.dagger")
	assert_eq(AssetSlots.item_key(2), "item.potion_red")
	assert_eq(AssetSlots.item_key(7), "item.%s" % AssetSlots.ITEM_NAMES[7], "empty entry: legacy table")
	AssetSlots.set_item_icons(PackedStringArray())
	assert_eq(AssetSlots.item_key(8), "item.%s" % AssetSlots.ITEM_NAMES[8])


func test_every_item_icon_resolves_to_a_64px_file() -> void:
	var reg: AssetSlots = AssetSlots.load_default()
	for k: String in ENGINE_ICONS + ItemText.WEAPON_ICONS + ItemText.ARMOR_ICONS:
		assert_true(reg.has(k), "slot %s" % k)
		assert_true(ResourceLoader.exists(reg.path_of(k)), "imported %s" % k)
		var tex: Texture2D = reg.item_texture(k)
		assert_eq(tex.get_width(), 64, k)
	assert_eq(reg.item_texture("item.nothing_here").get_width(), 64, "missing key draws item.etc")
	assert_eq(reg.item_texture("").get_width(), 64)


func test_tooltip_gear() -> void:
	var t: String = ItemText.tooltip(WELCOME, 3, 4, [5, 0, 0, 0])
	var lines: PackedStringArray = t.split("\n")
	assert_eq(lines[0], "+4 Blade [3]")
	assert_eq(lines[1], "Sword, ATK 65", "53 + 4 x 3 at weapon level 2")
	assert_eq(lines[2], "Cards: Gildscarab Card (DEF +10, Max HP +1000)  (1 / 3 sockets)")
	assert_eq(lines[3], "Weight 60")
	var e: String = ItemText.tooltip(WELCOME, 3, 0, [], PackedStringArray(["Buy: 10 Ryn"]), "ATK 53 (nothing worn)")
	assert_eq(e, "Blade [3]\nATK 53 (nothing worn)\n3 empty sockets\nWeight 60\nBuy: 10 Ryn")
	assert_eq(ItemText.tooltip(WELCOME, 0), "")


func test_tooltip_card_and_consumable() -> void:
	var t: String = ItemText.tooltip(WELCOME, 5, 0, [], PackedStringArray(["Owned: 1"]))
	assert_eq(t, "Gildscarab Card\nCard\nShield card: DEF +10, Max HP +1000\n"
			+ "Drops from Gildscarab (MVP), 1 in 10,000 (Solmera Culvert)\nWeight 1\nOwned: 1")
	assert_true(ItemText.tooltip(WELCOME, 2).begins_with("Red Potion\nConsumable, heals 45 HP"))
	assert_eq(ItemText.chance_text(100), "1 in 100")


func test_hud_inventory_line_uses_display_names() -> void:
	var you: Dictionary = {"inv": [{"item": 2, "count": 5}, {"item": 3, "count": 1, "refine": 4}, {"item": 0}]}
	assert_eq(HudText.inv_text(you, WELCOME), "Items: Red Potion x5, +4 Blade [3]")
	assert_eq(HudText.inv_text(you), "Items: 2x5 3")
