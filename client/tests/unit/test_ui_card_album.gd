extends TestCase
## ro4-cards: CardAlbum pure helpers (table, owned counts, texts, announcement) and the CardSparkle star.

const CARDS: Array = [
	{"id": 57, "name": "Blobbin Card", "monster": "Blobbin", "kind": 0, "slot": 2, "slot_name": "Armor",
		"effects": [[5, 2], [10, 1]], "text": "LUK +2, FLEE +1", "chance": 1, "mvp": false,
		"where": "the fields near the towns", "icon": "card.blobbin"},
	{"id": 59, "name": "Cocoonling Card", "monster": "", "kind": -1, "slot": 2, "slot_name": "Armor",
		"effects": [[12, 700]], "text": "Max HP +700", "chance": 0, "mvp": false, "where": "", "icon": "card.cocoonling"},
	{"id": 92, "name": "Gildscarab Card", "monster": "Gildscarab", "kind": 18, "slot": 3, "slot_name": "Shield",
		"effects": [[7, 10]], "text": "DEF +10, Max HP +1000, VIT +3, Max SP -60", "chance": 1, "mvp": true,
		"where": "Solmera Culvert", "icon": "card.gildscarab"},
]


func _you() -> Dictionary:
	var inv: Array = []
	for i: int in range(16):
		inv.append({"item": 0, "count": 0, "refine": 0, "cards": [0, 0, 0, 0]})
	inv[0] = {"item": 57, "count": 2, "refine": 0, "cards": [0, 0, 0, 0]}
	inv[1] = {"item": 3, "count": 1, "refine": 0, "cards": [57.0, 92, 0, 0]}
	return {"inv": inv, "equip": [1, -1, -1, -1, -1, -1, -1, -1], "storage": [0, 92, 5]}


func test_ids_and_is_card() -> void:
	var ids: Dictionary = CardAlbum.ids_of(CARDS)
	assert_eq(ids.size(), 3)
	var keep: Dictionary = CardAlbum.card_ids
	CardAlbum.card_ids = ids
	assert_true(CardAlbum.is_card(92))
	assert_false(CardAlbum.is_card(3))
	CardAlbum.card_ids = keep


func test_art_path() -> void:
	assert_eq(CardAlbum.art_path("card.grukk_brute"), "res://assets/cards/grukk_brute.png")
	assert_eq(CardAlbum.art_path("item.card"), "")
	assert_eq(CardAlbum.art_path("card."), "")


func test_owned_counts_inventory_sockets_and_storage() -> void:
	var owned: Dictionary = CardAlbum.owned_counts(_you(), CardAlbum.ids_of(CARDS))
	assert_eq(int(owned.get(57, 0)), 3, "2 carried + 1 socketed")
	assert_eq(int(owned.get(92, 0)), 2, "1 socketed + 1 in storage")
	assert_false(owned.has(59))
	assert_false(owned.has(5), "storage non-card ids are ignored")
	assert_true(CardAlbum.owned_counts({}, {}).is_empty())


func test_rows_lit_and_title() -> void:
	var rows: Array[Dictionary] = CardAlbum.card_rows(CARDS, _you())
	assert_eq(rows.size(), 3)
	assert_eq(rows[0]["owned"], 3)
	assert_eq(rows[1]["owned"], 0)
	assert_true(rows[2]["mvp"])
	assert_eq(CardAlbum.album_title(2, 27), "Card Album (Alt+A)   2 / 27 found")


func test_detail_text() -> void:
	var rows: Array[Dictionary] = CardAlbum.card_rows(CARDS, _you())
	var t: String = CardAlbum.detail_text(rows[0])
	assert_true(t.contains("Armor card: LUK +2, FLEE +1"), t)
	assert_true(t.contains("Drops from Blobbin, 1 in 10,000 (the fields near the towns)"), t)
	assert_true(t.contains("Owned: 3"), t)
	assert_true(CardAlbum.detail_text(rows[1]).contains("No monster in Aurelune drops it yet."))
	assert_true(CardAlbum.detail_text(rows[1]).contains("Not found yet"))
	assert_true(CardAlbum.detail_text(rows[2]).begins_with("Gildscarab Card\nCard\n"), CardAlbum.detail_text(rows[2]))
	assert_true(CardAlbum.detail_text(rows[2]).contains("Drops from Gildscarab (MVP)"))


func test_chance_text() -> void:
	assert_eq(CardAlbum.chance_text(1), "1 in 10,000")
	assert_eq(CardAlbum.chance_text(100), "1 in 100")
	assert_eq(CardAlbum.chance_text(0), "")


func test_announce_line_and_banner_fade() -> void:
	var l: Dictionary = CardAlbum.announce_line({"t": "card", "item": 57, "finder": null,
		"text": "Someone found a Blobbin Card!", "mvp": false})
	assert_eq(l["text"], "Someone found a Blobbin Card!")
	assert_eq(l["kind"], "system")
	assert_eq(l["color"], CardAlbum.ANNOUNCE_COLOR)
	var m: Dictionary = CardAlbum.announce_line({"t": "card", "finder": "Ana", "mvp": true})
	assert_eq(m["text"], "Ana found a card!")
	assert_eq(m["color"], CardAlbum.MVP_ANNOUNCE_COLOR)
	assert_near(CardAlbum.banner_alpha(CardAlbum.BANNER_MS), 1.0)
	assert_near(CardAlbum.banner_alpha(CardAlbum.BANNER_FADE_MS * 0.5), 0.5)
	assert_near(CardAlbum.banner_alpha(0.0), 0.0)


func test_row_of() -> void:
	assert_eq(String(CardAlbum.row_of(CARDS, 92).get("name", "")), "Gildscarab Card")
	assert_true(CardAlbum.row_of(CARDS, 1).is_empty())


func test_album_node_announces_into_a_sibling_chat() -> void:
	var root := Control.new()
	(Engine.get_main_loop() as SceneTree).root.add_child(root)
	var chat: ChatBox = ChatBox.new()
	root.add_child(chat)
	var album: CardAlbum = (load("res://game/ui/card_album.tscn") as PackedScene).instantiate() as CardAlbum
	root.add_child(album)
	album.set_welcome({"t": "welcome", "cards": CARDS})
	album.on_message({"t": "card", "item": 92, "kind": 18, "finder": "Ana", "mvp": true,
		"text": "Ana found a Gildscarab Card!"})
	assert_eq(chat.lines.size(), 1)
	assert_eq(String(chat.lines[0]["text"]), "Ana found a Gildscarab Card!")
	assert_eq(int(album.last_announce.get("item", 0)), 92)
	assert_true(CardAlbum.is_card(57))
	root.free()


func test_sparkle_stars_twinkle_inside_the_tile() -> void:
	var seen_on: bool = false
	var seen_off: bool = false
	for t: int in range(0, 900, 50):
		for i: int in range(CardSparkle.STARS):
			var s: Vector3 = CardSparkle.star(i, float(t))
			assert_true(absf(s.x) < 0.5 and absf(s.y) < 0.8, "star stays near its drop")
			seen_on = seen_on or s.z > 0.1
			seen_off = seen_off or s.z < 0.02
	assert_true(seen_on and seen_off, "stars grow and vanish")
	assert_near(CardSparkle.star(0, 0.0).z, CardSparkle.star(0, CardSparkle.PERIOD_MS).z, 0.0001, "periodic")
