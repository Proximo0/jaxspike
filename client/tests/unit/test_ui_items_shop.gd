extends TestCase

const WELCOME: Dictionary = {
	"items": [{}, {"name": "Gold", "kind": 1, "buy": 0, "sell": 0}, {"name": "Red Potion", "kind": 2, "buy": 50, "sell": 25},
		{"name": "Sword", "kind": 3, "buy": 100, "sell": 50}, {"name": "Cap", "kind": 4, "buy": 1, "sell": 1}],
	"world": {"shops": [{"tile": [10, 10], "town": 1}]},
	"shops": [{"town": 0, "name": "Prontera", "items": [2]}, {"town": 1, "name": "Geffen", "items": [0, 2, 3]}],
	"skills": {"names": ["a", "Discount", "Overcharge"]},
}


func _you() -> Dictionary:
	var inv: Array = []
	for i: int in range(4):
		inv.append({"item": 0, "count": 0, "refine": 0, "cards": [0, 0, 0, 0]})
	inv[0] = {"item": 2, "count": 4, "refine": 0, "cards": [0, 0, 0, 0]}
	inv[1] = {"item": 3, "count": 1, "refine": 0, "cards": [0, 0, 0, 0]}
	return {"inv": inv, "equip": [1, -1, -1, -1, -1, -1, -1, -1], "learned": [[1, 5], [2, 0]]}


func test_near_shop() -> void:
	var shops: Array = ShopPanel.shop_tiles(WELCOME)
	assert_eq(ShopPanel.near_shop(Vector2i(10, 10), shops), 1)
	assert_eq(ShopPanel.near_shop(Vector2i(11, 11), shops), 1)
	assert_eq(ShopPanel.near_shop(Vector2i(12, 10), shops), -1)
	assert_eq(ShopPanel.near_shop(Vector2i(10, 10), []), -1)


func test_price_pct() -> void:
	assert_eq(ShopPanel.price_pct(0), 0)
	assert_eq(ShopPanel.price_pct(1), 7)
	assert_eq(ShopPanel.price_pct(5), 15)
	assert_eq(ShopPanel.price_pct(10), 24)


func test_prices() -> void:
	assert_eq(ShopPanel.buy_price(50, 15), 42)
	assert_eq(ShopPanel.buy_price(1, 24), 1)
	assert_eq(ShopPanel.buy_price(0, 0), 1)
	assert_eq(ShopPanel.sell_price(25, 15), 28)
	assert_eq(ShopPanel.sell_price(25, 0), 25)


func test_skill_pct() -> void:
	assert_eq(ShopPanel.skill_pct(WELCOME, _you(), "Discount"), 15)
	assert_eq(ShopPanel.skill_pct(WELCOME, _you(), "Overcharge"), 0)
	assert_eq(ShopPanel.skill_pct(WELCOME, _you(), "Nope"), 0)
	assert_eq(ShopPanel.skill_pct({}, _you(), "Discount"), 0)


func test_buy_rows() -> void:
	var rows: Array[Dictionary] = ShopPanel.buy_rows(WELCOME, 1, _you())
	assert_eq(rows.size(), 2)
	assert_eq(rows[0]["item"], 2)
	assert_eq(rows[0]["name"], "Red Potion")
	assert_eq(rows[0]["price"], 42)
	assert_eq(rows[1]["price"], 85)
	assert_eq(ShopPanel.buy_rows(WELCOME, 0, {}).size(), 1)
	assert_eq(ShopPanel.buy_rows(WELCOME, 0, {})[0]["price"], 50)


func test_sell_rows_skip_worn() -> void:
	var rows: Array[Dictionary] = ShopPanel.sell_rows(WELCOME, _you())
	assert_eq(rows.size(), 1)
	assert_eq(rows[0]["slot"], 0)
	assert_eq(rows[0]["count"], 4)
	assert_eq(rows[0]["price"], 25)


func test_messages() -> void:
	assert_eq(ShopPanel.buy_message(2), {"t": "buy", "item": 2})
	assert_eq(ShopPanel.sell_message(0), {"t": "sell", "slot": 0})
