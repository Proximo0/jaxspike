extends TestCase

const RECIPE: Dictionary = {"id": 3, "inputs": [100, 100, 102], "output": 200, "skill": 1, "level": 2, "exp": 5, "station": 6}
const CRAFT: Dictionary = {"level": [1, 2, 1, 1, 1, 1], "exp": [0, 0, 0, 0, 0, 0]}


func _inv(entries: Array) -> Dictionary:
	var inv: Array = []
	for e: Array in entries:
		inv.append({"item": e[0], "count": e[1], "refine": 0, "cards": [0, 0, 0, 0]})
	return {"inv": inv}


func test_eligible() -> void:
	var counts: Dictionary = CraftPanel.inv_counts(_inv([[100, 2], [102, 1]]))
	var st: Dictionary = CraftPanel.recipe_status(RECIPE, CRAFT, counts, [6])
	assert_true(st["ok"])
	assert_eq((st["reasons"] as Array).size(), 0)


func test_each_missing_thing() -> void:
	var counts: Dictionary = CraftPanel.inv_counts(_inv([[100, 2], [102, 1]]))
	var no_level: Dictionary = CraftPanel.recipe_status(RECIPE, {"level": [1, 1, 1, 1, 1, 1]}, counts, [6])
	assert_false(no_level["ok"])
	assert_false(no_level["level_ok"])
	var few: Dictionary = CraftPanel.recipe_status(RECIPE, CRAFT, CraftPanel.inv_counts(_inv([[100, 1], [102, 1]])), [6])
	assert_false(few["inputs_ok"])
	var no_station: Dictionary = CraftPanel.recipe_status(RECIPE, CRAFT, counts, [7])
	assert_false(no_station["station_ok"])
	var anywhere: Dictionary = CraftPanel.recipe_status({"inputs": [100], "skill": 0, "level": 1, "station": -1}, CRAFT, counts, [])
	assert_true(anywhere["ok"])


func test_level_fraction() -> void:
	var table: Array = [0, 50, 200, 450]
	assert_near(CraftPanel.level_fraction(2, 125, table), 0.5)
	assert_near(CraftPanel.level_fraction(4, 999, table), 1.0)
	assert_near(CraftPanel.level_fraction(1, 0, []), 0.0)
	assert_near(CraftPanel.level_fraction(1, 25, table), 0.5)


func test_gather_adjacency() -> void:
	var store := EntityStore.new()
	store.apply_snap({"tick": 1, "full": true, "ents": [
		{"id": 9000, "kind": 201, "pos": [10, 11], "hp": 1},
		{"id": 9001, "kind": 204, "pos": [20, 20], "hp": 1},
		{"id": 9002, "kind": 202, "pos": [12, 10], "hp": 0},
		{"id": 9003, "kind": 206, "pos": [9, 10], "hp": 1},
		{"id": 9004, "kind": 205, "pos": [10, 9], "hp": 1, "item": 105},
	]})
	assert_true(CraftPanel.can_gather(store, Vector2i(10, 10)))
	assert_false(CraftPanel.can_gather(store, Vector2i(20, 22)))
	assert_false(CraftPanel.can_gather(store, Vector2i(12, 11)), "depleted coal only")
	assert_eq(CraftPanel.nearby_station_kinds(store, Vector2i(10, 10)), [6] as Array[int])
	assert_eq(CraftPanel.nodes_near(store, Vector2i(10, 10)).size(), 2, "drop sharing a node kind is ignored")
