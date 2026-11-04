extends TestCase
## ro4-pvp: Hud.arena_rect / Hud.pvp_badge_text and EntityLayer.pvp_marked (name colour of PvP-attackable players).


func _welcome() -> Dictionary:
	return {"world": {"pvp": {"arena": [229, 271, 242, 284], "gate": [[241, 270], [242, 270]], "zones": [false]}}}


func test_arena_rect() -> void:
	var r: Rect2i = Hud.arena_rect(_welcome())
	assert_eq(r.position, Vector2i(229, 271))
	assert_eq(r.size, Vector2i(14, 14))
	assert_eq(Hud.arena_rect({}).size, Vector2i.ZERO)
	assert_eq(Hud.arena_rect({"world": {"pvp": {"arena": [1, 2]}}}).size, Vector2i.ZERO)


func test_badge_text() -> void:
	var arena: Rect2i = Hud.arena_rect(_welcome())
	assert_eq(Hud.pvp_badge_text({"pos": [230, 272]}, arena), "PvP")
	assert_eq(Hud.pvp_badge_text({"pos": [242, 284], "pvp_kills": 3, "pvp_deaths": 1}, arena), "PvP 3/1")
	assert_eq(Hud.pvp_badge_text({"pos": [10, 10], "pvp": true}, arena), "PK")
	assert_eq(Hud.pvp_badge_text({"pos": [10, 10], "pvp_kills": 2}, arena), "", "no badge outside without PK")
	assert_eq(Hud.pvp_badge_text({"pos": [230, 272]}, Rect2i()), "", "no arena known")


func test_pvp_marked() -> void:
	var before: Rect2i = EntityLayer.pvp_arena
	EntityLayer.pvp_arena = Hud.arena_rect(_welcome())
	assert_true(EntityLayer.pvp_marked({"pos": [229, 271]}))
	assert_true(not EntityLayer.pvp_marked({"pos": [241, 270]}), "the gate is not arena floor")
	assert_true(EntityLayer.pvp_marked({"pos": [10, 10], "pk": true}))
	assert_true(not EntityLayer.pvp_marked({"name": "x"}))
	EntityLayer.pvp_arena = Rect2i()
	assert_true(not EntityLayer.pvp_marked({"pos": [229, 271]}))
	EntityLayer.pvp_arena = before
