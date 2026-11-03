extends TestCase


func test_pop_waits_rises_and_expires() -> void:
	var p := DamagePops.new()
	p.add(7, 42, 1000.0)
	assert_eq(p.active(999.0).size(), 0)
	var a: Array[Dictionary] = p.active(1000.0)
	assert_eq(a.size(), 1)
	assert_eq(int(a[0]["dmg"]), 42)
	assert_near(float(a[0]["rise"]), 0.0)
	var later: Array[Dictionary] = p.active(1000.0 + DamagePops.LIFE_MS * 0.5)
	assert_true(float(later[0]["rise"]) > 0.5)
	assert_eq(p.active(1000.0 + DamagePops.LIFE_MS).size(), 0)
	assert_eq(p.pending(), 0)


func test_pick_nearest_swinging_attacker() -> void:
	var c: Array[Dictionary] = [
		{"id": 1, "cell": Vector2(5, 0), "start_ms": 1000.0, "dur_ms": 500.0},
		{"id": 2, "cell": Vector2(1, 0), "start_ms": 1000.0, "dur_ms": 500.0},
		{"id": 3, "cell": Vector2(0, 1), "start_ms": 200.0, "dur_ms": 500.0},  # swing over long ago
		{"id": 4, "cell": Vector2(30, 0), "start_ms": 1000.0, "dur_ms": 500.0},  # too far
	]
	assert_eq(int(DamagePops.pick_attacker(Vector2.ZERO, 1000.0, c)["id"]), 2)
	assert_true(DamagePops.pick_attacker(Vector2.ZERO, 5000.0, c).is_empty())
