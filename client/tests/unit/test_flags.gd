extends TestCase


func test_state_hash_changes_with_provider() -> void:
	var f: Node = load("res://game/autoload/flags.gd").new()
	var before: int = f.state_hash()
	f.register_hash(func() -> int: return 7)
	assert_true(f.state_hash() != before)
	f.free()
