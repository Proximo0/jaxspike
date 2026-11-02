extends TestCase


class DebugTestStubStore extends RefCounted:
	var data: Dictionary = {"tick": 1}

	func snapshot() -> Dictionary:
		return data.duplicate(true)

	func restore(d: Dictionary) -> void:
		data = d.duplicate(true)


func _fill(df: DebugFrame, n: int) -> void:
	for i: int in range(n):
		df.begin(i)
		df.note("rx_bytes", i * 10)
		df.note("you_pos", [i, i])
		df.end()


func test_ring_keeps_last_600() -> void:
	var df := DebugFrame.new()
	_fill(df, 1000)
	assert_eq(df.size(), 600)
	assert_eq(df.total_frames(), 1000)
	var fr: Array[Dictionary] = df.frames()
	assert_eq(int(fr[0]["tick"]), 400)
	assert_eq(int(fr[599]["tick"]), 999)
	assert_eq(int(df.last()["rx_bytes"]), 9990)


func test_frame_has_contract_keys() -> void:
	var df := DebugFrame.new()
	df.begin(7)
	df.end()
	for k: String in ["tick", "frame", "rx_bytes", "ents", "you_pos", "latency_ms"]:
		assert_true(df.last().has(k), k)
	assert_eq(int(df.last()["tick"]), 7)


func test_note_and_end_without_begin_are_ignored() -> void:
	var df := DebugFrame.new()
	df.note("x", 1)
	df.end()
	assert_eq(df.size(), 0)


func test_snapshot_restore_round_trip() -> void:
	var df := DebugFrame.new()
	var store := DebugTestStubStore.new()
	df.attach_store(store)
	_fill(df, 650)
	store.data = {"tick": 42, "ents": {"3": {"hp": 5}}}
	var snap: Dictionary = df.snapshot()
	_fill(df, 50)
	store.data = {"tick": 99}
	assert_true(df.restore(snap))
	assert_eq(df.size(), 600)
	assert_eq(df.total_frames(), 650)
	assert_eq(int(df.last()["tick"]), 649)
	assert_eq(int(store.data["tick"]), 42)
	assert_eq(df.snapshot(), snap)


func test_restore_rejects_garbage() -> void:
	var df := DebugFrame.new()
	_fill(df, 3)
	assert_false(df.restore({"version": 9}))
	assert_eq(df.size(), 3)


func test_round_trip_with_entity_store() -> void:
	var store := EntityStore.new()
	store.apply_snap({"tick": 5, "full": true, "ents": [{"id": 700, "kind": 1, "pos": [3, 4], "hp": 9, "max_hp": 9}],
		"you": {"pos": [1, 2], "hp": 10, "max_hp": 10}})
	var df := DebugFrame.new()
	df.attach_store(store)
	var snap: Dictionary = df.snapshot()
	store.clear()
	assert_eq(store.count(), 0)
	assert_true(df.restore(snap))
	assert_eq(store.count(), 1)
	assert_eq(store.tick, 5)
