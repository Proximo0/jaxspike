extends TestCase


func test_acquire_is_stable_and_dense() -> void:
	var p := SlotPool.new()
	assert_eq(p.acquire(40), 0)
	assert_eq(p.acquire(7), 1)
	assert_eq(p.acquire(40), 0)
	assert_eq(p.active_count(), 2)
	assert_eq(p.capacity(), 2)


func test_release_reuses_lowest_free_slot() -> void:
	var p := SlotPool.new()
	for id: int in [10, 11, 12, 13]:
		p.acquire(id)
	assert_eq(p.release(12), 2)
	assert_eq(p.release(10), 0)
	assert_eq(p.release(99), -1)
	assert_eq(p.acquire(50), 0)
	assert_eq(p.acquire(51), 2)
	assert_eq(p.acquire(52), 4)
	assert_eq(p.capacity(), 5)
	assert_false(p.has_id(10))
	assert_eq(p.slot_of(10), -1)
	assert_eq(p.active_count(), 5)
