extends TestCase
## LabAnimFeed: deterministic, 200 entities, deltas, and the hurt / sit / pickup / dead / gone rows.

const SEED: int = 4242
const TICKS: int = 200


func _run(feed: LabAnimFeed, ticks: int) -> Array:
	var snaps: Array = []
	for i: int in range(ticks):
		snaps.append(feed.step())
	return snaps


func test_first_snap_is_full_with_200_ents() -> void:
	var feed := LabAnimFeed.new(SEED)
	var snap: Dictionary = feed.step()
	assert_true(bool(snap["full"]))
	assert_eq((snap["ents"] as Array).size(), 200)


func test_new_clips_appear() -> void:
	var feed := LabAnimFeed.new(SEED)
	var snaps: Array = _run(feed, TICKS)
	var hurt_with_h: int = 0
	var sit: int = 0
	var pickup: int = 0
	var dead: int = 0
	var gone_ids: Dictionary = {}
	var returned: int = 0
	for snap: Dictionary in snaps:
		for g: Variant in snap["gone"]:
			gone_ids[int(g)] = int(snap["tick"])
		for e: Variant in snap["ents"]:
			var row: Dictionary = e
			var a: int = int(row["a"])
			var id: int = int(row["id"])
			if a == 6 and row.has("h") and feed.is_walker(id):
				hurt_with_h += 1
			elif a == 2:
				sit += 1
			elif a == 3:
				pickup += 1
			elif a == 7:
				dead += 1
				assert_eq(int(row["hp"]), 0)
			if gone_ids.has(id) and int(snap["tick"]) > int(gone_ids[id]):
				returned += 1
	assert_true(hurt_with_h > 0, "hit walker row with h")
	assert_true(sit > 0, "sit row")
	assert_true(pickup > 0, "pickup row")
	assert_true(dead > 0, "dead row")
	assert_true(gone_ids.size() > 0, "gone id")
	assert_true(returned > 0, "gone id returns")


func test_same_seed_same_rows() -> void:
	var a: Array = _run(LabAnimFeed.new(SEED), TICKS)
	var b: Array = _run(LabAnimFeed.new(SEED), TICKS)
	assert_eq(a.size(), b.size())
	for i: int in range(a.size()):
		assert_eq((a[i] as Dictionary)["ents"], (b[i] as Dictionary)["ents"])
		assert_eq((a[i] as Dictionary)["gone"], (b[i] as Dictionary)["gone"])


func test_rows_per_tick_are_deltas() -> void:
	var feed := LabAnimFeed.new(SEED)
	var snaps: Array = _run(feed, TICKS)
	for i: int in range(1, snaps.size()):
		assert_true(((snaps[i] as Dictionary)["ents"] as Array).size() < 200, "tick %d is a delta" % (i + 1))
