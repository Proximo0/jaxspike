extends TestCase
## CursorState: the RO cursor for what is under the mouse, and the images exist for every state.


func test_pick() -> void:
	assert_eq(CursorState.pick(ViewMath.CAT_MONSTER, 0), CursorState.ATTACK)
	assert_eq(CursorState.pick(ViewMath.CAT_DROP, 0), CursorState.PICKUP)
	assert_eq(CursorState.pick(ViewMath.CAT_NODE, 0), CursorState.TALK)
	assert_eq(CursorState.pick(CursorState.NO_ENTITY, 9), CursorState.TALK, "kafra tile")
	assert_eq(CursorState.pick(CursorState.NO_ENTITY, 6), CursorState.BLOCKED, "wall")
	assert_eq(CursorState.pick(CursorState.NO_ENTITY, -1), CursorState.BLOCKED, "outside the map")
	assert_eq(CursorState.pick(CursorState.NO_ENTITY, 1), CursorState.WALK)
	assert_eq(CursorState.pick(ViewMath.CAT_PLAYER, 1), CursorState.WALK, "other players: plain arrow")
	assert_eq(CursorState.pick(ViewMath.CAT_MONSTER, 6), CursorState.ATTACK, "monster wins over terrain")


func test_images() -> void:
	for s: int in range(CursorState.NAMES.size()):
		var img: Image = CursorState.make_image(s)
		assert_eq(img.get_size(), Vector2i(CursorState.SIZE, CursorState.SIZE))
		var opaque: int = 0
		for y: int in range(img.get_height()):
			for x: int in range(img.get_width()):
				if img.get_pixel(x, y).a > 0.0:
					opaque += 1
		assert_true(opaque > 40, "%s drawn (%d px)" % [CursorState.NAMES[s], opaque])
		var hs: Vector2 = CursorState.hotspot(s)
		assert_true(hs.x >= 0 and hs.y >= 0 and hs.x < CursorState.SIZE and hs.y < CursorState.SIZE)
