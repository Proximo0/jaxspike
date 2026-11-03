extends TestCase


func test_clip_tables_cover_every_action() -> void:
	for a: int in range(ClipSelect.ACTION_COUNT):
		var p: Dictionary = PlaceholderClips.clip(a, true)
		assert_true(int(p["frames"]) >= 1, "player %d" % a)
		var m: Dictionary = PlaceholderClips.clip(a, false)
		assert_true(PlaceholderClips.MONSTER_ACTIONS.has(String(m["name"])), "monster %d" % a)


func test_poses_are_distinct() -> void:
	var wind: Dictionary = PlaceholderClips.pose("attack", 1, 6)
	var swing: Dictionary = PlaceholderClips.pose("attack", PlaceholderClips.ATK_FRAME, 6)
	assert_true(float(swing["arm"]) > float(wind["arm"]) + 1.0, "atk frame swings forward")
	assert_true(float(PlaceholderClips.pose("hurt", 0, 2)["flash"]) > 0.0)
	assert_near(float(PlaceholderClips.pose("dead", 2, 3)["lie"]), 1.0)
	var w0: float = float(PlaceholderClips.pose("walk", 0, 8)["leg"])
	var w2: float = float(PlaceholderClips.pose("walk", 2, 8)["leg"])
	assert_true(absf(w2 - w0) > 0.5, "legs stride")


func test_make_sheet_has_every_clip_in_the_act_layout() -> void:
	var sheet: ActSheet = PlaceholderClips.make_sheet(true, 2)
	for nm: String in PlaceholderClips.PLAYER_ACTIONS:
		assert_true(sheet.has_action(nm), nm)
		assert_eq(int(sheet.actions[nm]["rows"]), PlaceholderClips.ROWS)
	assert_eq(sheet.frame_w, PlaceholderClips.FRAME_W)
	assert_eq(sheet.frames.get_frame_count(&"walk_0"), 8)
	var r: Dictionary = sheet.resolve(ClipSelect.WALK, 6, true)
	assert_eq(r["anim"], &"walk_2")
	assert_true(bool(r["flip"]))
	var m: ActSheet = PlaceholderClips.make_sheet(false, 0)
	assert_true(m.has_action("attack"))
	assert_false(m.has_action("sit"))


func test_painted_frames_have_outline_and_colour() -> void:
	var img: Image = PlaceholderClips.paint_action(false, 0, "idle")
	assert_eq(img.get_width(), 4 * PlaceholderClips.FRAME_W)
	var feet: Vector2 = PlaceholderClips.ANCHOR
	var body: Color = img.get_pixel(int(feet.x), int(feet.y) - 10)
	assert_true(body.a > 0.99, "opaque body above the feet")
	assert_true(body.r > 0.5, "Poring pink")
	var corner: Color = img.get_pixel(1, 1)
	assert_true(corner.a < 0.01, "transparent background")
	var dark: int = 0
	for x: int in range(PlaceholderClips.FRAME_W):
		var c: Color = img.get_pixel(x, int(feet.y) - 10)
		if c.a > 0.99 and c.get_luminance() < 0.15:
			dark += 1
	assert_true(dark >= 2, "outline on both sides")
