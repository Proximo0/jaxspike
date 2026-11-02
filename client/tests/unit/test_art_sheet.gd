extends TestCase

const SHEET_JSON: String = """
{"image": "res://none.png", "frame_w": 8, "frame_h": 8,
 "animations": {
  "idle_down": {"frames": [[0, 0]], "fps": 4, "loop": true},
  "walk_down": {"frames": [[0, 0], [1, 0]], "fps": 6, "loop": true},
  "attack_down": {"frames": [[2, 0]], "fps": 8, "loop": false},
  "dead_down": {"frames": [[3, 0]], "fps": 1, "loop": false}}}
"""


func _texture() -> ImageTexture:
	var img: Image = Image.create(32, 16, false, Image.FORMAT_RGBA8)
	for row: int in range(2):
		for col: int in range(4):
			var c: Color = Color(0.1 + 0.2 * float(col), 0.2 + 0.3 * float(row), 0.5, 1.0)
			img.fill_rect(Rect2i(col * 8, row * 8, 8, 8), c)
	return ImageTexture.create_from_image(img)


func test_parse_defaults_and_values() -> void:
	var sheet: Dictionary = SpriteSheetLoader.parse(SHEET_JSON)
	assert_false(sheet.is_empty())
	assert_eq(sheet["frame_w"], 8)
	assert_eq(sheet["margin"], 0)
	assert_eq(sheet["spacing"], 0)
	var anims: Dictionary = sheet["animations"]
	assert_eq(anims.size(), 4)
	var errs: PackedStringArray = sheet["errors"]
	assert_eq(errs.size(), 0)


func test_parse_rejects_bad_input() -> void:
	assert_true(SpriteSheetLoader.parse("[1, 2]").is_empty())
	assert_true(SpriteSheetLoader.parse("not json").is_empty())
	assert_true(SpriteSheetLoader.parse("{\"frame_w\": 8, \"frame_h\": 8, \"animations\": {}}").is_empty())
	assert_true(SpriteSheetLoader.parse("{\"image\": \"a\", \"frame_h\": 8, \"animations\": {}}").is_empty())
	assert_true(SpriteSheetLoader.parse("{\"image\": \"a\", \"frame_w\": 8, \"frame_h\": 8, \"animations\": 3}").is_empty())


func test_parse_skips_bad_animation() -> void:
	var text: String = "{\"image\": \"a\", \"frame_w\": 8, \"frame_h\": 8, \"animations\": {" \
			+ "\"ok\": {\"frames\": [[0, 0]]}, \"empty\": {\"frames\": []}, \"bad\": {\"frames\": [[0]]}}}"
	var sheet: Dictionary = SpriteSheetLoader.parse(text)
	var anims: Dictionary = sheet["animations"]
	assert_eq(anims.size(), 1)
	var ok: Dictionary = anims["ok"]
	assert_near(float(ok["fps"]), 6.0)
	assert_eq(ok["loop"], true)
	var errs: PackedStringArray = sheet["errors"]
	assert_eq(errs.size(), 2)
	assert_true(errs.has("empty"))
	assert_true(errs.has("bad"))


func test_frame_rect_margin_spacing() -> void:
	var sheet: Dictionary = {"frame_w": 16, "frame_h": 16, "margin": 1, "spacing": 2}
	assert_eq(SpriteSheetLoader.frame_rect(sheet, 0, 0), Rect2i(1, 1, 16, 16))
	assert_eq(SpriteSheetLoader.frame_rect(sheet, 2, 1), Rect2i(37, 19, 16, 16))
	var plain: Dictionary = {"frame_w": 8, "frame_h": 8}
	assert_eq(SpriteSheetLoader.frame_rect(plain, 1, 1), Rect2i(8, 8, 8, 8))


func test_build_frames() -> void:
	var sheet: Dictionary = SpriteSheetLoader.parse(SHEET_JSON)
	var frames: SpriteFrames = SpriteSheetLoader.build_frames(sheet, _texture())
	assert_false(frames.has_animation("default"))
	assert_eq(frames.get_animation_names().size(), 4)
	assert_eq(frames.get_frame_count("walk_down"), 2)
	assert_eq(frames.get_frame_count("idle_down"), 1)
	assert_near(frames.get_animation_speed("walk_down"), 6.0)
	assert_near(frames.get_animation_speed("attack_down"), 8.0)
	assert_true(frames.get_animation_loop("walk_down"))
	assert_false(frames.get_animation_loop("attack_down"))
	assert_false(frames.get_animation_loop("dead_down"))
	var atlas: AtlasTexture = frames.get_frame_texture("walk_down", 1) as AtlasTexture
	assert_true(atlas != null)
	if atlas != null:
		assert_eq(atlas.region, Rect2(8, 0, 8, 8))
		assert_true(atlas.atlas != null)


func test_make_sprite_plays_anim() -> void:
	var sheet: Dictionary = SpriteSheetLoader.parse(SHEET_JSON)
	var frames: SpriteFrames = SpriteSheetLoader.build_frames(sheet, _texture())
	assert_true(SpriteSheetLoader.has_anim(frames, "walk_down"))
	assert_false(SpriteSheetLoader.has_anim(frames, "walk_up"))
	var sprite: AnimatedSprite2D = SpriteSheetLoader.make_sprite(frames, "attack_down")
	assert_true(sprite.centered)
	assert_eq(String(sprite.animation), "attack_down")
	sprite.free()
	var fallback: AnimatedSprite2D = SpriteSheetLoader.make_sprite(frames, "nope")
	assert_true(frames.has_animation(fallback.animation))
	fallback.free()


func test_load_sheet_missing_is_null() -> void:
	assert_true(SpriteSheetLoader.load_sheet("res://assets/sheets/no_such_sheet.json") == null)
