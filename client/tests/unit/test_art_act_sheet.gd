extends TestCase

const ACT: String = '{"frame_w": 8, "frame_h": 10, "anchor": [4, 9], "atk_frame": 2, "actions": {' \
		+ '"idle": {"row0": 0, "frames": 2, "delay_ms": 200, "mirror": true},' \
		+ '"attack": {"row0": 5, "frames": 4, "delay_ms": 100, "mirror": true},' \
		+ '"bad": {"frames": 0}}}'


func _tex(w: int, h: int) -> Texture2D:
	return ImageTexture.create_from_image(Image.create(w, h, false, Image.FORMAT_RGBA8))


func test_parse_fields() -> void:
	var m: Dictionary = ActSheet.parse(ACT)
	assert_eq(int(m["frame_w"]), 8)
	assert_eq(m["anchor"], Vector2(4, 9))
	assert_eq(int(m["atk_frame"]), 2)
	assert_true((m["actions"] as Dictionary).has("idle"))
	assert_false((m["actions"] as Dictionary).has("bad"))
	assert_true(ActSheet.parse("{").is_empty())
	assert_true(ActSheet.parse('{"frame_w": 8}').is_empty())


func test_build_five_rows_mirrored() -> void:
	var t: Texture2D = _tex(32, 100)  # 4 columns, 10 rows: idle rows 0..4, attack rows 5..9
	var s: ActSheet = ActSheet.build(ActSheet.parse(ACT), {"idle": t, "attack": t})
	assert_true(s.has_action("idle"))
	assert_eq(int(s.actions["attack"]["rows"]), 5)
	assert_eq(s.frames.get_frame_count(&"attack_4"), 4)
	assert_false(s.frames.has_animation(&"attack_5"))
	var r: Dictionary = s.resolve(ClipSelect.ATTACK, 6, true)
	assert_eq(r["anim"], &"attack_2")
	assert_true(bool(r["flip"]))
	var at: AtlasTexture = s.frame_texture(&"attack_0", 1) as AtlasTexture
	assert_eq(at.region, Rect2(8, 50, 8, 10))


func test_missing_action_falls_back() -> void:
	var t: Texture2D = _tex(32, 100)
	var s: ActSheet = ActSheet.build(ActSheet.parse(ACT), {"idle": t, "attack": t})
	assert_eq(String(s.resolve(ClipSelect.WALK, 0, true)["name"]), "idle")
	assert_eq(String(s.resolve(ClipSelect.CAST, 0, true)["name"]), "attack")
	assert_eq(String(s.resolve(ClipSelect.DEAD, 0, true)["name"]), "idle")


func test_load_act_missing_file_is_null() -> void:
	assert_eq(ActSheet.load_act("res://does/not/exist.json"), null)


func test_loads_art_anim_layout_when_present() -> void:
	# art-anim's sheets: res://assets/anim/<kind>/act.json with absolute res:// images per action
	if not FileAccess.file_exists("res://assets/anim/blobbin/act.json"):
		return
	var s: ActSheet = ActSheet.load_act("res://assets/anim/blobbin/act.json")
	assert_true(s != null, "blobbin act loads")
	assert_true(s.has_action("walk"))
	assert_eq(int(s.actions["walk"]["rows"]), 5)
	assert_true(s.frame_texture(s.resolve(ClipSelect.WALK, 6, false)["anim"], 0) != null)
