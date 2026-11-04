extends TestCase
## Round 2 animated sheets (docs/round2_contract.md 1.4): slots resolve, act json matches its pngs, frames move.

const MIN_FRAMES: int = 4
const MAX_FRAMES: int = 8
const DRAWN_ROWS: int = 5
const MIN_OPAQUE_PX: int = 80


func _load_json(path: String) -> Dictionary:
	var json := JSON.new()
	if json.parse(FileAccess.get_file_as_string(path)) != OK or not (json.data is Dictionary):
		return {}
	var d: Dictionary = json.data
	return d


func _load_png(path: String) -> Image:
	var img := Image.new()
	if img.load(ProjectSettings.globalize_path(path)) != OK:
		return null
	return img


func _frame(img: Image, fw: int, fh: int, row: int, col: int) -> Image:
	return img.get_region(Rect2i(col * fw, row * fh, fw, fh))


func _opaque_count(img: Image) -> int:
	var n: int = 0
	for y: int in range(img.get_height()):
		for x: int in range(img.get_width()):
			if img.get_pixel(x, y).a > 0.5:
				n += 1
	return n


func test_key_builders() -> void:
	assert_eq(AssetSlots.anim_key("poring", "walk"), "anim.poring.walk")
	assert_eq(AssetSlots.job_anim_key(1, 5), "anim.swordsman.attack")
	assert_eq(AssetSlots.job_anim_key(2, 8), "anim.mage.cast")
	assert_eq(AssetSlots.job_anim_key(99, 0), "anim.novice.idle")
	assert_eq(AssetSlots.job_anim_key(0, 42), "anim.novice.idle")
	assert_eq(AssetSlots.monster_anim_key(0, 1), "anim.poring.walk")
	assert_eq(AssetSlots.monster_anim_key(1, 7), "anim.orc_warrior.dead")
	assert_eq(AssetSlots.monster_anim_key(19, 8), "anim.eddga.attack")
	assert_eq(AssetSlots.monster_anim_key(5, 2), "anim.poporing.idle")
	assert_eq(AssetSlots.monster_anim_key(20, 0), "anim.unknown.idle")
	assert_eq(AssetSlots.ANIM_MONSTER_KINDS.size(), 20)
	assert_eq(AssetSlots.expected_anim_keys().size(), 7 * 9 + 20 * 5)


func test_default_registry_resolves_all_anim_slots() -> void:
	var reg: AssetSlots = AssetSlots.load_default()
	assert_eq(reg.errors.size(), 0, ", ".join(reg.errors))
	var missing: PackedStringArray = reg.missing_anim_keys()
	assert_eq(missing.size(), 0, ", ".join(missing))
	assert_true(reg.is_sheet("anim.thief.sit"))
	assert_eq(reg.anim_of("anim.thief.sit"), "sit")
	assert_eq(reg.path_of("anim.thief.sit"), "res://assets/anim/thief/act.json")
	assert_eq(reg.png_of("anim.thief.sit"), "res://assets/anim/thief/sit.png")
	# per-kind aliases client-anim's ActSheet looks up first (job index / engine monster kind -> act json)
	for j: int in range(AssetSlots.JOB_NAMES.size()):
		assert_eq(reg.path_of("anim.job.%d" % j), "res://assets/anim/%s/act.json" % AssetSlots.JOB_NAMES[j])
	for m: int in range(AssetSlots.ANIM_MONSTER_KINDS.size()):
		var want: String = "res://assets/anim/%s/act.json" % AssetSlots.ANIM_MONSTER_KINDS[m]
		assert_eq(reg.path_of("anim.monster.%d" % m), want)
	# old static slots keep working next to the anim ones
	assert_true(reg.has("tile.0"))
	assert_false(reg.is_sheet("tile.0"))
	assert_eq(reg.png_of("tile.0"), "")


func test_act_files_match_pngs() -> void:
	var reg: AssetSlots = AssetSlots.load_default()
	var kinds: Array[String] = []
	kinds.append_array(AssetSlots.JOB_NAMES)
	kinds.append_array(AssetSlots.ANIM_MONSTER_KINDS)
	for kind: String in kinds:
		var act_path: String = reg.path_of(AssetSlots.anim_key(kind, "idle"))
		var act: Dictionary = _load_json(act_path)
		assert_false(act.is_empty(), "%s: act json" % kind)
		if act.is_empty():
			continue
		var fw: int = int(act.get("frame_w", 0))
		var fh: int = int(act.get("frame_h", 0))
		var anchor: Array = act.get("anchor", [])
		assert_true(fw > 0 and fh > 0, "%s: frame size" % kind)
		assert_eq(anchor.size(), 2, "%s: anchor" % kind)
		assert_true(int(act.get("atk_frame", -1)) >= 0, "%s: atk_frame" % kind)
		var actions: Dictionary = act.get("actions", {})
		var want: Array[String] = AssetSlots.ANIM_ACTIONS if kind in AssetSlots.JOB_NAMES \
			else AssetSlots.ANIM_MONSTER_ACTIONS
		for a: String in want:
			assert_true(actions.has(a), "%s: action %s" % [kind, a])
			if not actions.has(a):
				continue
			var e: Dictionary = actions[a]
			var frames: int = int(e.get("frames", 0))
			assert_true(frames >= MIN_FRAMES and frames <= MAX_FRAMES, "%s.%s: frames %d" % [kind, a, frames])
			assert_true(int(e.get("delay_ms", 0)) > 0, "%s.%s: delay" % [kind, a])
			assert_eq(int(e.get("row0", -1)), 0, "%s.%s: row0" % [kind, a])
			assert_true(bool(e.get("mirror", false)), "%s.%s: mirror" % [kind, a])
			var png_path: String = reg.png_of(AssetSlots.anim_key(kind, a))
			assert_eq(String(e.get("image", "")), png_path, "%s.%s: image" % [kind, a])
			assert_eq(png_path, act_path.get_base_dir().path_join(String(e.get("file", ""))), "%s.%s" % [kind, a])
			var img: Image = _load_png(png_path)
			assert_true(img != null, "%s.%s: png loads" % [kind, a])
			if img == null:
				continue
			assert_eq(img.get_width(), fw * frames, "%s.%s: width" % [kind, a])
			assert_eq(img.get_height(), fh * DRAWN_ROWS, "%s.%s: height" % [kind, a])


func test_every_kind_draws_something_in_every_direction() -> void:
	var kinds: Array[String] = []
	kinds.append_array(AssetSlots.JOB_NAMES)
	kinds.append_array(AssetSlots.ANIM_MONSTER_KINDS)
	for kind: String in kinds:
		var act: Dictionary = _load_json("res://assets/anim/%s/act.json" % kind)
		var img: Image = _load_png("res://assets/anim/%s/idle.png" % kind)
		assert_true(img != null and not act.is_empty(), kind)
		if img == null or act.is_empty():
			continue
		for row: int in range(DRAWN_ROWS):
			var f: Image = _frame(img, int(act["frame_w"]), int(act["frame_h"]), row, 0)
			assert_true(_opaque_count(f) >= MIN_OPAQUE_PX, "%s row %d is near empty" % [kind, row])


func test_clips_have_real_motion() -> void:
	# a walk/attack/hurt/dead clip whose frames are all identical would play as a still
	for kind: String in ["novice", "poring", "wolf", "eddga"]:
		var act: Dictionary = _load_json("res://assets/anim/%s/act.json" % kind)
		var fw: int = int(act["frame_w"])
		var fh: int = int(act["frame_h"])
		for a: String in ["walk", "attack", "hurt", "dead"]:
			var img: Image = _load_png("res://assets/anim/%s/%s.png" % [kind, a])
			var e: Dictionary = act["actions"][a]
			var first: PackedByteArray = _frame(img, fw, fh, 0, 0).get_data()
			var moved: bool = false
			for col: int in range(1, int(e["frames"])):
				if _frame(img, fw, fh, 0, col).get_data() != first:
					moved = true
			assert_true(moved, "%s.%s frames all identical" % [kind, a])


func test_round3_sheets_are_hires_prerenders() -> void:
	# round 3 (ro3-art): every job and monster sheet is a Blender pre-render at 2x RO's native size
	# (jobs 96x128, monsters >= 96 px), its feet anchor sits in the lower part of the frame, and the hit frame
	# is frame 3 of the 6-frame attack (60 %).
	var kinds: Array[String] = []
	kinds.append_array(AssetSlots.JOB_NAMES)
	kinds.append_array(AssetSlots.ANIM_MONSTER_KINDS)
	for kind: String in kinds:
		var act: Dictionary = _load_json("res://assets/anim/%s/act.json" % kind)
		assert_false(act.is_empty(), kind)
		if act.is_empty():
			continue
		var fw: int = int(act["frame_w"])
		var fh: int = int(act["frame_h"])
		var is_job: bool = kind in AssetSlots.JOB_NAMES
		assert_true(fw >= 96 and fh >= (128 if is_job else 96), "%s: frame %dx%d is not hi-res" % [kind, fw, fh])
		var anchor: Array = act["anchor"]
		assert_true(float(anchor[1]) >= fh * 0.75 and float(anchor[1]) < fh, "%s: anchor y %s" % [kind, str(anchor[1])])
		assert_eq(int(act["atk_frame"]), 3, "%s: atk_frame" % kind)
		assert_eq(int(act["actions"]["attack"]["frames"]), 6, "%s: attack frames" % kind)
		assert_true(String(act.get("source", "")).contains("render_kind"), "%s: source tag" % kind)


func test_npc_sheets() -> void:
	var reg: AssetSlots = AssetSlots.load_default()
	for kind: String in ["npc_shop", "npc_kafra"]:
		var act: Dictionary = _load_json("res://assets/anim/%s/act.json" % kind)
		assert_false(act.is_empty(), "%s: act json" % kind)
		if act.is_empty():
			continue
		var fw: int = int(act["frame_w"])
		var fh: int = int(act["frame_h"])
		for a: String in ["idle", "talk"]:
			assert_true(reg.has(AssetSlots.anim_key(kind, a)), "%s.%s slot" % [kind, a])
			var e: Dictionary = act["actions"][a]
			assert_eq(int(e["frames"]), 4, "%s.%s frames" % [kind, a])
			var img: Image = _load_png("res://assets/anim/%s/%s.png" % [kind, a])
			assert_true(img != null, "%s.%s png" % [kind, a])
			if img == null:
				continue
			assert_eq(img.get_height(), fh * DRAWN_ROWS, "%s.%s height" % [kind, a])
			for row: int in range(DRAWN_ROWS):
				for col: int in range(4):
					assert_true(_opaque_count(_frame(img, fw, fh, row, col)) >= MIN_OPAQUE_PX, "%s.%s %d/%d empty" % [kind, a, row, col])
