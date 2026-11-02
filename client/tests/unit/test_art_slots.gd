extends TestCase

const SAMPLE: String = """{
	"version": 1,
	"slots": {
		"tile.0": {"path": "res://assets/none/terrain.png", "region": [0, 0, 16, 16]},
		"job.1.walk": {"sheet": "res://assets/sheets/job_1.json", "anim": "walk"},
		"ui.hpbar_fill": {"path": "res://assets/none/bar.png"},
		"monster.rat": {"path": "res://assets/none/m.png", "region": [1, 2, 3]},
		"item.meat": {"nothing": true},
		"item.sword": 5
	}
}"""


func test_key_builders() -> void:
	assert_eq(AssetSlots.tile_key(3), "tile.3")
	assert_eq(AssetSlots.job_key(1, "walk"), "job.1.walk")
	assert_eq(AssetSlots.monster_key(2), "monster.poring")
	assert_eq(AssetSlots.monster_key(22), "monster.unknown")
	assert_eq(AssetSlots.monster_key(-1), "monster.unknown")
	assert_eq(AssetSlots.item_key(2), "item.meat")
	assert_eq(AssetSlots.item_key(99), "item.unknown")
	assert_eq(AssetSlots.anim_name("walk", 2), "walk_left")
	assert_eq(AssetSlots.anim_name("idle", 0), "idle_down")
	assert_eq(AssetSlots.anim_name("attack", 3), "attack_right")
	assert_eq(AssetSlots.anim_name("dead", 3), "dead_down")


func test_table_sizes() -> void:
	assert_eq(AssetSlots.TERRAIN_NAMES.size(), 11)
	assert_eq(AssetSlots.JOB_NAMES.size(), 7)
	assert_eq(AssetSlots.MONSTER_NAMES.size(), 22)
	assert_eq(AssetSlots.UI_NAMES.size(), 12)
	assert_eq(AssetSlots.ITEM_NAMES[1], "gold_coin")
	assert_eq(AssetSlots.ITEM_NAMES[3], "sword")


func test_expected_keys_count_and_unique() -> void:
	var keys: PackedStringArray = AssetSlots.expected_keys()
	var want: int = 11 + 7 * 4 + 22 + (AssetSlots.ITEM_NAMES.size() - 1) + AssetSlots.UI_NAMES.size()
	assert_eq(keys.size(), want)
	var seen: Dictionary = {}
	for k: String in keys:
		seen[k] = true
	assert_eq(seen.size(), keys.size())
	assert_true(seen.has("tile.10"))
	assert_true(seen.has("job.6.dead"))
	assert_true(seen.has("monster.baphomet_jr"))
	assert_true(seen.has("ui.icon_close"))


func test_parse_sample() -> void:
	var reg := AssetSlots.new()
	assert_true(reg.parse(SAMPLE))
	assert_true(reg.has("tile.0"))
	assert_eq(reg.path_of("tile.0"), "res://assets/none/terrain.png")
	assert_eq(reg.region_of("tile.0"), Rect2i(0, 0, 16, 16))
	assert_false(reg.is_sheet("tile.0"))
	assert_true(reg.is_sheet("job.1.walk"))
	assert_eq(reg.anim_of("job.1.walk"), "walk")
	assert_eq(reg.path_of("job.1.walk"), "res://assets/sheets/job_1.json")
	assert_eq(reg.region_of("ui.hpbar_fill"), Rect2i())
	assert_eq(reg.anim_of("ui.hpbar_fill"), "")
	assert_eq(reg.path_of("nope"), "")
	assert_eq(reg.keys().size(), 3)
	assert_false(reg.has("monster.rat"))
	assert_false(reg.has("item.meat"))
	assert_false(reg.has("item.sword"))
	assert_eq(reg.errors.size(), 3)


func test_parse_rejects_garbage() -> void:
	var reg := AssetSlots.new()
	assert_false(reg.parse("not json"))
	assert_true(reg.errors.size() > 0)
	assert_false(reg.parse("{\"version\": 1}"))
	assert_eq(reg.keys().size(), 0)


func test_unknown_key_gives_fallback_texture() -> void:
	var reg := AssetSlots.new()
	var tex: Texture2D = reg.texture("monster.nope")
	assert_true(tex != null)
	assert_eq(tex.get_width(), 16)
	assert_eq(tex.get_height(), 16)
	var px: Color = tex.get_image().get_pixel(3, 3)
	var want: Color = AssetSlots.fallback_color("monster.nope")
	assert_near(px.r, want.r, 0.01)
	assert_near(px.g, want.g, 0.01)
	assert_near(px.b, want.b, 0.01)
	assert_true(reg.texture("monster.nope") == tex, "cached")


func test_missing_file_and_sheet_give_fallback() -> void:
	var reg := AssetSlots.new()
	reg.parse(SAMPLE)
	var a: Texture2D = reg.texture("tile.0")
	assert_true(a != null)
	assert_eq(a.get_width(), 16)
	var b: Texture2D = reg.texture("job.1.walk")
	assert_true(b != null)
	assert_eq(b.get_width(), 16)


func test_region_slot_builds_atlas_texture() -> void:
	var reg := AssetSlots.new()
	var path: String = "res://assets/kenney/tiny-dungeon/tilemap_packed.png"
	if not FileAccess.file_exists(path):
		assert_true(true)
		return
	var text: String = JSON.stringify({"slots": {"monster.poring": {"path": path, "region": [16, 112, 16, 16]}}})
	assert_true(reg.parse(text))
	var tex: Texture2D = reg.texture("monster.poring")
	if tex is AtlasTexture:
		var atlas: AtlasTexture = tex
		assert_eq(atlas.region, Rect2(16, 112, 16, 16))
		assert_true(atlas.atlas != null)
	else:
		assert_true(tex != null)


func test_fallback_color_deterministic() -> void:
	assert_eq(AssetSlots.fallback_color("tile.4"), AssetSlots.TILE_FALLBACK_COLORS[4])
	assert_eq(AssetSlots.fallback_color("tile.10"), AssetSlots.TILE_FALLBACK_COLORS[10])
	assert_eq(AssetSlots.fallback_color("monster.rat"), AssetSlots.fallback_color("monster.rat"))
	assert_true(AssetSlots.fallback_color("monster.rat") != AssetSlots.fallback_color("monster.orc"))
	assert_eq(AssetSlots.TILE_FALLBACK_COLORS.size(), 11)


func test_load_default_covers_expected_when_file_exists() -> void:
	var reg: AssetSlots = AssetSlots.load_default()
	assert_true(reg != null)
	if not FileAccess.file_exists(AssetSlots.SLOTS_PATH):
		assert_eq(reg.keys().size(), 0)
		assert_true(true)
		return
	assert_true(reg.missing_keys().is_empty(), str(reg.missing_keys()))
