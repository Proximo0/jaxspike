extends TestCase
## The placeholder art files: slots.json parses, every path exists, every expected key is present.

const SLOTS_PATH: String = "res://assets/slots.json"
const PACK_DIRS: Array[String] = ["tiny-dungeon", "tiny-town", "roguelike-rpg-pack", "roguelike-characters",
	"ui-pack-rpg-expansion", "ui-pack"]
const MONSTERS: Array[String] = ["rat", "orc", "poring", "drops", "poporing", "lunatic", "fabre", "pupa", "chonchon",
	"willow", "rocker", "spore", "thief_bug", "condor", "picky", "peco_peco", "wolf", "hode", "orc_warrior", "zombie",
	"skeleton", "baphomet_jr"]
const ITEMS: Array[String] = ["gold_coin", "meat", "sword", "potion_red", "potion_blue", "dagger", "bow", "staff",
	"mace", "axe", "shield", "armor", "hat", "boots", "garment", "ring", "card", "ore", "herb", "fish"]
const UIS: Array[String] = ["panel", "panel_title", "button", "button_pressed", "button_hover", "slot", "hpbar_bg",
	"hpbar_fill", "spbar_fill", "expbar_fill", "cursor", "icon_close"]
const ANIMS: Array[String] = ["idle", "walk", "attack", "dead"]


func _slots() -> Dictionary:
	var text: String = FileAccess.get_file_as_string(SLOTS_PATH)
	var parsed: Variant = JSON.parse_string(text)
	if typeof(parsed) != TYPE_DICTIONARY:
		return {}
	var d: Dictionary = parsed
	var s: Variant = d.get("slots", {})
	if typeof(s) != TYPE_DICTIONARY:
		return {}
	return s


func _expected() -> PackedStringArray:
	var keys: PackedStringArray = []
	for i: int in range(11):
		keys.append("tile.%d" % i)
	for j: int in range(7):
		for a: String in ANIMS:
			keys.append("job.%d.%s" % [j, a])
	for m: String in MONSTERS:
		keys.append("monster." + m)
	for it: String in ITEMS:
		keys.append("item." + it)
	for n: int in range(1, 4):
		keys.append("item.%d" % n)
	for u: String in UIS:
		keys.append("ui." + u)
	return keys


func test_slots_parse() -> void:
	assert_true(FileAccess.file_exists(SLOTS_PATH), "slots.json exists")
	assert_true(_slots().size() > 0, "slots.json parses with slots")


func test_every_path_exists() -> void:
	var slots: Dictionary = _slots()
	for k: Variant in slots.keys():
		var slot: Dictionary = slots[k]
		var p: String = str(slot.get("path", slot.get("sheet", "")))
		assert_true(p != "" and FileAccess.file_exists(p), "path of %s: %s" % [str(k), p])


func test_expected_keys_present() -> void:
	var slots: Dictionary = _slots()
	var keys: PackedStringArray = _expected()
	assert_eq(keys.size(), 11 + 28 + 22 + 23 + 12)
	for k: String in keys:
		assert_true(slots.has(k), "missing key " + k)


func test_job_sheets() -> void:
	for j: int in range(7):
		var path: String = "res://assets/sheets/job_%d.json" % j
		var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
		assert_true(typeof(parsed) == TYPE_DICTIONARY, "job json parses " + path)
		if typeof(parsed) != TYPE_DICTIONARY:
			continue
		var d: Dictionary = parsed
		var anims: Dictionary = d.get("animations", {})
		assert_eq(anims.size(), 13, path)
		assert_true(FileAccess.file_exists(str(d.get("image", ""))), "job image " + path)


func test_licenses_name_each_pack() -> void:
	assert_true(FileAccess.file_exists("res://assets/LICENSES.md"), "LICENSES.md exists")
	var text: String = FileAccess.get_file_as_string("res://assets/LICENSES.md")
	for pack: String in PACK_DIRS:
		if DirAccess.dir_exists_absolute(ProjectSettings.globalize_path("res://assets/kenney/" + pack)):
			assert_true(text.contains("client/assets/kenney/" + pack), "LICENSES names " + pack)
