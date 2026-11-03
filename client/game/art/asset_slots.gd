class_name AssetSlots
extends RefCounted
## Registry of art slots: maps game keys (tile.3, job.1.walk, monster.poring, item.meat, ui.panel) to textures.
## Reads `res://assets/slots.json`. A missing slot or file never yields null: a flat fallback texture is returned.

const SLOTS_PATH: String = "res://assets/slots.json"
const TERRAIN_NAMES: Array[String] = [
	"grass", "road", "floor", "temple", "water", "tree", "wall", "rock", "warp", "kafra", "station",
]
const JOB_NAMES: Array[String] = ["novice", "swordsman", "mage", "archer", "acolyte", "thief", "merchant"]
const ANIMS: Array[String] = ["idle", "walk", "attack", "dead"]
const DIRS: Array[String] = ["down", "up", "left", "right"]
const MONSTER_NAMES: Array[String] = [
	"rat", "orc", "poring", "drops", "poporing", "lunatic", "fabre", "pupa", "chonchon", "willow", "rocker",
	"spore", "thief_bug", "condor", "picky", "peco_peco", "wolf", "hode", "orc_warrior", "zombie", "skeleton",
	"baphomet_jr",
]
## Index = engine item id (0 is "none" and has no art).
const ITEM_NAMES: Array[String] = [
	"none", "gold_coin", "meat", "sword", "potion_red", "potion_blue", "dagger", "bow", "staff", "mace", "axe",
	"shield", "armor", "hat", "boots", "garment", "ring", "card", "ore", "herb", "fish",
]
const UI_NAMES: Array[String] = [
	"panel", "panel_title", "button", "button_pressed", "button_hover", "slot", "hpbar_bg", "hpbar_fill",
	"spbar_fill", "expbar_fill", "cursor", "icon_close",
]
## Round 2 animated sheets (docs/round2_contract.md 1.4): `anim.<kind>.<action>` -> act json + action png under
## `res://assets/anim/<kind>/`. Index = engine anim action id (`world["anim"]["action"]`).
const ANIM_ACTIONS: Array[String] = [
	"idle", "walk", "sit", "pickup", "standby", "attack", "hurt", "dead", "cast",
]
const ANIM_MONSTER_ACTIONS: Array[String] = ["idle", "walk", "attack", "hurt", "dead"]
## Monster kinds in engine order (`game/data/monsters.py`), unlike the legacy MONSTER_NAMES table.
const ANIM_MONSTER_KINDS: Array[String] = [
	"poring", "orc_warrior", "fabre", "lunatic", "drops", "poporing", "willow", "rocker", "spore", "thief_bug",
	"zombie", "wolf", "skeleton", "peco_peco", "hode", "soldier_skeleton", "bigfoot", "mummy",
	"golden_thief_bug", "eddga",
]
## Player-only actions a monster sheet lacks, and the monster action drawn instead.
const ANIM_MONSTER_FALLBACK: Dictionary = {"sit": "idle", "pickup": "idle", "standby": "idle", "cast": "attack"}
const FALLBACK_PX: int = 16
const TILE_FALLBACK_COLORS: Array[Color] = [
	Color(0.30, 0.55, 0.25),
	Color(0.62, 0.52, 0.36),
	Color(0.55, 0.50, 0.45),
	Color(0.80, 0.78, 0.70),
	Color(0.20, 0.35, 0.75),
	Color(0.10, 0.30, 0.12),
	Color(0.35, 0.30, 0.30),
	Color(0.45, 0.45, 0.50),
	Color(0.70, 0.30, 0.90),
	Color(0.95, 0.60, 0.70),
	Color(0.55, 0.40, 0.20),
]
const FALLBACK_SATURATION: float = 0.6
const FALLBACK_VALUE: float = 0.85
const HUE_STEPS: int = 360

var errors: PackedStringArray = []
## key -> {"path": String, "region": Rect2i, "has_region": bool, "sheet": bool, "anim": String, "png": String}
var _slots: Dictionary = {}
var _cache: Dictionary = {}


static func tile_key(terrain_id: int) -> String:
	return "tile.%d" % terrain_id


static func job_key(job: int, anim: String) -> String:
	return "job.%d.%s" % [job, anim]


static func monster_key(mkind: int) -> String:
	if mkind < 0 or mkind >= MONSTER_NAMES.size():
		return "monster.unknown"
	return "monster.%s" % MONSTER_NAMES[mkind]


static func item_key(item_id: int) -> String:
	if item_id < 0 or item_id >= ITEM_NAMES.size():
		return "item.unknown"
	return "item.%s" % ITEM_NAMES[item_id]


## SpriteFrames animation name: `<anim>_<dir>`; `dead` only exists facing down. Unknown dirs fall back to down.
static func anim_name(anim: String, dir: int) -> String:
	if anim == "dead":
		return "dead_down"
	var d: String = DIRS[dir] if dir >= 0 and dir < DIRS.size() else DIRS[0]
	return "%s_%s" % [anim, d]


static func anim_key(kind: String, action: String) -> String:
	return "anim.%s.%s" % [kind, action]


static func anim_action_name(action_id: int) -> String:
	if action_id < 0 or action_id >= ANIM_ACTIONS.size():
		return ANIM_ACTIONS[0]
	return ANIM_ACTIONS[action_id]


## Slot key for a job (0..6, JOB_NAMES order) and an engine anim action id.
static func job_anim_key(job: int, action_id: int) -> String:
	var j: int = job if job >= 0 and job < JOB_NAMES.size() else 0
	return anim_key(JOB_NAMES[j], anim_action_name(action_id))


## Slot key for an engine monster kind and an engine anim action id; player-only actions map to a monster one.
static func monster_anim_key(mkind: int, action_id: int) -> String:
	if mkind < 0 or mkind >= ANIM_MONSTER_KINDS.size():
		return "anim.unknown.idle"
	var action: String = anim_action_name(action_id)
	if ANIM_MONSTER_FALLBACK.has(action):
		action = ANIM_MONSTER_FALLBACK[action]
	return anim_key(ANIM_MONSTER_KINDS[mkind], action)


static func expected_anim_keys() -> PackedStringArray:
	var out: PackedStringArray = []
	for j: String in JOB_NAMES:
		for a: String in ANIM_ACTIONS:
			out.append(anim_key(j, a))
	for m: String in ANIM_MONSTER_KINDS:
		for a: String in ANIM_MONSTER_ACTIONS:
			out.append(anim_key(m, a))
	return out


static func expected_keys() -> PackedStringArray:
	var out: PackedStringArray = []
	for t: int in range(TERRAIN_NAMES.size()):
		out.append(tile_key(t))
	for j: int in range(JOB_NAMES.size()):
		for a: String in ANIMS:
			out.append(job_key(j, a))
	for m: int in range(MONSTER_NAMES.size()):
		out.append(monster_key(m))
	for i: int in range(1, ITEM_NAMES.size()):
		out.append(item_key(i))
	for u: String in UI_NAMES:
		out.append("ui.%s" % u)
	return out


## Deterministic colour: tiles use a fixed table, everything else a hue from the hash of the key.
static func fallback_color(key: String) -> Color:
	if key.begins_with("tile."):
		var suffix: String = key.substr(5)
		if suffix.is_valid_int():
			var id: int = suffix.to_int()
			if id >= 0 and id < TILE_FALLBACK_COLORS.size():
				return TILE_FALLBACK_COLORS[id]
	var hue: float = float(key.hash() % HUE_STEPS) / float(HUE_STEPS)
	return Color.from_hsv(hue, FALLBACK_SATURATION, FALLBACK_VALUE)


static func load_default() -> AssetSlots:
	var reg := AssetSlots.new()
	if not FileAccess.file_exists(SLOTS_PATH):
		return reg
	reg.parse(FileAccess.get_file_as_string(SLOTS_PATH))
	return reg


## Parses the slots json. Returns false when the document itself is unusable; bad entries are skipped and listed
## in `errors`. Replaces any earlier content.
func parse(text: String) -> bool:
	_slots.clear()
	_cache.clear()
	errors = []
	var json := JSON.new()
	if json.parse(text) != OK:
		errors.append("slots json: %s (line %d)" % [json.get_error_message(), json.get_error_line()])
		return false
	var data: Variant = json.data
	if not (data is Dictionary):
		errors.append("slots json is not an object")
		return false
	var root: Dictionary = data
	var raw: Variant = root.get("slots")
	if not (raw is Dictionary):
		errors.append("missing 'slots' object")
		return false
	var slots: Dictionary = raw
	for k: Variant in slots.keys():
		var key: String = str(k)
		var entry: Variant = slots[k]
		if not (entry is Dictionary):
			errors.append("%s: entry is not an object" % key)
			continue
		var parsed: Dictionary = _parse_entry(key, entry)
		if not parsed.is_empty():
			_slots[key] = parsed
	return true


func _parse_entry(key: String, entry: Dictionary) -> Dictionary:
	if entry.has("sheet"):
		var sheet: Variant = entry["sheet"]
		var anim: Variant = entry.get("anim")
		if not (sheet is String) or String(sheet).is_empty():
			errors.append("%s: sheet must be a non-empty string" % key)
			return {}
		if not (anim is String) or String(anim).is_empty():
			errors.append("%s: sheet slot needs an anim" % key)
			return {}
		var png: Variant = entry.get("path", "")
		return {
			"path": String(sheet), "region": Rect2i(), "has_region": false, "sheet": true, "anim": String(anim),
			"png": String(png) if png is String else "",
		}
	var path: Variant = entry.get("path")
	if not (path is String) or String(path).is_empty():
		errors.append("%s: needs a path or a sheet" % key)
		return {}
	var region := Rect2i()
	var has_region: bool = false
	if entry.has("region"):
		var r: Variant = entry["region"]
		if not (r is Array) or (r as Array).size() != 4:
			errors.append("%s: region must be [x, y, w, h]" % key)
			return {}
		var nums: Array = r
		var vals: Array[int] = []
		for i: int in range(4):
			var n: Variant = nums[i]
			if not (n is float or n is int) or float(n) != floorf(float(n)):
				errors.append("%s: region values must be integers" % key)
				return {}
			vals.append(int(n))
		if vals[2] <= 0 or vals[3] <= 0:
			errors.append("%s: region size must be positive" % key)
			return {}
		region = Rect2i(vals[0], vals[1], vals[2], vals[3])
		has_region = true
	return {"path": String(path), "region": region, "has_region": has_region, "sheet": false, "anim": "", "png": ""}


func has(key: String) -> bool:
	return _slots.has(key)


func keys() -> PackedStringArray:
	var out: PackedStringArray = []
	for k: Variant in _slots.keys():
		out.append(str(k))
	return out


func path_of(key: String) -> String:
	if not _slots.has(key):
		return ""
	var s: Dictionary = _slots[key]
	return s["path"]


func region_of(key: String) -> Rect2i:
	if not _slots.has(key):
		return Rect2i()
	var s: Dictionary = _slots[key]
	return s["region"]


func anim_of(key: String) -> String:
	if not _slots.has(key):
		return ""
	var s: Dictionary = _slots[key]
	return s["anim"]


## For an anim sheet slot: the action png (rows = directions 0..4, columns = frames). Empty when not given.
func png_of(key: String) -> String:
	if not _slots.has(key):
		return ""
	var s: Dictionary = _slots[key]
	return s["png"]


func is_sheet(key: String) -> bool:
	if not _slots.has(key):
		return false
	var s: Dictionary = _slots[key]
	return s["sheet"]


## Cached texture for a slot. Never null: unknown slot, sheet slot, missing file or failed load give the fallback.
func texture(key: String) -> Texture2D:
	if _cache.has(key):
		var cached: Texture2D = _cache[key]
		return cached
	var tex: Texture2D = _build_texture(key)
	_cache[key] = tex
	return tex


func _build_texture(key: String) -> Texture2D:
	if not _slots.has(key) or is_sheet(key):
		return fallback_texture(key)
	var path: String = path_of(key)
	if not FileAccess.file_exists(path) and not ResourceLoader.exists(path):
		return fallback_texture(key)
	var loaded: Texture2D = load(path) as Texture2D
	if loaded == null:
		return fallback_texture(key)
	var s: Dictionary = _slots[key]
	if bool(s["has_region"]):
		var atlas := AtlasTexture.new()
		atlas.atlas = loaded
		var rect: Rect2i = s["region"]
		atlas.region = Rect2(rect)
		return atlas
	return loaded


func fallback_texture(key: String) -> ImageTexture:
	var img: Image = Image.create(FALLBACK_PX, FALLBACK_PX, false, Image.FORMAT_RGBA8)
	img.fill(fallback_color(key))
	return ImageTexture.create_from_image(img)


## Expected keys whose slot is absent or whose file does not exist.
func missing_keys() -> PackedStringArray:
	var out: PackedStringArray = []
	for k: String in expected_keys():
		var p: String = path_of(k)
		if p.is_empty() or not FileAccess.file_exists(p):
			out.append(k)
	return out


## Expected anim keys whose slot is absent or whose act json or png does not exist.
func missing_anim_keys() -> PackedStringArray:
	var out: PackedStringArray = []
	for k: String in expected_anim_keys():
		var act: String = path_of(k)
		var png: String = png_of(k)
		if act.is_empty() or png.is_empty() or not FileAccess.file_exists(act) or not FileAccess.file_exists(png):
			out.append(k)
	return out
