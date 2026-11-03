class_name ZoneBanner
extends Control
## RO zone name on entry: a big outlined title near the top of the screen that fades in, holds and fades out
## whenever the zone under the own player changes. Zones come from `welcome.world.zones` (team ro-world2,
## [{"name", "box": [x0, y0, x1, y1], "kind"}]) when present; until then from the dungeon boxes, the protection
## zone (towns) and the nearest town (fields). Feed `observe(world, map, tile, town)` every frame.

const FADE_IN_S: float = 0.3
const HOLD_S: float = 2.5
const FADE_OUT_S: float = 0.9
const KIND_TOWN: String = "town"
const KIND_FIELD: String = "field"
const KIND_DUNGEON: String = "dungeon"
const CAVE_TERRAIN: int = 11
const KIND_COLORS: Dictionary = {
	"town": Color(1.0, 0.92, 0.55),
	"field": Color(0.75, 1.0, 0.70),
	"dungeon": Color(1.0, 0.60, 0.55),
}

var zone_key: String = ""
var zone: Dictionary = {}
var shown: int = 0  ## Times the banner was (re)started.
var age_s: float = -1.0  ## Seconds since the banner started; < 0 when idle.
var title: Label
var subtitle: Label


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	var box := VBoxContainer.new()
	box.set_anchors_preset(Control.PRESET_FULL_RECT)
	box.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(box)
	title = Label.new()
	title.theme_type_variation = &"ZoneTitle"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	box.add_child(title)
	subtitle = Label.new()
	subtitle.theme_type_variation = &"ZoneSubtitle"
	subtitle.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	box.add_child(subtitle)
	modulate.a = 0.0


## Recomputes the zone at `tile`; restarts the banner when its key changed. Returns true on a change.
func observe(world: Dictionary, map: MapData, tile: Vector2i, town: int) -> bool:
	var z: Dictionary = zone_at(world, map, tile, town)
	if String(z["key"]) == zone_key:
		return false
	zone_key = String(z["key"])
	zone = z
	show_zone(String(z["name"]), String(z["kind"]))
	return true


func show_zone(zone_name: String, kind: String) -> void:
	shown += 1
	age_s = 0.0
	if title != null:
		title.text = zone_name
		title.add_theme_color_override("font_color", KIND_COLORS.get(kind, Color.WHITE))
		subtitle.text = kind.capitalize()


func _process(delta: float) -> void:
	if age_s < 0.0:
		return
	age_s += delta
	modulate.a = alpha_at(age_s)
	if age_s > FADE_IN_S + HOLD_S + FADE_OUT_S:
		age_s = -1.0


# ---- pure helpers ----

## Banner opacity `t` seconds after it started.
static func alpha_at(t: float) -> float:
	if t < 0.0:
		return 0.0
	if t < FADE_IN_S:
		return t / FADE_IN_S
	if t < FADE_IN_S + HOLD_S:
		return 1.0
	return clampf(1.0 - (t - FADE_IN_S - HOLD_S) / FADE_OUT_S, 0.0, 1.0)


static func box_has(box: Variant, tile: Vector2i) -> bool:
	if typeof(box) != TYPE_ARRAY or (box as Array).size() < 4:
		return false
	var b: Array = box
	return tile.x >= int(b[0]) and tile.y >= int(b[1]) and tile.x <= int(b[2]) and tile.y <= int(b[3])


## The zone at `tile`: {key, name, kind}. `world` = welcome.world ({} when absent), `town` = you.town (home town,
## used only when nothing else names the place).
static func zone_at(world: Dictionary, map: MapData, tile: Vector2i, town: int) -> Dictionary:
	var zones: Variant = world.get("zones")
	if typeof(zones) == TYPE_ARRAY:
		for z: Variant in zones as Array:
			if typeof(z) == TYPE_DICTIONARY and box_has((z as Dictionary).get("box"), tile):
				var zd: Dictionary = z
				var zn: String = String(zd.get("name", "?"))
				return {"key": "zone:" + zn, "name": String(zd.get("title", zn)), "kind": String(zd.get("kind", KIND_FIELD))}
	var dungeons: Variant = world.get("dungeons")
	if typeof(dungeons) == TYPE_ARRAY:
		for d: Variant in dungeons as Array:
			if typeof(d) == TYPE_DICTIONARY and box_has((d as Dictionary).get("box"), tile):
				var dn: String = String((d as Dictionary).get("name", "Dungeon"))
				return {"key": "dungeon:" + dn, "name": dn, "kind": KIND_DUNGEON}
	var in_town: bool = map != null and map.pz_at(tile.x, tile.y)
	var near: String = nearest_town(world, tile, town)
	if map != null and map.at(tile.x, tile.y) == CAVE_TERRAIN:
		return {"key": "dungeon:?", "name": "Dungeon", "kind": KIND_DUNGEON}
	if in_town:
		var tn: String = near if near != "" else "Town"
		return {"key": "town:" + tn, "name": tn, "kind": KIND_TOWN}
	var fn: String = (near + " Field") if near != "" else "Field"
	return {"key": "field:" + fn, "name": fn, "kind": KIND_FIELD}


## Name of the town whose centre is nearest to `tile` ("" without towns); falls back to towns[town] by index.
static func nearest_town(world: Dictionary, tile: Vector2i, town: int) -> String:
	var towns: Variant = world.get("towns")
	if typeof(towns) != TYPE_ARRAY or (towns as Array).is_empty():
		return ""
	var best: String = ""
	var best_d: float = INF
	for t: Variant in towns as Array:
		if typeof(t) != TYPE_DICTIONARY:
			continue
		var c: Vector2 = ViewMath.to_vec((t as Dictionary).get("center"))
		if c == Vector2.INF:
			continue
		var d: float = c.distance_squared_to(Vector2(tile))
		if d < best_d:
			best_d = d
			best = String((t as Dictionary).get("name", ""))
	if best == "" and town >= 0 and town < (towns as Array).size():
		best = String(((towns as Array)[town] as Dictionary).get("name", ""))
	return best
