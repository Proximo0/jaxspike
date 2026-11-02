class_name ViewMath
extends RefCounted
## Pure maths and tables of the thin client view: tuning fallbacks, screen/tile conversion, colours per kind.

const DEFAULT_TILE_PX: float = 32.0
const DEFAULT_INTERP_DELAY_TICKS: float = 1.0
const DEFAULT_CAMERA_ZOOM: float = 1.0
const DEFAULT_NAME_LABEL_PX: float = 12.0
const DEFAULT_TICK_MS: float = 120.0

const CAT_PLAYER: int = 0
const CAT_MONSTER: int = 1
const CAT_DROP: int = 2
const DROP_KIND_BASE: int = 100

## Flat colour per terrain id 0..7 (grass, road, floor, temple, water, tree, wall, rock).
const TERRAIN_COLORS: Array[Color] = [
	Color(0.30, 0.55, 0.25),
	Color(0.62, 0.52, 0.36),
	Color(0.55, 0.50, 0.45),
	Color(0.80, 0.78, 0.70),
	Color(0.20, 0.35, 0.75),
	Color(0.10, 0.30, 0.12),
	Color(0.35, 0.30, 0.30),
	Color(0.45, 0.45, 0.50),
]
const MONSTER_COLORS: Array[Color] = [
	Color(0.85, 0.20, 0.20),
	Color(0.90, 0.40, 0.15),
	Color(0.70, 0.10, 0.35),
	Color(0.95, 0.30, 0.45),
	Color(0.60, 0.15, 0.15),
	Color(0.90, 0.55, 0.30),
]
const PLAYER_COLOR: Color = Color(0.25, 0.45, 0.95)
const OWN_COLOR: Color = Color(1.0, 1.0, 1.0)
const DROP_COLOR: Color = Color(0.95, 0.85, 0.20)


## A live-tunable number from `Tuning.shared()`, or `fallback` while the row does not exist yet.
static func tuned(field: String, fallback: float) -> float:
	var v: Variant = Tuning.shared().get(field)
	if v == null:
		return fallback
	return float(v)


static func terrain_color(terrain_id: int) -> Color:
	if terrain_id < 0 or terrain_id >= TERRAIN_COLORS.size():
		return Color(1.0, 0.0, 1.0)
	return TERRAIN_COLORS[terrain_id]


## 0 player, 1 monster, 2 drop (wire kinds: players 0, monsters 1 + mkind, drops 100 + item id).
static func category(kind: int) -> int:
	if kind == 0:
		return CAT_PLAYER
	if kind >= DROP_KIND_BASE:
		return CAT_DROP
	return CAT_MONSTER


static func kind_color(kind: int, is_own: bool) -> Color:
	match category(kind):
		CAT_PLAYER:
			return OWN_COLOR if is_own else PLAYER_COLOR
		CAT_DROP:
			return DROP_COLOR
	return MONSTER_COLORS[posmod(kind - 1, MONSTER_COLORS.size())]


## Quad edge as a fraction of the tile for each category.
static func kind_scale(kind: int) -> float:
	match category(kind):
		CAT_DROP:
			return 0.5
		CAT_PLAYER:
			return 0.8
	return 0.85


static func has_hp_bar(kind: int) -> bool:
	return category(kind) != CAT_DROP


## Tile pixel centre of tile (x, y).
static func tile_center(tile: Vector2, tile_px: float) -> Vector2:
	return (tile + Vector2(0.5, 0.5)) * tile_px


## World pixel position -> tile coordinates (floor).
static func world_to_tile(world_px: Vector2, tile_px: float) -> Vector2i:
	if tile_px <= 0.0:
		return Vector2i.ZERO
	return Vector2i(floori(world_px.x / tile_px), floori(world_px.y / tile_px))


## Screen (viewport) pixel -> world pixel for a Camera2D centred on `camera_center` with `zoom` (>1 zooms in).
static func screen_to_world(screen_px: Vector2, camera_center: Vector2, zoom: float, viewport_size: Vector2) -> Vector2:
	var z: float = maxf(zoom, 0.0001)
	return camera_center + (screen_px - viewport_size * 0.5) / z


static func screen_to_tile(screen_px: Vector2, camera_center: Vector2, zoom: float, viewport_size: Vector2,
		tile_px: float) -> Vector2i:
	return world_to_tile(screen_to_world(screen_px, camera_center, zoom, viewport_size), tile_px)


static func in_map(tile: Vector2i, map_size: int) -> bool:
	return tile.x >= 0 and tile.y >= 0 and tile.x < map_size and tile.y < map_size


## A wire position (Array [x, y] or Vector2i) -> Vector2. Zero for anything else.
static func to_vec(v: Variant) -> Vector2:
	if v is Vector2i:
		var vi: Vector2i = v
		return Vector2(vi)
	if v is Vector2:
		return v
	if v is Array:
		var a: Array = v
		if a.size() >= 2:
			return Vector2(float(a[0]), float(a[1]))
	return Vector2.ZERO


static func hp_fraction(hp: int, max_hp: int) -> float:
	if max_hp <= 0:
		return 0.0
	return clampf(float(hp) / float(max_hp), 0.0, 1.0)


## Terrain ids a character cannot walk on: water, tree, wall, rock.
const BLOCKED_TERRAIN: Array[int] = [4, 5, 6, 7]
const MARKER_STALL_TICKS: int = 10
const MARKER_FLASH_MS: float = 400.0
const MARKER_COLOR: Color = Color(1.0, 1.0, 1.0, 0.9)
const MARKER_BLOCKED_COLOR: Color = Color(1.0, 0.15, 0.15, 0.95)


static func is_blocked_terrain(terrain_id: int) -> bool:
	return BLOCKED_TERRAIN.has(terrain_id)


## True when the click target `tile` is a blocked tile of `map` (unloaded map and outside tiles are not blocked here).
static func is_blocked_tile(map: MapData, tile: Vector2i) -> bool:
	return map != null and is_blocked_terrain(map.at(tile.x, tile.y))


## The walk target marker is finished when the own player stands on it or `stalled_ticks` server ticks passed
## without the own position changing.
static func marker_done(target: Vector2i, own_pos: Vector2i, stalled_ticks: int,
		max_stalled: int = MARKER_STALL_TICKS) -> bool:
	return own_pos == target or stalled_ticks >= max_stalled
