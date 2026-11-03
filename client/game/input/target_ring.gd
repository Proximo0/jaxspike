class_name TargetRing
extends Node2D
## Click target highlight: a pulsing ellipse under the selected entity (red on monsters, green on players, gold on
## drops and nodes), following its drawn position. `bind(layer, targeting)` with the EntityLayer and the skill
## bar's TargetSelect; hidden while nothing is selected or the entity is not drawn.

const COLORS: Array[Color] = [Color(0.40, 1.0, 0.45), Color(1.0, 0.30, 0.25), Color(1.0, 0.85, 0.30), Color(1.0, 0.85, 0.30)]

var layer: EntityLayer = null
var targeting: TargetSelect = null
var target_id: int = TargetSelect.NO_TARGET
var cat: int = ViewMath.CAT_MONSTER
var _t: float = 0.0


func bind(entity_layer: EntityLayer, t: TargetSelect) -> void:
	layer = entity_layer
	targeting = t
	z_index = 4


func _process(delta: float) -> void:
	_t += delta
	target_id = targeting.selected_id if targeting != null else TargetSelect.NO_TARGET
	var p: Vector2 = Vector2.INF
	if layer != null and target_id != TargetSelect.NO_TARGET:
		p = layer.world_pos_of(target_id)
	visible = p != Vector2.INF
	if not visible:
		return
	position = p
	if layer.store != null:
		var e: Dictionary = layer.store.get_ent(target_id)
		cat = ViewMath.entity_category(int(e.get("kind", 1)), e.has("item"))
	queue_redraw()


## Ring radii (x, y) in pixels for a tile size and pulse phase `t` seconds.
static func radii(tile_px: float, t: float) -> Vector2:
	var k: float = 0.48 + 0.05 * sin(t * TAU * 1.2)
	return Vector2(tile_px * k, tile_px * k * 0.5)


func _draw() -> void:
	var tile_px: float = ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)
	var r: Vector2 = radii(tile_px, _t)
	var feet := Vector2(0.0, tile_px * 0.32)
	var pts := PackedVector2Array()
	for i: int in range(33):
		var a: float = TAU * float(i) / 32.0
		pts.append(feet + Vector2(cos(a) * r.x, sin(a) * r.y))
	var col: Color = COLORS[clampi(cat, 0, COLORS.size() - 1)]
	draw_polyline(pts, Color(0, 0, 0, 0.6), 4.0)
	draw_polyline(pts, col, 2.0)
