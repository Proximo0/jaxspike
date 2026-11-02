class_name EntitySprite
extends Node2D
## One pooled entity: coloured quad, optional HP bar and name label, all drawn in _draw (no child nodes).
## The node origin is the tile centre. Call `configure` when something other than position changed.

const BAR_H: float = 4.0
const BAR_BG: Color = Color(0.25, 0.05, 0.05, 0.9)
const BAR_FG: Color = Color(0.25, 0.85, 0.35)
const LABEL_COLOR: Color = Color(1.0, 1.0, 1.0)
const OWN_RING: Color = Color(1.0, 1.0, 1.0, 0.9)
const LABEL_SHADOW: Color = Color(0.0, 0.0, 0.0, 0.8)

var kind: int = 0
var is_own: bool = false
var label: String = ""
var hp_frac: float = 1.0
var tile_px: float = 32.0
var label_px: int = 12
var tex: Texture2D = null  ## Art for this entity when its slot resolved; null draws the coloured quad.
var depleted: bool = false  ## Nodes with hp 0 are drawn grey.
var cat: int = ViewMath.CAT_MONSTER


## Returns true when a redraw was queued.
## `new_cat` -1 derives the category from the kind; the layer passes ViewMath.entity_category for drops and nodes.
func configure(new_kind: int, own: bool, new_label: String, new_hp_frac: float, new_tile_px: float,
		new_label_px: int, new_tex: Texture2D = null, new_depleted: bool = false, new_cat: int = -1) -> bool:
	var want_cat: int = new_cat if new_cat >= 0 else ViewMath.category(new_kind)
	if new_kind == kind and own == is_own and new_label == label and is_equal_approx(new_hp_frac, hp_frac) \
			and is_equal_approx(new_tile_px, tile_px) and new_label_px == label_px and new_tex == tex \
			and new_depleted == depleted and want_cat == cat:
		return false
	tex = new_tex
	depleted = new_depleted
	cat = want_cat
	texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	kind = new_kind
	is_own = own
	label = new_label
	hp_frac = new_hp_frac
	tile_px = new_tile_px
	label_px = new_label_px
	queue_redraw()
	return true


func quad_color() -> Color:
	if cat == ViewMath.CAT_NODE and depleted:
		return ViewMath.NODE_DEPLETED_COLOR
	return ViewMath.cat_color(cat, kind, is_own)


## Polygon points of a node shape (0 diamond, 1 circle, 2 triangle, 3 square) in a box of half-size `half`.
static func node_points(shape: int, half: float) -> PackedVector2Array:
	match shape:
		0:
			return PackedVector2Array([Vector2(0, -half), Vector2(half, 0), Vector2(0, half), Vector2(-half, 0)])
		1:
			var pts := PackedVector2Array()
			for i: int in range(12):
				var a: float = TAU * float(i) / 12.0
				pts.append(Vector2(cos(a), sin(a)) * half)
			return pts
		2:
			return PackedVector2Array([Vector2(0, -half), Vector2(half, half), Vector2(-half, half)])
	return PackedVector2Array([Vector2(-half, -half), Vector2(half, -half), Vector2(half, half), Vector2(-half, half)])


func _draw() -> void:
	var edge: float = tile_px * ViewMath.cat_scale(cat)
	var half: float = edge * 0.5
	if tex != null:
		var tint: Color = ViewMath.NODE_DEPLETED_COLOR if (cat == ViewMath.CAT_NODE and depleted) else Color.WHITE
		draw_texture_rect(tex, Rect2(-half, -half, edge, edge), false, tint)
		if is_own:
			draw_rect(Rect2(-half, -half, edge, edge), OWN_RING, false, 1.0)
	elif cat == ViewMath.CAT_NODE:
		draw_colored_polygon(node_points(ViewMath.node_shape(kind - ViewMath.NODE_KIND_BASE), half), quad_color())
	else:
		draw_rect(Rect2(-half, -half, edge, edge), quad_color())
	var top: float = -half
	if ViewMath.cat_has_hp_bar(cat):
		var w: float = tile_px * 0.9
		var bar_y: float = -tile_px * 0.5 - BAR_H - 1.0
		draw_rect(Rect2(-w * 0.5, bar_y, w, BAR_H), BAR_BG)
		draw_rect(Rect2(-w * 0.5, bar_y, w * hp_frac, BAR_H), BAR_FG)
		top = bar_y
	if label != "":
		var font: Font = ThemeDB.fallback_font
		var sz: Vector2 = font.get_string_size(label, HORIZONTAL_ALIGNMENT_LEFT, -1, label_px)
		var pos := Vector2(-sz.x * 0.5, top - 3.0)
		draw_string(font, pos + Vector2(1, 1), label, HORIZONTAL_ALIGNMENT_LEFT, -1, label_px, LABEL_SHADOW)
		draw_string(font, pos, label, HORIZONTAL_ALIGNMENT_LEFT, -1, label_px, LABEL_COLOR)
