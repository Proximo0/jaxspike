class_name EntitySprite
extends Node2D
## One pooled entity: coloured quad, optional HP bar and name label, all drawn in _draw (no child nodes).
## The node origin is the tile centre. Call `configure` when something other than position changed.

const BAR_H: float = 4.0
const BAR_BG: Color = Color(0.25, 0.05, 0.05, 0.9)
const BAR_FG: Color = Color(0.25, 0.85, 0.35)
const LABEL_COLOR: Color = Color(1.0, 1.0, 1.0)
const LABEL_SHADOW: Color = Color(0.0, 0.0, 0.0, 0.8)

var kind: int = 0
var is_own: bool = false
var label: String = ""
var hp_frac: float = 1.0
var tile_px: float = 32.0
var label_px: int = 12


## Returns true when a redraw was queued.
func configure(new_kind: int, own: bool, new_label: String, new_hp_frac: float, new_tile_px: float,
		new_label_px: int) -> bool:
	if new_kind == kind and own == is_own and new_label == label and is_equal_approx(new_hp_frac, hp_frac) \
			and is_equal_approx(new_tile_px, tile_px) and new_label_px == label_px:
		return false
	kind = new_kind
	is_own = own
	label = new_label
	hp_frac = new_hp_frac
	tile_px = new_tile_px
	label_px = new_label_px
	queue_redraw()
	return true


func quad_color() -> Color:
	return ViewMath.kind_color(kind, is_own)


func _draw() -> void:
	var edge: float = tile_px * ViewMath.kind_scale(kind)
	var half: float = edge * 0.5
	draw_rect(Rect2(-half, -half, edge, edge), quad_color())
	var top: float = -half
	if ViewMath.has_hp_bar(kind):
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
