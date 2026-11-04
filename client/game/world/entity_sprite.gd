class_name EntitySprite
extends Node2D
## One pooled entity: coloured quad, optional HP bar and name label, all drawn in _draw (no child nodes).
## The node origin is the tile centre. Call `configure` when something other than position changed.
## Players and monsters are animated once `set_clip` is called (round2 contract 1.3): one ActSheet frame (art-anim's
## sheet or a baked PlaceholderClips sheet), feet on the tile, mirrored for directions 5..7; drops and nodes keep
## the quad / static texture.

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

const FEET_Y: float = 0.28  ## feet sit this many tiles below the tile centre
const HEAD_PX: float = 40.0  ## animated figure height at a 32 px tile (bar and label go above it)

var animated: bool = false  ## set_clip was called: draw the clip instead of the quad
var look: int = 0  ## job (players) or monster art index
var sheet: ActSheet = null
var clip_name: String = "idle"
var clip_anim: StringName = &""
var frame: int = 0
var frames: int = 1
var face: int = 0
var flip: bool = false
var flash: float = 0.0  ## hurt tint


## Sets the clip frame to draw. Returns true when a redraw was queued (only when something visible changed).
func set_clip(new_sheet: ActSheet, new_look: int, new_clip: String, new_anim: StringName, new_frame: int,
		new_frames: int, new_face: int, new_flip: bool, new_flash: float) -> bool:
	if animated and new_sheet == sheet and new_look == look and new_clip == clip_name and new_anim == clip_anim 			and new_frame == frame and new_frames == frames and new_face == face and new_flip == flip 			and is_equal_approx(new_flash, flash):
		return false
	animated = true
	sheet = new_sheet
	look = new_look
	clip_name = new_clip
	clip_anim = new_anim
	frame = new_frame
	frames = new_frames
	face = new_face
	flip = new_flip
	flash = new_flash
	queue_redraw()
	return true


## Back to the static look (quad or texture): drops, nodes, or an entity that lost its clip.
func clear_clip() -> void:
	if animated:
		animated = false
		sheet = null
		queue_redraw()


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
	var head_top: float = -half
	var clip_drawn: bool = animated and _draw_clip()
	if clip_drawn:
		head_top = tile_px * FEET_Y - _clip_height()
	elif tex != null:
		var tint: Color = ViewMath.NODE_DEPLETED_COLOR if (cat == ViewMath.CAT_NODE and depleted) else Color.WHITE
		draw_texture_rect(tex, Rect2(-half, -half, edge, edge), false, tint)
		if is_own:
			draw_rect(Rect2(-half, -half, edge, edge), OWN_RING, false, 1.0)
	elif cat == ViewMath.CAT_NODE:
		draw_colored_polygon(node_points(ViewMath.node_shape(kind - ViewMath.NODE_KIND_BASE), half), quad_color())
	else:
		draw_rect(Rect2(-half, -half, edge, edge), quad_color())
	var top: float = head_top
	if ViewMath.cat_has_hp_bar(cat):
		var w: float = tile_px * 0.9
		var bar_y: float = minf(-tile_px * 0.5, head_top) - BAR_H - 1.0
		draw_rect(Rect2(-w * 0.5, bar_y, w, BAR_H), BAR_BG)
		draw_rect(Rect2(-w * 0.5, bar_y, w * hp_frac, BAR_H), BAR_FG)
		top = bar_y
	if label != "":
		var font: Font = ThemeDB.fallback_font
		var sz: Vector2 = font.get_string_size(label, HORIZONTAL_ALIGNMENT_LEFT, -1, label_px)
		var pos := Vector2(-sz.x * 0.5, top - 3.0)
		draw_string(font, pos + Vector2(1, 1), label, HORIZONTAL_ALIGNMENT_LEFT, -1, label_px, LABEL_SHADOW)
		draw_string(font, pos, label, HORIZONTAL_ALIGNMENT_LEFT, -1, label_px, LABEL_COLOR)


## Draws the clip frame with the feet at FEET_Y; false when the sheet has no such frame (the quad is drawn).
func _draw_clip() -> bool:
	if sheet == null:
		return false
	var tex_f: Texture2D = sheet.frame_texture(clip_anim, frame)
	if tex_f == null:
		return false
	var s: float = tile_px / 32.0 * sheet.scale  # 2x sheets carry scale 0.5
	var feet := Vector2(0.0, tile_px * FEET_Y)
	if is_own:
		draw_arc(feet, 11.0 * s, 0.0, TAU, 20, OWN_RING, 1.5)
	var size := Vector2(sheet.frame_w, sheet.frame_h) * s
	var anc: Vector2 = sheet.anchor * s
	var tint: Color = Color.WHITE.lerp(Color(1.0, 0.6, 0.6), flash * 0.4)  # sheets bake their own flinch flash
	draw_set_transform(feet, 0.0, Vector2(-1.0 if flip else 1.0, 1.0))
	draw_texture_rect(tex_f, Rect2(-anc, size), false, tint)
	draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)
	return true


## Height (px) above the feet that the bar and label clear.
func _clip_height() -> float:
	if sheet != null and sheet.scale > 0.0 and sheet.anchor.y > 0.0:
		return sheet.anchor.y * sheet.scale * tile_px / 32.0
	return HEAD_PX * tile_px / 32.0
