class_name DamageLayer
extends Node2D
## Draws the floating damage numbers EntityLayer hands it each frame (RO style: big outlined digits rising from
## the target's head). Thin wrapper: the timing lives in DamagePops.

const FONT_PX: float = 20.0
const OUTLINE_PX: int = 5
const COLOR_OTHER: Color = Color(1.0, 1.0, 1.0)
const COLOR_OWN: Color = Color(1.0, 0.35, 0.30)  ## damage taken by the own character
const COLOR_OUTLINE: Color = Color(0.08, 0.04, 0.06)

var items: Array[Dictionary] = []  ## {pos (px), dmg, alpha, own, crit, seq, text (optional override)}
var scale_px: float = 1.0


func show_items(new_items: Array[Dictionary], new_scale: float) -> void:
	if new_items.is_empty() and items.is_empty():
		return
	items = new_items
	scale_px = new_scale
	queue_redraw()


func _draw() -> void:
	var font: Font = ThemeDB.fallback_font
	for it: Dictionary in items:
		var dmg: int = int(it["dmg"])
		var kind: int = DamageStyle.kind_of(dmg, bool(it.get("own", false)), bool(it.get("crit", false)))
		var size: int = roundi(DamageStyle.font_px(kind) * scale_px)
		var text: String = String(it.get("text", DamageStyle.text_of(dmg, kind)))
		var w: float = font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, size).x
		var p: Vector2 = (it["pos"] as Vector2) - Vector2(w * 0.5 - DamageStyle.drift_px(int(it.get("seq", 0)), kind) * scale_px, 0.0)
		var a: float = float(it["alpha"])
		var col: Color = DamageStyle.color_of(kind)
		draw_string_outline(font, p, text, HORIZONTAL_ALIGNMENT_LEFT, -1, size, DamageStyle.outline_px(), Color(COLOR_OUTLINE, a))
		draw_string(font, p, text, HORIZONTAL_ALIGNMENT_LEFT, -1, size, Color(col, a))
