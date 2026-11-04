class_name CardSparkle
extends Node2D
## Twinkling gold sparkles and a soft glow over card drops on the ground (round 4, ro4-cards), so a 1-in-10,000 drop
## reads at a glance in the anime look. EntityLayer hands it the drawn positions of card drop entities every frame
## (`show_at`); it redraws only while it has points. `star` is pure and unit tested.

const STARS: int = 4
const PERIOD_MS: float = 900.0
const GLOW_COLOR: Color = Color(1.0, 0.85, 0.3, 0.28)
const STAR_COLOR: Color = Color(1.0, 0.97, 0.75, 1.0)
const STAR_EDGE: Color = Color(1.0, 0.7, 0.2, 1.0)

var points: PackedVector2Array = PackedVector2Array()
var now_ms: float = 0.0
var tile_px: float = 32.0


func show_at(pts: PackedVector2Array, render_ms: float, tile: float) -> void:
	if pts.is_empty() and points.is_empty():
		return
	points = pts
	now_ms = render_ms
	tile_px = tile
	queue_redraw()


## Star `i` of a drop at time `t_ms`: Vector3(dx, dy, size) in tiles; size 0 = between twinkles.
static func star(i: int, t_ms: float) -> Vector3:
	var phase: float = fposmod(t_ms / PERIOD_MS + float(i) / float(STARS), 1.0)
	var ang: float = TAU * float(i) / float(STARS) + 0.6
	var r: float = 0.32 + 0.08 * float(i % 2)
	var s: float = sin(phase * PI)
	return Vector3(cos(ang) * r, sin(ang) * r - 0.25, 0.16 * s * s)


func _draw() -> void:
	for p: Vector2 in points:
		draw_circle(p + Vector2(0.0, -0.1 * tile_px), 0.42 * tile_px, GLOW_COLOR)
		for i: int in range(STARS):
			var st: Vector3 = star(i, now_ms)
			if st.z <= 0.01:
				continue
			var c: Vector2 = p + Vector2(st.x, st.y) * tile_px
			var a: float = st.z * tile_px
			var b: float = a * 0.28
			var pts := PackedVector2Array([c + Vector2(0, -a), c + Vector2(b, -b), c + Vector2(a, 0), c + Vector2(b, b),
					c + Vector2(0, a), c + Vector2(-b, b), c + Vector2(-a, 0), c + Vector2(-b, -b)])
			draw_colored_polygon(pts, STAR_EDGE)
			var inner := PackedVector2Array()
			for q: Vector2 in pts:
				inner.append(c + (q - c) * 0.6)
			draw_colored_polygon(inner, STAR_COLOR)
