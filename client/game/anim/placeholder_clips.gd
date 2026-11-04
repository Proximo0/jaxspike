class_name PlaceholderClips
extends RefCounted
## Procedural placeholder sheets in RO's chibi anime style (round2 contract 1.4 art direction): big head, small
## body, dark outline, two-tone cel shading, saturated colours, bouncy walk, visible swing, flinch and fall.
## `make_sheet` paints every clip of a job or monster look once, at runtime, into images in the 1.4 layout (rows 0..4,
## directions 5..7 mirrored) and returns an ActSheet, so placeholders play through exactly the path art-anim's sheets
## use and cost one texture blit per entity per frame. `pose` turns (clip, frame) into body parameters.
## Painting is span fills on an Image (native fill_rect per row of each ellipse), about 0.3 ms per frame cell.

const PLAYER_ACTIONS: Dictionary = {
	"idle": {"frames": 4, "delay_ms": 220.0}, "walk": {"frames": 8, "delay_ms": 100.0},
	"sit": {"frames": 1, "delay_ms": 200.0}, "pickup": {"frames": 3, "delay_ms": 100.0},
	"standby": {"frames": 4, "delay_ms": 150.0}, "attack": {"frames": 6, "delay_ms": 100.0},
	"hurt": {"frames": 2, "delay_ms": 150.0}, "dead": {"frames": 3, "delay_ms": 150.0},
	"cast": {"frames": 4, "delay_ms": 150.0},
}
const MONSTER_ACTIONS: Dictionary = {
	"idle": {"frames": 4, "delay_ms": 200.0}, "walk": {"frames": 6, "delay_ms": 120.0},
	"attack": {"frames": 5, "delay_ms": 100.0}, "hurt": {"frames": 2, "delay_ms": 150.0},
	"dead": {"frames": 3, "delay_ms": 150.0},
}
const ATK_FRAME: int = 3
const FRAME_W: int = 56
const FRAME_H: int = 60
const ANCHOR: Vector2 = Vector2(28.0, 55.0)  ## feet pixel in a frame
const ROWS: int = 5  ## directions 0..4; 5..7 mirror 3..1
const OUTLINE: Color = Color(0.13, 0.08, 0.12)
const SKIN: Color = Color(1.0, 0.86, 0.74)
const EYE: Color = Color(0.16, 0.12, 0.30)
const BLADE: Color = Color(0.88, 0.92, 1.0)
const SHADOW: Color = Color(0.0, 0.0, 0.0, 0.28)
## Per job, AssetSlots.JOB_NAMES order (novice, swordsman, mage, archer, acolyte, thief, merchant): [outfit, hair],
## saturated RO-like palettes.
const JOB_COLORS: Array = [
	[Color(0.85, 0.62, 0.30), Color(0.55, 0.30, 0.15)], [Color(0.25, 0.45, 0.90), Color(0.95, 0.80, 0.30)],
	[Color(0.55, 0.25, 0.75), Color(0.90, 0.35, 0.45)], [Color(0.30, 0.70, 0.30), Color(0.95, 0.60, 0.20)],
	[Color(0.95, 0.95, 0.98), Color(0.35, 0.55, 0.95)], [Color(0.30, 0.30, 0.40), Color(0.75, 0.20, 0.25)],
	[Color(0.90, 0.75, 0.25), Color(0.40, 0.25, 0.20)],
]
## Monster body colours by art index (Blobbin pink first), cycled.
const MONSTER_COLORS: Array[Color] = [
	Color(1.0, 0.55, 0.70), Color(0.55, 0.85, 0.35), Color(0.45, 0.65, 1.0), Color(1.0, 0.75, 0.25),
	Color(0.75, 0.50, 0.95), Color(0.95, 0.40, 0.30), Color(0.40, 0.85, 0.85), Color(0.85, 0.85, 0.85),
]

## Painter state for one frame cell: image, cell origin, feet, pose transform, hurt tint.
var _img: Image = null
var _cell: Vector2i = Vector2i.ZERO
var _feet: Vector2 = ANCHOR
var _sx: float = 1.0
var _sy: float = 1.0
var _flash: float = 0.0


static func actions_for(is_player: bool) -> Dictionary:
	return PLAYER_ACTIONS if is_player else MONSTER_ACTIONS


## Act-style clip info for an engine action: {name, frames, delay_ms, atk_frame} (the action falls back like ActSheet).
static func clip(action: int, is_player: bool) -> Dictionary:
	var acts: Dictionary = actions_for(is_player)
	var a: int = ClipSelect.clip_action(action, is_player)
	var tries: int = 0
	while not acts.has(ClipSelect.NAMES[a]) and tries < 3:
		a = ClipSelect.fallback(a)
		tries += 1
	var nm: String = ClipSelect.NAMES[a] if acts.has(ClipSelect.NAMES[a]) else "idle"
	var info: Dictionary = acts[nm]
	return {"name": nm, "frames": int(info["frames"]), "delay_ms": float(info["delay_ms"]), "atk_frame": ATK_FRAME}


## Act meta (ActSheet.parse format) of the placeholder sheet.
static func meta(is_player: bool) -> Dictionary:
	var acts: Dictionary = {}
	var src: Dictionary = actions_for(is_player)
	for nm: String in src:
		acts[nm] = {"row0": 0, "frames": int(src[nm]["frames"]), "delay_ms": float(src[nm]["delay_ms"]), "mirror": true,
				"rows": ROWS, "image": ""}
	return {"frame_w": FRAME_W, "frame_h": FRAME_H, "anchor": ANCHOR, "atk_frame": ATK_FRAME, "image": "",
			"actions": acts}


## Paints the whole placeholder sheet of a job (players) or monster art index and wraps it in an ActSheet.
static func make_sheet(is_player: bool, look: int) -> ActSheet:
	var textures: Dictionary = {}
	for nm: String in actions_for(is_player):
		textures[nm] = ImageTexture.create_from_image(paint_action(is_player, look, nm))
	var sheet: ActSheet = ActSheet.build(meta(is_player), textures)
	sheet.source = "placeholder:%s.%d" % ["job" if is_player else "monster", look]
	return sheet


## One action's image: frames across, directions 0..4 down.
static func paint_action(is_player: bool, look: int, clip_name: String) -> Image:
	var n: int = int(actions_for(is_player)[clip_name]["frames"])
	var img: Image = Image.create(n * FRAME_W, ROWS * FRAME_H, false, Image.FORMAT_RGBA8)
	var p := PlaceholderClips.new()
	p._img = img
	for row: int in range(ROWS):
		for f: int in range(n):
			p._paint_cell(Vector2i(f * FRAME_W, row * FRAME_H), is_player, look, clip_name, f, n, row)
	return img


## Body parameters of a clip frame: bob (px up), sx/sy (squash), lean (px toward the face), arm (weapon angle,
## radians, 0 = down, + = forward), leg (stride -1..1), lie (0..1 fallen), flash (0..1 hurt tint), arms_up.
static func pose(clip_name: String, frame: int, frames: int) -> Dictionary:
	var p: Dictionary = {"bob": 0.0, "sx": 1.0, "sy": 1.0, "lean": 0.0, "arm": 0.3, "leg": 0.0, "lie": 0.0,
			"flash": 0.0, "arms_up": 0.0}
	var n: int = maxi(frames, 1)
	var u: float = float(frame) / float(n)
	match clip_name:
		"idle":
			p["bob"] = sin(u * TAU) * 1.0
			p["sy"] = 1.0 + sin(u * TAU) * 0.03
		"standby":
			p["bob"] = absf(sin(u * TAU)) * 1.5
			p["lean"] = 1.5
			p["arm"] = 0.9
		"walk":
			var ph: float = u * TAU  # one cycle = two cells: a stride per half cycle
			p["leg"] = sin(ph)
			p["bob"] = absf(sin(ph)) * 3.0
			p["sy"] = 1.0 - absf(cos(ph)) * 0.06
			p["sx"] = 1.0 + absf(cos(ph)) * 0.05
			p["arm"] = 0.3 - sin(ph) * 0.5
		"sit":
			p["sy"] = 0.78
			p["sx"] = 1.08
		"pickup":
			p["sy"] = 0.82 if frame == 1 else 0.92
			p["lean"] = 3.0
			p["arm"] = 1.2
		"attack":
			if frame < ATK_FRAME:  # wind up: arm back, lean back
				var w: float = float(frame + 1) / float(ATK_FRAME)
				p["arm"] = -0.6 - 1.4 * w
				p["lean"] = -2.0 * w
				p["sy"] = 1.0 + 0.05 * w
			else:  # swing through and recover
				var r: float = float(frame - ATK_FRAME) / float(maxi(n - ATK_FRAME, 1))
				p["arm"] = 1.9 - 1.0 * r
				p["lean"] = 4.0 * (1.0 - r)
				p["sx"] = 1.08 - 0.08 * r
				p["sy"] = 0.94 + 0.06 * r
		"hurt":
			p["lean"] = -3.5
			p["flash"] = 1.0 if frame == 0 else 0.5
			p["sy"] = 0.92
			p["arm"] = -0.4
		"dead":
			p["lie"] = float(frame + 1) / float(n)
		"cast":
			p["arms_up"] = 1.0
			p["bob"] = 1.0 + sin(u * TAU) * 1.0
			p["arm"] = 2.6
	return p


static func job_colors(job: int) -> Array:
	return JOB_COLORS[posmod(job, JOB_COLORS.size())]


static func monster_color(art_index: int) -> Color:
	return MONSTER_COLORS[posmod(art_index, MONSTER_COLORS.size())]


func _paint_cell(cell: Vector2i, is_player: bool, look: int, clip_name: String, frame: int, frames: int, face: int) -> void:
	var p: Dictionary = pose(clip_name, frame, frames)
	var fd: Vector2i = ClipSelect.FACE_DELTAS[face]
	_cell = cell
	_sx = 1.0
	_sy = 1.0
	_flash = 0.0
	_feet = ANCHOR
	_fill_ellipse(Vector2.ZERO, Vector2(11.0, 3.4), SHADOW)
	_feet = ANCHOR + Vector2(float(p["lean"]) * float(fd.x), 0.0)
	_sx = float(p["sx"])
	_sy = float(p["sy"])
	_flash = float(p["flash"])
	if is_player:
		_paint_player(p, look, fd)
	else:
		_paint_monster(p, look, fd, clip_name)


## Local figure coordinates (feet at 0, up negative) to image pixels, with the pose squash.
func _px(v: Vector2) -> Vector2:
	return Vector2(_cell) + _feet + Vector2(v.x * _sx, v.y * _sy)


func _fill_ellipse(c: Vector2, r: Vector2, col: Color) -> void:
	var cc: Vector2 = _px(c)
	var rr := Vector2(r.x * _sx, r.y * _sy)
	if rr.x <= 0.0 or rr.y <= 0.0:
		return
	var y0: int = maxi(int(floorf(cc.y - rr.y)), _cell.y)
	var y1: int = mini(int(ceilf(cc.y + rr.y)), _cell.y + FRAME_H - 1)
	for y: int in range(y0, y1 + 1):
		var dy: float = (float(y) + 0.5 - cc.y) / rr.y
		if absf(dy) >= 1.0:
			continue
		var hw: float = rr.x * sqrt(1.0 - dy * dy)
		var x0: int = maxi(roundi(cc.x - hw), _cell.x)
		var x1: int = mini(roundi(cc.x + hw), _cell.x + FRAME_W)
		if x1 > x0:
			_img.fill_rect(Rect2i(x0, y, x1 - x0, 1), col)


func _fill_rect(r: Rect2, col: Color) -> void:
	var a: Vector2 = _px(r.position)
	var b: Vector2 = _px(r.end)
	var lo := Vector2i(maxi(roundi(minf(a.x, b.x)), _cell.x), maxi(roundi(minf(a.y, b.y)), _cell.y))
	var hi := Vector2i(mini(roundi(maxf(a.x, b.x)), _cell.x + FRAME_W), mini(roundi(maxf(a.y, b.y)), _cell.y + FRAME_H))
	if hi.x > lo.x and hi.y > lo.y:
		_img.fill_rect(Rect2i(lo, hi - lo), col)


## A thick line as a row of discs.
func _line(a: Vector2, b: Vector2, width: float, col: Color) -> void:
	var steps: int = maxi(int(ceilf(a.distance_to(b))), 1)
	for i: int in range(steps + 1):
		_fill_ellipse(a.lerp(b, float(i) / float(steps)), Vector2(width, width) * 0.5, col)


## Outlined, two-tone cel-shaded ellipse: outline, shadow tone, base tone offset up-left, small highlight.
func _blob(c: Vector2, r: Vector2, base: Color, shine: bool = true) -> void:
	var col: Color = base.lerp(Color(1.0, 0.35, 0.35), _flash * 0.6)
	_fill_ellipse(c, r + Vector2(1.5, 1.5), OUTLINE)
	_fill_ellipse(c, r, col.darkened(0.28))
	_fill_ellipse(c + Vector2(-r.x * 0.12, -r.y * 0.14), r * 0.84, col)
	if shine:
		_fill_ellipse(c + Vector2(-r.x * 0.42, -r.y * 0.45), r * 0.2, col.lightened(0.55))


func _paint_player(p: Dictionary, job: int, fd: Vector2i) -> void:
	var cols: Array = job_colors(job)
	var outfit: Color = cols[0]
	var hair: Color = cols[1]
	var bob: float = float(p["bob"])
	var dx: float = float(fd.x)
	var back: bool = fd.y < 0  # facing north: we see the back of the head
	var lie: float = float(p["lie"])
	var side_x: float = dx if dx != 0.0 else 1.0
	if lie > 0.0:
		_paint_fallen(lie, outfit, hair, side_x)
		return
	var stride: float = float(p["leg"])
	for side: int in [-1, 1]:  # short chibi legs; the stride swings them along the facing
		var lift: float = maxf(0.0, stride * float(side)) * 2.5
		var swing: float = stride * float(side) * 2.0 * (1.0 if dx != 0.0 else 0.4)
		_blob(Vector2(float(side) * 3.4 + swing, -3.4 - lift), Vector2(2.6, 3.4), outfit.darkened(0.45), false)
	var body_c := Vector2(0.0, -11.5 - bob)
	var arm: float = float(p["arm"])
	var shoulder: Vector2 = body_c + Vector2(5.0 * side_x, -2.0)
	var tip: Vector2 = shoulder + Vector2(sin(arm) * side_x, cos(arm)) * 13.0
	var blade_behind: bool = back
	if blade_behind:
		_paint_blade(shoulder, tip)
	_blob(body_c, Vector2(7.0, 6.8), outfit)
	if not blade_behind:
		_paint_blade(shoulder, tip)
	_blob(shoulder, Vector2(2.3, 2.3), SKIN, false)
	if float(p["arms_up"]) > 0.0:
		_blob(body_c + Vector2(-5.0 * side_x, -9.0), Vector2(2.3, 2.3), SKIN, false)
		_fill_ellipse(body_c + Vector2(0.0, -28.0 + bob), Vector2(3.0, 3.0), Color(0.6, 0.9, 1.0, 0.85))
		_fill_ellipse(body_c + Vector2(0.0, -28.0 + bob), Vector2(1.4, 1.4), Color.WHITE)
	var head_c := Vector2(dx * 0.8, -26.0 - bob)  # big head: about 40% of the height
	if back:
		_blob(head_c, Vector2(10.5, 9.8), hair)
		_fill_ellipse(head_c + Vector2(0.0, 2.5), Vector2(7.0, 5.0), hair.darkened(0.15))
		return
	_blob(head_c, Vector2(10.0, 9.4), SKIN)
	_blob(head_c + Vector2(-dx * 1.2, -4.5), Vector2(10.6, 5.8), hair)  # fringe
	var sideways: bool = fd.y == 0
	for side: int in [-1, 1]:
		if sideways and float(side) == -dx:
			continue  # side view: only the near eye
		var e: Vector2 = head_c + Vector2(dx * 2.6 + float(side) * 3.7, 1.8)
		_fill_rect(Rect2(e - Vector2(1.4, 2.6), Vector2(2.8, 5.0)), EYE)
		_fill_rect(Rect2(e - Vector2(1.0, 2.2), Vector2(1.2, 1.4)), Color.WHITE)
	_fill_ellipse(head_c + Vector2(dx * 3.0 - 5.5, 4.5), Vector2(1.6, 0.9), Color(1.0, 0.55, 0.6, 0.9))  # blush
	_fill_ellipse(head_c + Vector2(dx * 3.0 + 5.5, 4.5), Vector2(1.6, 0.9), Color(1.0, 0.55, 0.6, 0.9))


func _paint_blade(a: Vector2, b: Vector2) -> void:
	_line(a, b, 3.6, OUTLINE)
	_line(a, b, 1.6, BLADE)


## Falling over in three steps: kneel, half down, flat on the ground with x_x eyes.
func _paint_fallen(lie: float, outfit: Color, hair: Color, side_x: float) -> void:
	var k: float = clampf(lie, 0.0, 1.0)
	var body_c := Vector2(-4.0 * k * side_x, -11.0 + 6.0 * k)
	var head_c := Vector2(-14.0 * k * side_x, -25.0 + 19.0 * k)
	_blob(Vector2(6.0 * k * side_x, -3.0), Vector2(3.0 + 2.0 * k, 2.6), outfit.darkened(0.45), false)
	_blob(body_c, Vector2(7.0 + 2.0 * k, 6.8 - 2.0 * k), outfit)
	_blob(head_c, Vector2(10.0, 9.4 - 1.5 * k), SKIN)
	_blob(head_c + Vector2(side_x * 4.0 * k, -4.0 * (1.0 - k)), Vector2(9.0, 5.5), hair)
	if k >= 0.99:
		for ex: float in [-3.5, 3.5]:
			var e: Vector2 = head_c + Vector2(ex, 1.5)
			_line(e + Vector2(-1.5, -1.5), e + Vector2(1.5, 1.5), 1.0, OUTLINE)
			_line(e + Vector2(-1.5, 1.5), e + Vector2(1.5, -1.5), 1.0, OUTLINE)


func _paint_monster(p: Dictionary, look: int, fd: Vector2i, clip_name: String) -> void:
	var base: Color = monster_color(look)
	var hop: float = float(p["bob"]) * (2.4 if clip_name == "walk" else 1.0)
	var lie: float = float(p["lie"])
	var r := Vector2(12.0 + lie * 5.0, 10.0 - lie * 5.0)
	var c := Vector2(0.0, -r.y - hop)
	_blob(c, r, base)
	_fill_ellipse(c + Vector2(r.x * 0.15, -r.y * 0.9), Vector2(2.0, 3.0), base.darkened(0.35))  # little sprout
	if lie >= 0.99:
		for ex: float in [-4.0, 4.0]:
			var e: Vector2 = c + Vector2(ex, 0.0)
			_line(e + Vector2(-1.5, -1.5), e + Vector2(1.5, 1.5), 1.0, OUTLINE)
			_line(e + Vector2(-1.5, 1.5), e + Vector2(1.5, -1.5), 1.0, OUTLINE)
		return
	if fd.y < 0:
		return  # facing away: no face
	var dx: float = float(fd.x)
	for side: int in [-1, 1]:
		var e: Vector2 = c + Vector2(dx * 3.0 + float(side) * 4.0, -1.0)
		_fill_rect(Rect2(e - Vector2(1.3, 2.4), Vector2(2.6, 4.6)), EYE)
		_fill_rect(Rect2(e - Vector2(0.9, 2.0), Vector2(1.0, 1.2)), Color.WHITE)
	var mouth_open: float = 2.4 if clip_name == "attack" else 0.9
	_fill_ellipse(c + Vector2(dx * 3.0, 4.0), Vector2(2.0, mouth_open), OUTLINE)
