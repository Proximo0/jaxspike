class_name CursorState
extends RefCounted
## RO cursor states from what is under the mouse: sword on a monster, speech bubble on an NPC tile (kafra) or a
## gather/craft node, grab hand on a drop, crossed arrow on blocked terrain, plain arrow (walk) elsewhere.
## `pick` is pure; `make_image` draws the placeholder cursor art (chibi-bright, dark outline) once per state.

const WALK: int = 0
const ATTACK: int = 1
const TALK: int = 2
const PICKUP: int = 3
const BLOCKED: int = 4
const NAMES: Array[String] = ["walk", "attack", "talk", "pickup", "blocked"]
const NO_ENTITY: int = -1
const KAFRA_TERRAIN: int = 9
const SIZE: int = 32
const OUTLINE: Color = Color(0.10, 0.08, 0.12)


## `hover_cat` is the ViewMath category of the entity under the mouse (NO_ENTITY for none, the own player counts
## as none); `terrain_id` the tile under the mouse (-1 outside the map).
static func pick(hover_cat: int, terrain_id: int) -> int:
	match hover_cat:
		ViewMath.CAT_MONSTER:
			return ATTACK
		ViewMath.CAT_DROP:
			return PICKUP
		ViewMath.CAT_NODE:
			return TALK
	if terrain_id == KAFRA_TERRAIN:
		return TALK
	if terrain_id < 0 or ViewMath.is_blocked_terrain(terrain_id):
		return BLOCKED
	return WALK


## Hotspot of each state's image (the arrow tip, or the centre of the round icons).
static func hotspot(state: int) -> Vector2:
	return Vector2(2, 2) if state == WALK or state == BLOCKED else Vector2(SIZE / 2, SIZE / 2)


## A SIZE x SIZE RGBA cursor image for `state`.
static func make_image(state: int) -> Image:
	var img: Image = Image.create(SIZE, SIZE, false, Image.FORMAT_RGBA8)
	img.fill(Color(0, 0, 0, 0))
	match state:
		ATTACK:
			# Sword: diagonal blade (silver) with a gold guard.
			for i: int in range(4, 24):
				_cursor_dot(img, Vector2i(i, i), 2, Color(0.92, 0.94, 1.0))
			for i: int in range(18, 26):
				_cursor_dot(img, Vector2i(i, 42 - i), 1, Color(1.0, 0.80, 0.25))
			for i: int in range(23, 29):
				_cursor_dot(img, Vector2i(i, i), 1, Color(0.55, 0.30, 0.15))
		TALK:
			# Speech bubble: white ellipse with a tail and three dots.
			for y: int in range(4, 22):
				for x: int in range(3, 29):
					var dx: float = (float(x) - 16.0) / 13.0
					var dy: float = (float(y) - 13.0) / 9.0
					if dx * dx + dy * dy <= 1.0:
						img.set_pixel(x, y, Color(1, 1, 1))
			for i: int in range(0, 6):
				_cursor_dot(img, Vector2i(10 - i, 21 + i), 1, Color(1, 1, 1))
			for x: int in [10, 16, 22]:
				_cursor_dot(img, Vector2i(x, 13), 1, Color(0.35, 0.55, 0.95))
		PICKUP:
			# Grab hand: a pink palm with four fingers.
			for y: int in range(14, 26):
				for x: int in range(9, 23):
					img.set_pixel(x, y, Color(1.0, 0.82, 0.72))
			for f: int in range(4):
				for y: int in range(6, 14):
					_cursor_dot(img, Vector2i(10 + f * 4, y), 1, Color(1.0, 0.82, 0.72))
		_:
			# Arrow (walk), red-tinted when blocked.
			var fill: Color = Color(1.0, 0.95, 0.80) if state == WALK else Color(1.0, 0.40, 0.40)
			for y: int in range(0, 22):
				for x: int in range(0, (y * 2) / 3 + 1):
					img.set_pixel(2 + x, 2 + y, fill)
			for i: int in range(0, 9):
				_cursor_dot(img, Vector2i(8 + i / 2, 18 + i), 1, fill)
			if state == BLOCKED:
				for i: int in range(18, 30):
					_cursor_dot(img, Vector2i(i, i), 1, Color(0.85, 0.05, 0.05))
					_cursor_dot(img, Vector2i(i, 48 - i), 1, Color(0.85, 0.05, 0.05))
	_cursor_outline(img)
	return img


static func _cursor_dot(img: Image, c: Vector2i, r: int, col: Color) -> void:
	for y: int in range(c.y - r, c.y + r + 1):
		for x: int in range(c.x - r, c.x + r + 1):
			if x >= 0 and y >= 0 and x < SIZE and y < SIZE:
				img.set_pixel(x, y, col)


## Dark 1 px outline around every opaque pixel (the clean RO cartoon edge).
static func _cursor_outline(img: Image) -> void:
	var src: Image = img.duplicate() as Image
	for y: int in range(SIZE):
		for x: int in range(SIZE):
			if src.get_pixel(x, y).a > 0.0:
				continue
			for d: Vector2i in [Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1), Vector2i(0, -1)]:
				var nx: int = x + d.x
				var ny: int = y + d.y
				if nx >= 0 and ny >= 0 and nx < SIZE and ny < SIZE and src.get_pixel(nx, ny).a > 0.0:
					img.set_pixel(x, y, OUTLINE)
					break
