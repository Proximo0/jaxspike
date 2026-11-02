class_name SpriteSheetLoader
extends RefCounted
## Frame-table JSON -> SpriteFrames, plus an AnimatedSprite2D factory. Pure parsing and resource building.

const DEFAULT_FPS: float = 6.0


## True for an int, or a float with no fraction (JSON numbers parse as float).
static func _is_whole(v: Variant) -> bool:
	if v is int:
		return true
	if v is float:
		var f: float = v
		return f == floorf(f)
	return false


## Parse sheet JSON text. {} on any structural error; bad animations are skipped and named in "errors".
static func parse(text: String) -> Dictionary:
	var json: JSON = JSON.new()
	if json.parse(text) != OK:
		return {}
	var data: Variant = json.data
	if not (data is Dictionary):
		return {}
	var src: Dictionary = data
	if not (src.get("image") is String):
		return {}
	if not _is_whole(src.get("frame_w")) or not _is_whole(src.get("frame_h")):
		return {}
	if not (src.get("animations") is Dictionary):
		return {}
	var errors: PackedStringArray = PackedStringArray()
	var anims_out: Dictionary = {}
	var anims_in: Dictionary = src["animations"]
	for key: Variant in anims_in.keys():
		var anim_name: String = str(key)
		var parsed: Dictionary = _parse_anim(anims_in[key])
		if parsed.is_empty():
			errors.append(anim_name)
		else:
			anims_out[anim_name] = parsed
	var margin: int = int(src["margin"]) if _is_whole(src.get("margin")) else 0
	var spacing: int = int(src["spacing"]) if _is_whole(src.get("spacing")) else 0
	return {
		"image": String(src["image"]),
		"frame_w": int(src["frame_w"]),
		"frame_h": int(src["frame_h"]),
		"margin": margin,
		"spacing": spacing,
		"animations": anims_out,
		"errors": errors,
	}


static func _parse_anim(v: Variant) -> Dictionary:
	if not (v is Dictionary):
		return {}
	var d: Dictionary = v
	if not (d.get("frames") is Array):
		return {}
	var raw: Array = d["frames"]
	if raw.is_empty():
		return {}
	var cells: Array[Vector2i] = []
	for i: int in range(raw.size()):
		var cell: Variant = raw[i]
		if not (cell is Array):
			return {}
		var pair: Array = cell
		if pair.size() != 2 or not _is_whole(pair[0]) or not _is_whole(pair[1]):
			return {}
		cells.append(Vector2i(int(pair[0]), int(pair[1])))
	var fps: float = DEFAULT_FPS
	var fps_raw: Variant = d.get("fps")
	if fps_raw is int or fps_raw is float:
		fps = float(fps_raw)
	var loop: bool = true
	var loop_raw: Variant = d.get("loop")
	if loop_raw is bool:
		loop = loop_raw
	return {"frames": cells, "fps": fps, "loop": loop}


static func frame_rect(sheet: Dictionary, col: int, row: int) -> Rect2i:
	var w: int = int(sheet.get("frame_w", 0))
	var h: int = int(sheet.get("frame_h", 0))
	var margin: int = int(sheet.get("margin", 0))
	var spacing: int = int(sheet.get("spacing", 0))
	return Rect2i(margin + col * (w + spacing), margin + row * (h + spacing), w, h)


static func build_frames(sheet: Dictionary, texture: Texture2D) -> SpriteFrames:
	var frames: SpriteFrames = SpriteFrames.new()
	frames.remove_animation("default")
	var anims: Dictionary = sheet.get("animations", {})
	for key: Variant in anims.keys():
		var anim_name: String = str(key)
		var a: Dictionary = anims[key]
		frames.add_animation(anim_name)
		frames.set_animation_speed(anim_name, float(a.get("fps", DEFAULT_FPS)))
		frames.set_animation_loop(anim_name, bool(a.get("loop", true)))
		var cells: Array = a.get("frames", [])
		for i: int in range(cells.size()):
			var cell: Vector2i = cells[i]
			var atlas: AtlasTexture = AtlasTexture.new()
			atlas.atlas = texture
			atlas.region = Rect2(frame_rect(sheet, cell.x, cell.y))
			frames.add_frame(anim_name, atlas)
	return frames


## Read the JSON, load its image; null when either is missing or invalid.
static func load_sheet(json_path: String) -> SpriteFrames:
	if not FileAccess.file_exists(json_path):
		return null
	var text: String = FileAccess.get_file_as_string(json_path)
	var sheet: Dictionary = parse(text)
	if sheet.is_empty():
		return null
	var image_path: String = sheet["image"]
	if not ResourceLoader.exists(image_path):
		return null
	var tex: Texture2D = load(image_path) as Texture2D
	if tex == null:
		return null
	return build_frames(sheet, tex)


static func make_sprite(frames: SpriteFrames, anim: String) -> AnimatedSprite2D:
	var sprite: AnimatedSprite2D = AnimatedSprite2D.new()
	sprite.sprite_frames = frames
	sprite.centered = true
	var names: PackedStringArray = frames.get_animation_names()
	if frames.has_animation(anim):
		sprite.play(anim)
	elif names.size() > 0:
		sprite.play(names[0])
	return sprite


static func has_anim(frames: SpriteFrames, anim: String) -> bool:
	return frames != null and frames.has_animation(anim)
