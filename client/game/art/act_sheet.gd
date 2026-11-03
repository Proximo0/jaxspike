class_name ActSheet
extends RefCounted
## An 8-direction action sheet (round2 contract 1.4) as SpriteFrames: animation "<action>_<row>" per action and
## present direction row. JSON "act":
##   {"frame_w", "frame_h", "anchor": [x, y], "atk_frame": 3, "image": "x.png" (optional),
##    "actions": {"idle": {"row0": 0, "frames": 4, "delay_ms": 150, "mirror": true, "image": "x_idle.png"}, ...}}
## Rows of an action start at `row0` of its image (the action's own "image" or "file", else the top-level "image",
## else "<json name>_<action>.png", else "<action>.png", else "<json name>.png"; relative names resolve next to the
## JSON, res:// paths are used as they are); 8 rows, or 5 (directions 0..4) when "mirror" is true and
## "rows" says 5 or the image has fewer than row0 + 8 rows; directions 5..7 then mirror 3..1 (ClipSelect.sheet_row).
## Pure: parse() and build() take text and images; load_act() reads files.

var frame_w: int = 0
var frame_h: int = 0
var anchor: Vector2 = Vector2.ZERO  ## pixel in a frame that sits on the entity's cell centre (feet)
var atk_frame: int = -1
var actions: Dictionary = {}  ## name -> {frames, delay_ms, rows, mirror}
var frames: SpriteFrames = SpriteFrames.new()
var source: String = ""


static func _is_num(v: Variant) -> bool:
	return v is int or v is float


## Parses act JSON text: {} on a structural error; actions with bad fields are skipped.
static func parse(text: String) -> Dictionary:
	var json := JSON.new()
	if json.parse(text) != OK or not (json.data is Dictionary):
		return {}
	var src: Dictionary = json.data
	if not _is_num(src.get("frame_w")) or not _is_num(src.get("frame_h")) or not (src.get("actions") is Dictionary):
		return {}
	var fw: int = int(src["frame_w"])
	var fh: int = int(src["frame_h"])
	if fw <= 0 or fh <= 0:
		return {}
	var out_actions: Dictionary = {}
	var acts: Dictionary = src["actions"]
	for key: Variant in acts:
		var a: Variant = acts[key]
		if not (a is Dictionary):
			continue
		var d: Dictionary = a
		if not _is_num(d.get("frames")) or int(d["frames"]) <= 0:
			continue
		var rows: int = int(d["rows"]) if _is_num(d.get("rows")) else (int(src["rows"]) if _is_num(src.get("rows")) else -1)
		out_actions[str(key)] = {
			"row0": int(d["row0"]) if _is_num(d.get("row0")) else 0,
			"frames": int(d["frames"]),
			"delay_ms": float(d["delay_ms"]) if _is_num(d.get("delay_ms")) else 150.0,
			"mirror": bool(d.get("mirror", true)),
			"rows": rows,
			"image": String(d["image"]) if d.get("image") is String else "",
			"file": String(d["file"]) if d.get("file") is String else "",
		}
	var anc := Vector2(fw * 0.5, fh * 0.9)
	var anc_raw: Variant = src.get("anchor")
	if anc_raw is Array and (anc_raw as Array).size() >= 2 and _is_num(anc_raw[0]) and _is_num(anc_raw[1]):
		anc = Vector2(float(anc_raw[0]), float(anc_raw[1]))
	return {
		"frame_w": fw, "frame_h": fh, "anchor": anc, "actions": out_actions,
		"atk_frame": int(src["atk_frame"]) if _is_num(src.get("atk_frame")) else -1,
		"image": String(src["image"]) if src.get("image") is String else "",
	}


## Builds a sheet from parsed meta and textures keyed by action name (one texture may serve several actions).
static func build(meta: Dictionary, textures: Dictionary) -> ActSheet:
	var sheet := ActSheet.new()
	if meta.is_empty():
		return sheet
	sheet.frame_w = int(meta["frame_w"])
	sheet.frame_h = int(meta["frame_h"])
	sheet.anchor = meta["anchor"]
	sheet.atk_frame = int(meta["atk_frame"])
	if sheet.frames.has_animation(&"default"):
		sheet.frames.remove_animation(&"default")
	var acts: Dictionary = meta["actions"]
	for action_name: String in acts:
		var tex: Texture2D = textures.get(action_name) as Texture2D
		if tex == null:
			continue
		var a: Dictionary = acts[action_name]
		var row0: int = int(a["row0"])
		var n: int = int(a["frames"])
		var avail: int = int(tex.get_height()) / sheet.frame_h - row0
		var rows: int = int(a["rows"])
		if rows <= 0:
			rows = 8 if avail >= 8 or not bool(a["mirror"]) else 5
		rows = mini(rows, avail)
		if rows <= 0:
			continue
		var cols: int = int(tex.get_width()) / sheet.frame_w
		n = mini(n, cols)
		if n <= 0:
			continue
		for r: int in range(rows):
			var anim := StringName("%s_%d" % [action_name, r])
			sheet.frames.add_animation(anim)
			var idx: int = ClipSelect.NAMES.find(action_name)
			sheet.frames.set_animation_loop(anim, idx < 0 or ClipSelect.loops(idx))
			sheet.frames.set_animation_speed(anim, 1000.0 / maxf(float(a["delay_ms"]), 1.0))
			for f: int in range(n):
				var at := AtlasTexture.new()
				at.atlas = tex
				at.region = Rect2(f * sheet.frame_w, (row0 + r) * sheet.frame_h, sheet.frame_w, sheet.frame_h)
				sheet.frames.add_frame(anim, at)
		sheet.actions[action_name] = {"frames": n, "delay_ms": float(a["delay_ms"]), "rows": rows, "mirror": bool(a["mirror"])}
	return sheet


## Loads "<dir>/<name>.json" and its images; null when the JSON is missing or broken or no action has an image.
static func load_act(json_path: String) -> ActSheet:
	if not FileAccess.file_exists(json_path):
		return null
	var meta: Dictionary = parse(FileAccess.get_file_as_string(json_path))
	if meta.is_empty():
		return null
	var dir: String = json_path.get_base_dir()
	var stem: String = json_path.get_file().get_basename()
	var textures: Dictionary = {}
	var acts: Dictionary = meta["actions"]
	for action_name: String in acts:
		var a: Dictionary = acts[action_name]
		for cand: String in [String(a["image"]), String(a.get("file", "")), String(meta["image"]),
				"%s_%s.png" % [stem, action_name], "%s.png" % action_name, "%s.png" % stem]:
			if cand == "":
				continue
			var tex: Texture2D = _act_sheet_texture(cand if cand.contains("://") else dir.path_join(cand))
			if tex != null:
				textures[action_name] = tex
				break
	if textures.is_empty():
		return null
	var sheet: ActSheet = build(meta, textures)
	sheet.source = json_path
	return sheet if not sheet.actions.is_empty() else null


static func _act_sheet_texture(path: String) -> Texture2D:
	if ResourceLoader.exists(path):
		return load(path) as Texture2D
	if FileAccess.file_exists(path):
		var img: Image = Image.load_from_file(path)
		if img != null and not img.is_empty():
			return ImageTexture.create_from_image(img)
	return null


func has_action(action_name: String) -> bool:
	return actions.has(action_name)


## Clip for an engine action and face: {name, anim, flip, frames, delay_ms}; falls back through ClipSelect.fallback
## and finally to any action the sheet has; {} when the sheet is empty.
func resolve(action: int, face: int, is_player: bool) -> Dictionary:
	if actions.is_empty():
		return {}
	var a: int = ClipSelect.clip_action(action, is_player)
	var tries: int = 0
	while not actions.has(ClipSelect.NAMES[a]) and tries < 3:
		a = ClipSelect.fallback(a)
		tries += 1
	var action_name: String = ClipSelect.NAMES[a]
	if not actions.has(action_name):
		action_name = String(actions.keys()[0])
	var info: Dictionary = actions[action_name]
	var rf: Vector2i = ClipSelect.sheet_row(face, int(info["rows"]))
	var row: int = mini(rf.x, int(info["rows"]) - 1)
	return {"name": action_name, "anim": StringName("%s_%d" % [action_name, row]), "flip": rf.y == 1,
			"frames": int(info["frames"]), "delay_ms": float(info["delay_ms"])}


func frame_texture(anim: StringName, index: int) -> Texture2D:
	if not frames.has_animation(anim):
		return null
	var n: int = frames.get_frame_count(anim)
	if n == 0:
		return null
	return frames.get_frame_texture(anim, clampi(index, 0, n - 1))
