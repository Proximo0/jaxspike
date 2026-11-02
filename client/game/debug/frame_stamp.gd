class_name FrameStamp
extends CanvasLayer
## Debug frame stamp. A dark plate in the bottom-left with a text line
## (frame, state hash, fps, render size, Flags.commands) and a 112-bit code strip, so an agent that cannot
## watch the game can tell which frame and state a screenshot shows. Decode a PNG with tools/frame_code.py.
## Hidden by default; F10 ("debug_stamp") toggles it; always shown when Flags.args has "stamp".

const STAMP_LAYER: int = 128
const CODE_COLS: int = 56
const CODE_ROWS: int = 2
const FINDER_COLS: int = 3
const CELL: int = 8
const PAD: float = 6.0
const GAP: float = 4.0
const MARGIN: float = 8.0
const FONT_SIZE: int = 16
const PLATE_COLOR := Color(0.05, 0.06, 0.09, 0.9)
const TEXT_COLOR := Color(1.0, 1.0, 1.0)
const FINDER_COLOR := Color(1.0, 0.0, 0.0)
const ONE_COLOR := Color(1.0, 1.0, 1.0)
const ZERO_COLOR := Color(0.0, 0.0, 0.0)

var _plate: Control = null
var _text: String = ""
var _code: PackedByteArray = PackedByteArray()
var _ascent: float = 0.0
var _line_height: float = 0.0


func _init() -> void:
	layer = STAMP_LAYER
	visible = false


func _ready() -> void:
	_plate = Control.new()
	_plate.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_plate.draw.connect(_on_plate_draw)
	add_child(_plate)
	if Flags.args.has("stamp"):
		visible = true


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("debug_stamp") and not Flags.args.has("stamp"):
		visible = not visible


func _process(_delta: float) -> void:
	if not visible or _plate == null:
		return
	_refresh()
	_plate.queue_redraw()


func _refresh() -> void:
	var font: Font = ThemeDB.fallback_font
	var state_hash: int = Flags.state_hash()
	var render_size: Vector2i = get_viewport().size
	var fps: int = int(Engine.get_frames_per_second())
	_text = "frame %d  hash %s  fps %d  %dx%d  cmds %d" % [
		Flags.frame, _hex64(state_hash), fps, render_size.x, render_size.y, Flags.commands,
	]
	_code = FrameCode.encode(Flags.frame, state_hash)
	_ascent = font.get_ascent(FONT_SIZE)
	_line_height = font.get_height(FONT_SIZE)
	var text_w: float = font.get_string_size(_text, HORIZONTAL_ALIGNMENT_LEFT, -1, FONT_SIZE).x
	var strip_w: float = float((CODE_COLS + 2 * FINDER_COLS) * CELL)
	var strip_h: float = float(CODE_ROWS * CELL)
	var plate_size := Vector2(
		maxf(text_w, strip_w) + 2.0 * PAD,
		PAD + _line_height + GAP + strip_h + PAD,
	)
	var view: Vector2 = get_viewport().get_visible_rect().size
	_plate.size = plate_size
	_plate.position = Vector2(MARGIN, view.y - plate_size.y - MARGIN)


func _on_plate_draw() -> void:
	if _code.size() != FrameCode.BYTE_COUNT:
		return
	var font: Font = ThemeDB.fallback_font
	_plate.draw_rect(Rect2(Vector2.ZERO, _plate.size), PLATE_COLOR)
	_plate.draw_string(font, Vector2(PAD, PAD + _ascent), _text, HORIZONTAL_ALIGNMENT_LEFT, -1, FONT_SIZE, TEXT_COLOR)
	var origin := Vector2(PAD, PAD + _line_height + GAP)
	var cell: float = float(CELL)
	var right_finder_col: int = FINDER_COLS + CODE_COLS
	for r: int in range(CODE_ROWS):
		var y: float = origin.y + float(r * CELL)
		for c: int in range(FINDER_COLS):
			_plate.draw_rect(Rect2(origin.x + float(c * CELL), y, cell, cell), FINDER_COLOR)
			_plate.draw_rect(Rect2(origin.x + float((right_finder_col + c) * CELL), y, cell, cell), FINDER_COLOR)
		for c: int in range(CODE_COLS):
			var i: int = r * CODE_COLS + c
			var bit: int = (_code[i >> 3] >> (7 - (i & 7))) & 1
			var color: Color = ONE_COLOR if bit == 1 else ZERO_COLOR
			_plate.draw_rect(Rect2(origin.x + float((FINDER_COLS + c) * CELL), y, cell, cell), color)


static func _hex64(v: int) -> String:
	return "%08x%08x" % [(v >> 32) & 0xFFFFFFFF, v & 0xFFFFFFFF]
