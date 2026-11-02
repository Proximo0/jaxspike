class_name RoThemeBuilder
extends RefCounted
## Builds the Ragnarok-style UI Theme in code (flat dark blue-grey panels, light borders, 12 px text).
## build_theme.gd saves it to ro_theme.tres; the .tres is the artefact the game loads.

const PANEL_BG := Color(0.16, 0.19, 0.26, 0.96)
const PANEL_BORDER := Color(0.62, 0.68, 0.80)
const TITLE_BG := Color(0.22, 0.28, 0.40)
const BUTTON_BG := Color(0.28, 0.33, 0.45)
const BUTTON_HOVER := Color(0.36, 0.42, 0.57)
const BUTTON_PRESSED := Color(0.18, 0.22, 0.32)
const BUTTON_DISABLED := Color(0.20, 0.22, 0.28)
const TEXT := Color(0.92, 0.93, 0.96)
const TEXT_DISABLED := Color(0.50, 0.52, 0.58)
const HP := Color(0.20, 0.75, 0.30)
const SP := Color(0.25, 0.45, 0.95)
const EXP := Color(0.95, 0.80, 0.25)
const BAR_BG := Color(0.08, 0.09, 0.12)
const CLEAR := Color(0.0, 0.0, 0.0, 0.0)
const FONT_SIZE: int = 12
const MARGIN_X: float = 4.0
const MARGIN_Y: float = 2.0
const RADIUS: int = 2


static func box(bg: Color, border: Color) -> StyleBoxFlat:
	var s := StyleBoxFlat.new()
	s.bg_color = bg
	s.border_color = border
	s.set_border_width_all(1)
	s.set_corner_radius_all(RADIUS)
	s.content_margin_left = MARGIN_X
	s.content_margin_right = MARGIN_X
	s.content_margin_top = MARGIN_Y
	s.content_margin_bottom = MARGIN_Y
	return s


static func build() -> Theme:
	var t := Theme.new()
	t.default_font_size = FONT_SIZE
	var panel: StyleBoxFlat = box(PANEL_BG, PANEL_BORDER)
	t.set_stylebox("panel", "Panel", panel)
	t.set_stylebox("panel", "PanelContainer", box(PANEL_BG, PANEL_BORDER))
	t.set_type_variation("TitleBar", "Panel")
	t.set_stylebox("panel", "TitleBar", box(TITLE_BG, PANEL_BORDER))
	t.set_color("font_color", "Label", TEXT)
	t.set_font_size("font_size", "Label", FONT_SIZE)
	# Buttons
	t.set_stylebox("normal", "Button", box(BUTTON_BG, PANEL_BORDER))
	t.set_stylebox("hover", "Button", box(BUTTON_HOVER, PANEL_BORDER))
	t.set_stylebox("pressed", "Button", box(BUTTON_PRESSED, PANEL_BORDER))
	t.set_stylebox("disabled", "Button", box(BUTTON_DISABLED, TEXT_DISABLED))
	var focus: StyleBoxFlat = box(CLEAR, EXP)
	focus.set_border_width_all(2)
	t.set_stylebox("focus", "Button", focus)
	t.set_color("font_color", "Button", TEXT)
	t.set_color("font_hover_color", "Button", Color.WHITE)
	t.set_color("font_pressed_color", "Button", TEXT)
	t.set_color("font_focus_color", "Button", TEXT)
	t.set_color("font_disabled_color", "Button", TEXT_DISABLED)
	t.set_font_size("font_size", "Button", FONT_SIZE)
	# LineEdit
	t.set_stylebox("normal", "LineEdit", box(BAR_BG, PANEL_BORDER))
	t.set_stylebox("focus", "LineEdit", box(BAR_BG, EXP))
	t.set_stylebox("read_only", "LineEdit", box(BUTTON_DISABLED, TEXT_DISABLED))
	t.set_color("font_color", "LineEdit", TEXT)
	t.set_color("caret_color", "LineEdit", TEXT)
	t.set_color("font_placeholder_color", "LineEdit", TEXT_DISABLED)
	t.set_font_size("font_size", "LineEdit", FONT_SIZE)
	# ItemList
	t.set_stylebox("panel", "ItemList", box(BAR_BG, PANEL_BORDER))
	t.set_stylebox("focus", "ItemList", box(CLEAR, EXP))
	t.set_stylebox("selected", "ItemList", box(BUTTON_BG, PANEL_BORDER))
	t.set_stylebox("selected_focus", "ItemList", box(BUTTON_HOVER, EXP))
	t.set_stylebox("hovered", "ItemList", box(BUTTON_PRESSED, PANEL_BORDER))
	t.set_stylebox("cursor", "ItemList", box(CLEAR, EXP))
	t.set_color("font_color", "ItemList", TEXT)
	t.set_color("font_selected_color", "ItemList", Color.WHITE)
	t.set_font_size("font_size", "ItemList", FONT_SIZE)
	# ScrollContainer
	t.set_stylebox("panel", "ScrollContainer", StyleBoxEmpty.new())
	# ProgressBar and the three bar variations
	t.set_stylebox("background", "ProgressBar", box(BAR_BG, PANEL_BORDER))
	t.set_stylebox("fill", "ProgressBar", box(HP, HP.darkened(0.4)))
	t.set_color("font_color", "ProgressBar", TEXT)
	t.set_font_size("font_size", "ProgressBar", FONT_SIZE)
	var bars: Dictionary = {"HpBar": HP, "SpBar": SP, "ExpBar": EXP}
	for variation: String in bars:
		var c: Color = bars[variation]
		t.set_type_variation(variation, "ProgressBar")
		t.set_stylebox("background", variation, box(BAR_BG, PANEL_BORDER))
		t.set_stylebox("fill", variation, box(c, c.darkened(0.4)))
	return t


static func save(path: String) -> int:
	return ResourceSaver.save(build(), path)
