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
const HOTKEY_BG := Color(0.24, 0.27, 0.36)
const CLOSE_HOVER := Color(0.70, 0.25, 0.25)
const TIP_BG := Color(0.10, 0.11, 0.16, 0.97)
const TIP_DIM := Color(0.70, 0.73, 0.80)
const TIP_BONUS := Color(0.55, 0.95, 0.55)
const TIP_EXTRA := Color(0.95, 0.90, 0.70)
const ZONE_TITLE_PX: int = 34
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
	add_feel(t)
	add_windows(t)
	return t


## ro6-ui: window chrome (title bar text, close button) and the rich item tooltip (popup frame, label tones).
static func add_windows(t: Theme) -> void:
	var bar: StyleBoxFlat = box(TITLE_BG, PANEL_BORDER)
	bar.content_margin_left = 6.0
	bar.content_margin_right = 2.0
	t.set_stylebox("panel", "TitleBar", bar)
	t.set_type_variation("PanelTitle", "Label")
	t.set_color("font_color", "PanelTitle", Color.WHITE)
	t.set_color("font_outline_color", "PanelTitle", Color(0, 0, 0, 0.7))
	t.set_constant("outline_size", "PanelTitle", 2)
	t.set_font_size("font_size", "PanelTitle", 13)
	t.set_type_variation("CloseButton", "Button")
	var close: StyleBoxFlat = box(BUTTON_PRESSED, PANEL_BORDER)
	close.content_margin_left = 0.0
	close.content_margin_right = 0.0
	close.content_margin_top = 0.0
	close.content_margin_bottom = 0.0
	t.set_stylebox("normal", "CloseButton", close)
	var close_hover: StyleBoxFlat = close.duplicate() as StyleBoxFlat
	close_hover.bg_color = CLOSE_HOVER
	t.set_stylebox("hover", "CloseButton", close_hover)
	var close_pressed: StyleBoxFlat = close.duplicate() as StyleBoxFlat
	close_pressed.bg_color = CLOSE_HOVER.darkened(0.3)
	t.set_stylebox("pressed", "CloseButton", close_pressed)
	t.set_font_size("font_size", "CloseButton", 12)
	var tip: StyleBoxFlat = box(TIP_BG, EXP.darkened(0.25))
	tip.content_margin_left = 8.0
	tip.content_margin_right = 8.0
	tip.content_margin_top = 6.0
	tip.content_margin_bottom = 6.0
	t.set_stylebox("panel", "TooltipPanel", tip)
	t.set_color("font_color", "TooltipLabel", TEXT)
	t.set_font_size("font_size", "TooltipLabel", FONT_SIZE)
	t.set_type_variation("NoticeLine", "Label")
	t.set_color("font_color", "NoticeLine", EXP.lightened(0.3))
	t.set_color("font_outline_color", "NoticeLine", Color(0.05, 0.04, 0.08, 0.95))
	t.set_constant("outline_size", "NoticeLine", 5)
	t.set_font_size("font_size", "NoticeLine", 16)
	var tones: Dictionary = {"TipName": Color.WHITE, "TipDim": TIP_DIM, "TipBonus": TIP_BONUS, "TipCard": EXP,
			"TipExtra": TIP_EXTRA}
	for v: String in tones:
		t.set_type_variation(v, "Label")
		t.set_color("font_color", v, tones[v])
		t.set_font_size("font_size", v, 14 if v == "TipName" else FONT_SIZE)


## Team client-feel variations: basic info window, bar numbers, zone banner, minimap frame, chat window and tabs,
## RO hotkey slots (bevelled square slots on a darker strip).
static func add_feel(t: Theme) -> void:
	var info: StyleBoxFlat = box(PANEL_BG, PANEL_BORDER)
	info.content_margin_left = 6.0
	info.content_margin_right = 6.0
	info.content_margin_top = 4.0
	info.content_margin_bottom = 4.0
	t.set_type_variation("InfoWindow", "PanelContainer")
	t.set_stylebox("panel", "InfoWindow", info)
	t.set_type_variation("InfoName", "Label")
	t.set_color("font_color", "InfoName", Color.WHITE)
	t.set_font_size("font_size", "InfoName", 14)
	t.set_type_variation("InfoBadge", "Label")
	t.set_color("font_color", "InfoBadge", HP.lightened(0.3))
	t.set_type_variation("BarText", "Label")
	t.set_color("font_color", "BarText", Color.WHITE)
	t.set_color("font_outline_color", "BarText", Color(0, 0, 0, 0.9))
	t.set_constant("outline_size", "BarText", 3)
	t.set_font_size("font_size", "BarText", 10)
	t.set_type_variation("ZoneTitle", "Label")
	t.set_font_size("font_size", "ZoneTitle", ZONE_TITLE_PX)
	t.set_color("font_color", "ZoneTitle", Color.WHITE)
	t.set_color("font_outline_color", "ZoneTitle", Color(0.08, 0.06, 0.12))
	t.set_constant("outline_size", "ZoneTitle", 8)
	t.set_type_variation("ZoneSubtitle", "Label")
	t.set_font_size("font_size", "ZoneSubtitle", 14)
	t.set_color("font_color", "ZoneSubtitle", TEXT)
	t.set_color("font_outline_color", "ZoneSubtitle", Color(0.08, 0.06, 0.12))
	t.set_constant("outline_size", "ZoneSubtitle", 4)
	t.set_type_variation("MinimapFrame", "PanelContainer")
	t.set_stylebox("panel", "MinimapFrame", box(PANEL_BG, PANEL_BORDER))
	t.set_type_variation("ChatWindow", "PanelContainer")
	var chat: StyleBoxFlat = box(Color(PANEL_BG, 0.82), PANEL_BORDER)
	t.set_stylebox("panel", "ChatWindow", chat)
	# TabBar (chat tabs): the selected tab takes the title colour.
	t.set_stylebox("tab_selected", "TabBar", box(TITLE_BG, EXP))
	t.set_stylebox("tab_unselected", "TabBar", box(BUTTON_PRESSED, PANEL_BORDER))
	t.set_stylebox("tab_hovered", "TabBar", box(BUTTON_HOVER, PANEL_BORDER))
	t.set_stylebox("tab_disabled", "TabBar", box(BUTTON_DISABLED, TEXT_DISABLED))
	t.set_stylebox("tab_focus", "TabBar", StyleBoxEmpty.new())
	t.set_color("font_selected_color", "TabBar", Color.WHITE)
	t.set_color("font_unselected_color", "TabBar", TEXT_DISABLED.lightened(0.2))
	t.set_color("font_hovered_color", "TabBar", TEXT)
	t.set_font_size("font_size", "TabBar", FONT_SIZE)
	# Hotkey bar: dark strip, square bevelled slots with a gold rim when pressed.
	t.set_type_variation("HotkeyBar", "PanelContainer")
	t.set_stylebox("panel", "HotkeyBar", box(BAR_BG.lightened(0.05), PANEL_BORDER))
	t.set_type_variation("HotkeySlot", "Button")
	var slot: StyleBoxFlat = box(HOTKEY_BG, HOTKEY_BG.lightened(0.35))
	slot.border_width_bottom = 2
	slot.border_width_right = 2
	slot.border_color = HOTKEY_BG.lightened(0.35)
	t.set_stylebox("normal", "HotkeySlot", slot)
	var hover: StyleBoxFlat = slot.duplicate() as StyleBoxFlat
	hover.bg_color = HOTKEY_BG.lightened(0.15)
	t.set_stylebox("hover", "HotkeySlot", hover)
	var pressed: StyleBoxFlat = box(HOTKEY_BG.darkened(0.3), EXP)
	pressed.set_border_width_all(2)
	t.set_stylebox("pressed", "HotkeySlot", pressed)
	t.set_stylebox("disabled", "HotkeySlot", box(BUTTON_DISABLED.darkened(0.2), TEXT_DISABLED))
	t.set_font_size("font_size", "HotkeySlot", 10)
	t.set_color("font_outline_color", "HotkeySlot", Color(0, 0, 0, 0.9))
	t.set_constant("outline_size", "HotkeySlot", 2)


static func save(path: String) -> int:
	return ResourceSaver.save(build(), path)
