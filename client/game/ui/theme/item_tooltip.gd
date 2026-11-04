class_name ItemTooltip
extends RefCounted
## ro6-ui: the one rich item tooltip, shared by the inventory, the shop and the card album. It draws an
## ItemText.tooltip_spec: the item icon (or the card portrait), the name with a coloured "+N" refine, the type line,
## the gear bonus in green, each socketed card with its portrait and effect, empty sockets, the card's own effect
## and drop lines, the weight and the caller's extra lines (price, owned, compound). The tooltip popup takes its
## frame from the theme's TooltipPanel. Hosts: the TipButton / TipRect / TipLabel inner classes return build(tip)
## from `_make_custom_tooltip`; their `tooltip_text` holds the plain text (fallback and tests).

const ICON_PX: Vector2 = Vector2(40, 40)
const CARD_HEAD_PX: Vector2 = Vector2(45, 60)
const CARD_ART_PX: Vector2 = Vector2(24, 32)
const SOCKET_PX: Vector2 = Vector2(10, 13)
const SOCKET_COLOR: Color = Color(0.1, 0.08, 0.12, 0.9)
const SOCKET_RIM: Color = Color(0.62, 0.68, 0.80)
const WRAP_W: float = 260.0
const SEPARATION: int = 2

static var _slots: AssetSlots = null
static var _art: Dictionary = {}  ## card art path -> Texture2D or null


## The tooltip control for a spec (no item id gives null, so Godot falls back to the plain tooltip_text).
static func build(spec: Dictionary) -> Control:
	if int(spec.get("id", 0)) <= 0:
		return null
	var box := VBoxContainer.new()
	box.name = "ItemTooltip"
	box.add_theme_constant_override("separation", SEPARATION)
	var head := HBoxContainer.new()
	head.add_theme_constant_override("separation", 6)
	box.add_child(head)
	var art: String = String(spec.get("card_art", ""))
	var tex: Texture2D = card_texture(art) if art != "" else null
	var icon := TextureRect.new()
	icon.name = "Icon"
	icon.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	icon.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	icon.custom_minimum_size = CARD_HEAD_PX if tex != null else ICON_PX
	icon.texture = tex if tex != null else item_texture(String(spec.get("icon_key", "")))
	head.add_child(icon)
	var titles := VBoxContainer.new()
	titles.add_theme_constant_override("separation", 0)
	titles.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	head.add_child(titles)
	var name_row := HBoxContainer.new()
	name_row.add_theme_constant_override("separation", 4)
	titles.add_child(name_row)
	var ref: String = String(spec.get("refine_text", ""))
	if ref != "":
		var rl: Label = _tip_label(ref, &"TipName")
		rl.name = "Refine"
		rl.add_theme_color_override("font_color", spec.get("refine_color", Color.WHITE) as Color)
		name_row.add_child(rl)
	var nl: Label = _tip_label(String(spec.get("name", "")), &"TipName")
	nl.name = "Name"
	name_row.add_child(nl)
	titles.add_child(_tip_label(String(spec.get("type_line", "")), &"TipDim"))
	var bonus: String = String(spec.get("bonus", ""))
	if bonus != "":
		box.add_child(_tip_label("Bonus: " + bonus, &"TipBonus"))
	for c: Variant in spec.get("cards", []) as Array:
		box.add_child(_card_row(c as Dictionary))
	var empty: int = int(spec.get("empty_sockets", 0))
	if empty > 0:
		box.add_child(_socket_row(empty))
	var cl: PackedStringArray = spec.get("card_lines", PackedStringArray()) as PackedStringArray
	for i: int in range(cl.size()):
		var l: Label = _tip_label(cl[i], &"TipBonus" if i == 0 else &"TipDim")
		l.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		l.custom_minimum_size = Vector2(WRAP_W, 0)
		box.add_child(l)
	var w: int = int(spec.get("weight", 0))
	if w > 0:
		box.add_child(_tip_label("Weight %d" % w, &"TipDim"))
	for e: String in spec.get("extra", PackedStringArray()) as PackedStringArray:
		box.add_child(_tip_label(e, &"TipExtra"))
	return box


## Every label text of a built tooltip in tree order (tests and captures read it).
static func texts(c: Node) -> PackedStringArray:
	var out: PackedStringArray = []
	if c is Label:
		out.append((c as Label).text)
	for ch: Node in c.get_children():
		out.append_array(texts(ch))
	return out


static func _tip_label(text: String, variation: StringName) -> Label:
	var l := Label.new()
	l.text = text
	l.theme_type_variation = variation
	return l


static func _card_row(c: Dictionary) -> Control:
	var row := HBoxContainer.new()
	row.name = "Card"
	row.add_theme_constant_override("separation", 6)
	var art := TextureRect.new()
	art.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	art.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	art.custom_minimum_size = CARD_ART_PX
	art.texture = card_texture(String(c.get("art", "")))
	row.add_child(art)
	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 0)
	col.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	row.add_child(col)
	col.add_child(_tip_label(String(c.get("name", "")), &"TipCard"))
	var eff: String = String(c.get("text", ""))
	if eff != "":
		col.add_child(_tip_label(eff, &"TipBonus"))
	return row


static func _socket_row(n: int) -> Control:
	var row := HBoxContainer.new()
	row.name = "Sockets"
	row.add_theme_constant_override("separation", 3)
	for i: int in range(n):
		var p := Panel.new()
		p.custom_minimum_size = SOCKET_PX
		var sb := StyleBoxFlat.new()
		sb.bg_color = SOCKET_COLOR
		sb.border_color = SOCKET_RIM
		sb.set_border_width_all(1)
		p.add_theme_stylebox_override("panel", sb)
		p.size_flags_vertical = Control.SIZE_SHRINK_CENTER
		row.add_child(p)
	row.add_child(_tip_label("%d empty socket%s" % [n, "" if n == 1 else "s"], &"TipDim"))
	return row


## The 64 px item icon of a slots.json key (AssetSlots, loaded once).
static func item_texture(key: String) -> Texture2D:
	if key == "":
		return null
	if _slots == null:
		_slots = AssetSlots.load_default()
	return _slots.item_texture(key) if _slots != null else null


## The portrait of a card icon key (card.<snake>), cached; null when the art is missing.
static func card_texture(icon: String) -> Texture2D:
	var path: String = CardAlbum.art_path(icon)
	if path == "":
		return null
	if _art.has(path):
		return _art[path] as Texture2D
	var tex: Texture2D = null
	if ResourceLoader.exists(path):
		tex = load(path) as Texture2D
	elif FileAccess.file_exists(path):
		var img: Image = Image.load_from_file(ProjectSettings.globalize_path(path))
		if img != null and not img.is_empty():
			tex = ImageTexture.create_from_image(img)
	_art[path] = tex
	return tex


## A Button whose tooltip is the rich item tooltip of `tip` (an ItemText.tooltip_spec).
class TipButton extends Button:
	var tip: Dictionary = {}

	func set_tip(spec: Dictionary) -> void:
		tip = spec
		tooltip_text = String(spec.get("text", ""))

	func _make_custom_tooltip(_for_text: String) -> Object:
		return ItemTooltip.build(tip)


## A TextureRect with the rich item tooltip (shop icons, album thumbs).
class TipRect extends TextureRect:
	var tip: Dictionary = {}

	func set_tip(spec: Dictionary) -> void:
		tip = spec
		tooltip_text = String(spec.get("text", ""))

	func _make_custom_tooltip(_for_text: String) -> Object:
		return ItemTooltip.build(tip)


## A Label with the rich item tooltip (shop row names).
class TipLabel extends Label:
	var tip: Dictionary = {}

	func set_tip(spec: Dictionary) -> void:
		tip = spec
		tooltip_text = String(spec.get("text", ""))

	func _make_custom_tooltip(_for_text: String) -> Object:
		return ItemTooltip.build(tip)
