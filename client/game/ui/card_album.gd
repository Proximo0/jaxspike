class_name CardAlbum
extends PanelContainer
## Card album (Alt+A; plain A is the WASD walk-left key): every card of `welcome.cards` as a framed portrait from
## res://assets/cards, owned ones lit (inventory, sockets of carried gear, waybell storage), the rest a dark
## silhouette. Hover or click a card for its effect text, socket and where it drops. Also announces the server's
## world-wide `card` message (docs/cards_wire.md): a gold line in the chat window and a short banner with the
## portrait at the top of the screen. The static helpers are pure and unit tested (test_ui_card_album.gd).

const TOGGLE_KEY: Key = KEY_A
const KIND_CARD: int = 5
const CARD_ART_DIR: String = "res://assets/cards"
const GRID_COLS: int = 6
const THUMB_PX: Vector2 = Vector2(72, 96)
const DETAIL_PX: Vector2 = Vector2(192, 256)
const UNOWNED_TINT: Color = Color(0.12, 0.12, 0.18, 1.0)
const ANNOUNCE_COLOR: Color = Color(1.0, 0.82, 0.25)
const MVP_ANNOUNCE_COLOR: Color = Color(1.0, 0.5, 0.95)
const BANNER_MS: float = 5000.0
const BANNER_FADE_MS: float = 800.0
const BUTTON_TEXT: String = "Cards (Alt+A)"
const BUTTON_MARGIN: float = 8.0
const BUTTON_SIZE: Vector2 = Vector2(150, 36)

## item id -> true for every card id of the last welcome (EntityLayer reads it for the drop sparkle).
static var card_ids: Dictionary = {}

var session: NetSession = null
var welcome: Dictionary = {}
var cards: Array = []  ## welcome.cards rows
var rows: Array[Dictionary] = []  ## last card_rows
var selected: int = -1  ## index into rows
var title_label: Label
var title_bar: PanelTitle
var grid: GridContainer
var detail_art: TextureRect
var detail_label: Label
var thumbs: Array[ItemTooltip.TipRect] = []
var banner: PanelContainer = null
var banner_art: TextureRect = null
var banner_label: Label = null
var banner_left_ms: float = 0.0
var hud_button: Button = null  ## ro5-items: the HUD button that toggles the album (bottom right)
var last_announce: Dictionary = {}  ## the last `card` message seen (tests, sims)
var _art_cache: Dictionary = {}  ## path -> Texture2D or null
var _synced_version: int = -1


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	var box := VBoxContainer.new()
	add_child(box)
	title_bar = PanelTitle.make(self, box, "Card Album", close)  # ro6-ui: shared title, close, drag
	title_label = title_bar.title_label
	var body := HBoxContainer.new()
	box.add_child(body)
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size = Vector2((THUMB_PX.x + 6.0) * GRID_COLS + 16.0, 420)
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	body.add_child(scroll)
	grid = GridContainer.new()
	grid.columns = GRID_COLS
	scroll.add_child(grid)
	var side := VBoxContainer.new()
	side.custom_minimum_size = Vector2(DETAIL_PX.x + 8.0, 0)
	body.add_child(side)
	detail_art = TextureRect.new()
	detail_art.custom_minimum_size = DETAIL_PX
	detail_art.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	detail_art.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	side.add_child(detail_art)
	detail_label = Label.new()
	detail_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	detail_label.custom_minimum_size = Vector2(DETAIL_PX.x, 0)
	side.add_child(detail_label)
	visible = false
	call_deferred("_make_banner")
	call_deferred("_make_hud_button")
	refresh()


func bind(s: NetSession) -> void:
	session = s


func set_welcome(msg: Dictionary) -> void:
	welcome = msg
	var v: Variant = msg.get("cards")
	cards = (v as Array).duplicate() if typeof(v) == TYPE_ARRAY else []
	CardAlbum.card_ids = ids_of(cards)
	_synced_version = -1
	_rebuild_grid()
	refresh()


## Every polled message: welcome (the table) and `card` (the announcement).
func on_message(msg: Dictionary) -> void:
	match String(msg.get("t", "")):
		"welcome":
			set_welcome(msg)
		"card":
			announce(msg)


func _process(delta: float) -> void:
	if session != null and session.store.version != _synced_version:
		refresh()
	if banner != null and banner_left_ms > 0.0:
		banner_left_ms = maxf(banner_left_ms - delta * 1000.0, 0.0)
		banner.modulate.a = banner_alpha(banner_left_ms)
		banner.visible = banner_left_ms > 0.0


func _unhandled_key_input(event: InputEvent) -> void:
	var k := event as InputEventKey
	if k != null and k.pressed and not k.echo and k.keycode == TOGGLE_KEY and k.alt_pressed:
		toggle()
		get_viewport().set_input_as_handled()


## Hides the album (the title bar's close button).
func close() -> void:
	visible = false


## Shows or hides the album (Alt+A and the HUD button).
func toggle() -> void:
	visible = not visible
	if visible:
		refresh()


## ro5-items: a "Cards" button in the bottom right corner of the panel layer, so the album needs no key.
func _make_hud_button() -> void:
	var parent: Node = get_parent()
	if parent == null or hud_button != null:
		return
	hud_button = Button.new()
	hud_button.name = "CardAlbumButton"
	hud_button.theme = theme
	hud_button.text = BUTTON_TEXT
	hud_button.focus_mode = Control.FOCUS_NONE
	hud_button.icon = card_texture("card.gildscarab")
	hud_button.expand_icon = true
	hud_button.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	hud_button.offset_left = -BUTTON_SIZE.x - BUTTON_MARGIN
	hud_button.offset_top = -BUTTON_SIZE.y - BUTTON_MARGIN
	hud_button.offset_right = -BUTTON_MARGIN
	hud_button.offset_bottom = -BUTTON_MARGIN
	hud_button.tooltip_text = "Card album: every card in Aurelune and the ones you own"
	hud_button.pressed.connect(toggle)
	parent.add_child(hud_button)


## Recomputes owned counts from the session's `you` and relights the grid.
func refresh() -> void:
	if not is_node_ready():
		return
	var you: Dictionary = session.store.you if session != null else {}
	_synced_version = session.store.version if session != null else -1
	rows = card_rows(cards, you)
	if thumbs.size() != rows.size():
		_rebuild_grid()
	var n_owned: int = 0
	for i: int in range(rows.size()):
		var lit: bool = int(rows[i]["owned"]) > 0
		n_owned += 1 if lit else 0
		thumbs[i].modulate = Color.WHITE if lit else UNOWNED_TINT
		thumbs[i].set_tip(detail_tip(rows[i], welcome))
	title_label.text = album_title(n_owned, rows.size())
	if selected < 0 and not rows.is_empty():
		selected = 0
	_show_detail()


func select(i: int) -> void:
	if i >= 0 and i < rows.size():
		selected = i
		_show_detail()


## Shows the announcement: a chat line in the ChatBox sibling (if any) and the banner. Returns the chat line.
func announce(msg: Dictionary) -> Dictionary:
	last_announce = msg
	var line: Dictionary = announce_line(msg)
	var parent: Node = get_parent()
	if parent != null:
		for n: Node in parent.get_children():
			if n is ChatBox:
				(n as ChatBox).add_line(line)
	if banner != null:
		banner_label.text = String(line["text"])
		banner_label.add_theme_color_override("font_color", line["color"])
		var row: Dictionary = row_of(cards, int(msg.get("item", 0)))
		banner_art.texture = card_texture(String(row.get("icon", "")))
		banner_left_ms = BANNER_MS
		banner.modulate.a = 1.0
		banner.visible = true
	return line


func _make_banner() -> void:
	var parent: Node = get_parent()
	if parent == null or banner != null:
		return
	banner = PanelContainer.new()
	banner.name = "CardBanner"
	banner.theme = theme
	banner.mouse_filter = Control.MOUSE_FILTER_IGNORE
	banner.set_anchors_preset(Control.PRESET_CENTER_TOP)
	banner.offset_top = 90.0
	banner.offset_left = -260.0
	banner.offset_right = 260.0
	var hb := HBoxContainer.new()
	hb.alignment = BoxContainer.ALIGNMENT_CENTER
	banner.add_child(hb)
	banner_art = TextureRect.new()
	banner_art.custom_minimum_size = THUMB_PX
	banner_art.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	banner_art.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	hb.add_child(banner_art)
	banner_label = Label.new()
	banner_label.add_theme_font_size_override("font_size", 26)
	banner_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	hb.add_child(banner_label)
	banner.visible = false
	parent.add_child(banner)


func _rebuild_grid() -> void:
	if grid == null:
		return
	for c: Node in grid.get_children():
		grid.remove_child(c)
		c.queue_free()
	thumbs.clear()
	var rs: Array[Dictionary] = card_rows(cards, {})
	for i: int in range(rs.size()):
		var t := ItemTooltip.TipRect.new()
		t.custom_minimum_size = THUMB_PX
		t.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		t.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
		t.texture = card_texture(String(rs[i]["icon"]))
		t.mouse_filter = Control.MOUSE_FILTER_STOP
		t.gui_input.connect(_on_thumb_input.bind(i))
		t.mouse_entered.connect(select.bind(i))
		grid.add_child(t)
		thumbs.append(t)


func _on_thumb_input(event: InputEvent, i: int) -> void:
	var mb := event as InputEventMouseButton
	if mb != null and mb.pressed and mb.button_index == MOUSE_BUTTON_LEFT:
		select(i)


func _show_detail() -> void:
	if detail_label == null:
		return
	if selected < 0 or selected >= rows.size():
		detail_label.text = "No cards yet."
		detail_art.texture = null
		return
	var r: Dictionary = rows[selected]
	detail_art.texture = card_texture(String(r["icon"]))
	detail_art.modulate = Color.WHITE if int(r["owned"]) > 0 else UNOWNED_TINT
	detail_label.text = detail_text(r, welcome)


## Texture of a card icon key (`card.<snake>`), cached; null when the art file is missing.
func card_texture(icon: String) -> Texture2D:
	var path: String = art_path(icon)
	if path == "":
		return null
	if _art_cache.has(path):
		return _art_cache[path] as Texture2D
	var tex: Texture2D = null
	if ResourceLoader.exists(path):
		tex = load(path) as Texture2D
	elif FileAccess.file_exists(path):
		var img: Image = Image.load_from_file(ProjectSettings.globalize_path(path))
		if img != null and not img.is_empty():
			tex = ImageTexture.create_from_image(img)
	_art_cache[path] = tex
	return tex


# ---- pure helpers ----

## {id: true} for every row of welcome.cards.
static func ids_of(card_table: Array) -> Dictionary:
	var out: Dictionary = {}
	for r: Variant in card_table:
		if typeof(r) == TYPE_DICTIONARY:
			out[int((r as Dictionary).get("id", 0))] = true
	return out


## True when `item` is a card id of the last welcome (false before any welcome).
static func is_card(item: int) -> bool:
	return CardAlbum.card_ids.has(item)


## res:// path of a card icon key: `card.blobbin` -> res://assets/cards/blobbin.png ("" for a bad key).
static func art_path(icon: String) -> String:
	if not icon.begins_with("card.") or icon.length() <= 5:
		return ""
	return "%s/%s.png" % [CARD_ART_DIR, icon.substr(5)]


## The welcome.cards row of an item id ({} when none).
static func row_of(card_table: Array, item: int) -> Dictionary:
	for r: Variant in card_table:
		if typeof(r) == TYPE_DICTIONARY and int((r as Dictionary).get("id", 0)) == item:
			return r
	return {}


## Card id -> units the player holds: carried card stacks, cards socketed in carried gear, waybell storage ids.
static func owned_counts(you: Dictionary, ids: Dictionary) -> Dictionary:
	var out: Dictionary = {}
	var inv: Variant = you.get("inv")
	if typeof(inv) == TYPE_ARRAY:
		for raw: Variant in inv as Array:
			if typeof(raw) != TYPE_DICTIONARY:
				continue
			var d: Dictionary = raw
			var item: int = int(d.get("item", 0))
			if ids.has(item):
				out[item] = int(out.get(item, 0)) + maxi(int(d.get("count", 1)), 1)
			var cs: Variant = d.get("cards")
			if typeof(cs) == TYPE_ARRAY:
				for c: Variant in cs as Array:
					if ids.has(int(c)):
						out[int(c)] = int(out.get(int(c), 0)) + 1
	var st: Variant = you.get("storage")
	if typeof(st) == TYPE_ARRAY:
		for raw: Variant in st as Array:
			if ids.has(int(raw)):
				out[int(raw)] = int(out.get(int(raw), 0)) + 1
	return out


## One row per card (welcome order): id, name, monster, text, where, slot_name, chance, mvp, icon, owned.
static func card_rows(card_table: Array, you: Dictionary) -> Array[Dictionary]:
	var owned: Dictionary = owned_counts(you, ids_of(card_table))
	var out: Array[Dictionary] = []
	for r: Variant in card_table:
		if typeof(r) != TYPE_DICTIONARY:
			continue
		var d: Dictionary = r
		var id: int = int(d.get("id", 0))
		out.append({
			"id": id, "name": String(d.get("name", "")), "monster": String(d.get("monster", "")),
			"text": String(d.get("text", "")), "where": String(d.get("where", "")),
			"slot_name": String(d.get("slot_name", "")), "chance": int(d.get("chance", 0)),
			"mvp": bool(d.get("mvp", false)), "icon": String(d.get("icon", "")), "owned": int(owned.get(id, 0)),
		})
	return out


static func album_title(n_owned: int, n_all: int) -> String:
	return "Card Album (Alt+A)   %d / %d found" % [n_owned, n_all]


## '1 in 10,000' for a per-10000 chance ("" for 0).
static func chance_text(per_10000: int) -> String:
	return ItemText.chance_text(per_10000)


## The hover / detail text of a card row: the shared item tooltip (ItemText) plus the owned line.
static func detail_text(row: Dictionary, welcome_msg: Dictionary = {}) -> String:
	var n: int = int(row.get("owned", 0))
	var owned: String = "Owned: %d" % n if n > 0 else "Not found yet"
	var id: int = int(row.get("id", 0))
	if not ItemText.info(welcome_msg, id).is_empty():
		return ItemText.tooltip(welcome_msg, id, 0, [], PackedStringArray([owned]))
	var lines: PackedStringArray = PackedStringArray()
	lines.append(String(row.get("name", "")))
	lines.append("Card")
	lines.append_array(ItemText.card_lines(row))
	lines.append(owned)
	return "\n".join(lines)


## ro6-ui: the rich tooltip spec of a card row (ItemText.tooltip_spec plus the owned line); its `text` is
## detail_text. Without the welcome.items row: {"text": detail_text} only, so the plain tooltip shows.
static func detail_tip(row: Dictionary, welcome_msg: Dictionary = {}) -> Dictionary:
	var id: int = int(row.get("id", 0))
	if ItemText.info(welcome_msg, id).is_empty():
		return {"text": detail_text(row, welcome_msg)}
	var n: int = int(row.get("owned", 0))
	return ItemText.tooltip_spec(welcome_msg, id, 0, [], PackedStringArray(["Owned: %d" % n if n > 0 else "Not found yet"]))


## The chat line for a `card` message: {text, color, kind "system"}.
static func announce_line(msg: Dictionary) -> Dictionary:
	var text: String = String(msg.get("text", ""))
	if text == "":
		var who: Variant = msg.get("finder")
		text = "%s found a card!" % (String(who) if typeof(who) == TYPE_STRING else "Someone")
	var mvp: bool = bool(msg.get("mvp", false))
	return {"text": text, "color": MVP_ANNOUNCE_COLOR if mvp else ANNOUNCE_COLOR, "kind": "system"}


## Banner opacity for the remaining time: 1 until the last BANNER_FADE_MS, then a linear fade to 0.
static func banner_alpha(left_ms: float) -> float:
	return clampf(left_ms / BANNER_FADE_MS, 0.0, 1.0)
