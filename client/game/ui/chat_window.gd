class_name ChatWindow
extends ChatBox
## RO chat window: ChatBox (same API: bind, set_welcome, on_message, submit, add_line, lines, parse_command...) with
## tabs All / Party / Whisper / System that filter the log. Typing in the Party tab sends to the party without
## `/p`; server notices (`store.notice`) land in System. Swap `chat_box.tscn` for `chat_window.tscn` in GameView.

const TABS: Array[String] = ["All", "Party", "Whisper", "System"]
const TAB_ALL: int = 0
const TAB_PARTY: int = 1
const TAB_WHISPER: int = 2
const TAB_SYSTEM: int = 3

var tab_bar: TabBar
var current_tab: int = TAB_ALL
var unread: Array[int] = [0, 0, 0, 0]
var _notices: NoticeFeed = NoticeFeed.new()


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	theme_type_variation = &"ChatWindow"
	custom_minimum_size = Vector2(420, 170)
	var box := VBoxContainer.new()
	add_child(box)
	tab_bar = TabBar.new()
	tab_bar.focus_mode = Control.FOCUS_NONE
	for t: String in TABS:
		tab_bar.add_tab(t)
	tab_bar.tab_changed.connect(select_tab)
	box.add_child(tab_bar)
	log_view = RichTextLabel.new()
	log_view.bbcode_enabled = false
	log_view.scroll_following = true
	log_view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	box.add_child(log_view)
	input = LineEdit.new()
	input.placeholder_text = "Enter to chat. /p party, /w name text"
	input.max_length = 200
	input.text_submitted.connect(submit)
	box.add_child(input)


func _process(_delta: float) -> void:
	if session == null:
		return
	var n: String = _notices.observe(session.store.snaps, session.store.notice)
	if n != "":
		add_line({"text": n, "color": kind_color("system"), "kind": "system"})


func select_tab(i: int) -> void:
	current_tab = clampi(i, 0, TABS.size() - 1)
	unread[current_tab] = 0
	if tab_bar != null and tab_bar.current_tab != current_tab:
		tab_bar.current_tab = current_tab
	if input != null:
		input.placeholder_text = "Party chat (Enter)" if current_tab == TAB_PARTY else "Enter to chat. /p party, /w name text"
	_render()
	_chat_window_titles()


func add_line(line: Dictionary) -> void:
	lines.append(line)
	if lines.size() > MAX_LINES:
		lines.pop_front()
	for t: int in range(TABS.size()):
		if t != current_tab and t != TAB_ALL and tab_shows(t, String(line.get("kind", "say"))):
			unread[t] += 1
	_render()
	_chat_window_titles()


## The Party tab sends plain text to the party; everything else behaves like ChatBox.
func submit(text: String) -> Dictionary:
	return super.submit(route_text(text, current_tab))


## Lines of `lines` the tab `t` shows.
func visible_lines(t: int) -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	for l: Dictionary in lines:
		if tab_shows(t, String(l.get("kind", "say"))):
			out.append(l)
	return out


func _render() -> void:
	if log_view == null:
		return
	log_view.clear()
	for l: Dictionary in visible_lines(current_tab):
		log_view.push_color(l["color"])
		log_view.add_text(String(l["text"]) + "\n")
		log_view.pop()


func _chat_window_titles() -> void:
	if tab_bar == null:
		return
	for t: int in range(TABS.size()):
		tab_bar.set_tab_title(t, TABS[t] if unread[t] == 0 else "%s (%d)" % [TABS[t], unread[t]])


# ---- pure helpers ----

## True when tab `t` shows a line of `kind` (All shows everything).
static func tab_shows(t: int, kind: String) -> bool:
	match t:
		TAB_PARTY:
			return kind == "party"
		TAB_WHISPER:
			return kind == "whisper"
		TAB_SYSTEM:
			return kind == "system"
	return true


## In the Party tab a line without a leading `/` is a party line.
static func route_text(text: String, t: int) -> String:
	var s: String = text.strip_edges()
	if t == TAB_PARTY and s != "" and not s.begins_with("/"):
		return "/p " + s
	return text
