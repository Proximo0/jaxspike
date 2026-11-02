class_name ChatBox
extends PanelContainer
## Chat log plus input line. Enter focuses the line; `/p text` party, `/w name text` whisper, anything else says.
## Shows `chat` messages in kind colours and `mvp` broadcasts as system lines. Feed every polled message to on_message.

signal message_sent(msg: Dictionary)

const FOCUS_KEY: Key = KEY_ENTER
const MAX_LINES: int = 100
const COLORS: Dictionary = {
	"say": Color(1.0, 1.0, 1.0),
	"party": Color(0.45, 0.75, 1.0),
	"whisper": Color(1.0, 0.55, 0.85),
	"system": Color(1.0, 0.9, 0.4),
}

var session: NetSession = null
var welcome: Dictionary = {}
var lines: Array[Dictionary] = []  ## {text, color, kind}
var log_view: RichTextLabel
var input: LineEdit


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	custom_minimum_size = Vector2(380, 150)
	var box := VBoxContainer.new()
	add_child(box)
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


func bind(s: NetSession) -> void:
	session = s


func set_welcome(msg: Dictionary) -> void:
	welcome = msg


## Handles one decoded message: welcome (kept for mvp names), chat and mvp (appended to the log).
func on_message(msg: Dictionary) -> void:
	match String(msg.get("t", "")):
		"welcome":
			set_welcome(msg)
		"chat":
			add_line(format_chat(msg))
		"mvp":
			add_line(format_mvp(msg, welcome))


func _unhandled_key_input(event: InputEvent) -> void:
	var k := event as InputEventKey
	if k != null and k.pressed and not k.echo and (k.keycode == FOCUS_KEY or k.keycode == KEY_KP_ENTER):
		if input != null:
			input.grab_focus()
			get_viewport().set_input_as_handled()


func add_line(line: Dictionary) -> void:
	lines.append(line)
	if lines.size() > MAX_LINES:
		lines.pop_front()
	if log_view != null:
		log_view.clear()
		for l: Dictionary in lines:
			log_view.push_color(l["color"])
			log_view.add_text(String(l["text"]) + "\n")
			log_view.pop()


## Parses and sends the typed line, clears and unfocuses the input. Returns the message sent ({} for empty input).
func submit(text: String) -> Dictionary:
	var msg: Dictionary = parse_command(text)
	if input != null:
		input.text = ""
		input.release_focus()
	if msg.is_empty():
		return {}
	if session != null:
		session.send(msg)
	if String(msg["kind"]) == "whisper":
		add_line({"text": "To %s: %s" % [msg["to"], msg["text"]], "color": COLORS["whisper"], "kind": "whisper"})
	message_sent.emit(msg)
	return msg


# ---- pure helpers ----

## Typed line -> `chat` message, {} when there is nothing to send (empty, or `/w` without name and text).
static func parse_command(raw: String) -> Dictionary:
	var text: String = raw.strip_edges()
	if text == "":
		return {}
	if text.begins_with("/"):
		var parts: PackedStringArray = text.split(" ", false, 1)
		var cmd: String = parts[0].to_lower()
		var rest: String = parts[1].strip_edges() if parts.size() > 1 else ""
		match cmd:
			"/p", "/party":
				if rest == "":
					return {}
				return {"t": "chat", "kind": "party", "text": rest, "to": null}
			"/w", "/whisper":
				var wp: PackedStringArray = rest.split(" ", false, 1)
				if wp.size() < 2 or wp[1].strip_edges() == "":
					return {}
				return {"t": "chat", "kind": "whisper", "text": wp[1].strip_edges(), "to": wp[0]}
			"/s", "/say":
				if rest == "":
					return {}
				return {"t": "chat", "kind": "say", "text": rest, "to": null}
	return {"t": "chat", "kind": "say", "text": text, "to": null}


static func kind_color(kind: String) -> Color:
	return COLORS.get(kind, COLORS["say"])


## Incoming `chat` message -> {text, color, kind}.
static func format_chat(msg: Dictionary) -> Dictionary:
	var kind: String = String(msg.get("kind", "say"))
	var who: String = String(msg.get("from", ""))
	var body: String = String(msg.get("text", ""))
	var text: String = body
	match kind:
		"say":
			text = "%s: %s" % [who, body]
		"party":
			text = "[Party] %s: %s" % [who, body]
		"whisper":
			text = "%s whispers: %s" % [who, body]
	return {"text": text, "color": kind_color(kind), "kind": kind}


## `mvp` broadcast -> a system line, naming the boss from welcome.mvp_kinds.
static func format_mvp(msg: Dictionary, welcome_msg: Dictionary) -> Dictionary:
	var kind: int = int(msg.get("kind", -1))
	var boss: String = "an MVP"
	var names: Variant = welcome_msg.get("mvp_kinds")
	if typeof(names) == TYPE_ARRAY and kind >= 0 and kind < (names as Array).size() and (names as Array)[kind] != null:
		boss = String((names as Array)[kind])
	var killer: Variant = msg.get("killer")
	var text: String = "MVP %s was slain!" % boss
	if killer != null and String(killer) != "":
		text = "%s slew the MVP %s!" % [String(killer), boss]
	return {"text": text, "color": kind_color("system"), "kind": "system"}
