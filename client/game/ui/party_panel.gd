class_name PartyPanel
extends PanelContainer
## Party window (key P toggles it): party id, leader flag, member names and Create / Leave / Invite / Accept buttons
## (docs/party_wire.md). `you.members` is sent only on change, so the last list is cached here.

signal message_sent(msg: Dictionary)

const TOGGLE_KEY: Key = KEY_P
const OPS: Array[String] = ["create", "leave", "invite", "accept"]

var session: NetSession = null
var members: Array[String] = []
var party_id: int = -1
var is_leader: bool = false
var status_label: Label
var members_label: Label
var create_button: Button
var leave_button: Button
var invite_button: Button
var accept_button: Button
var target_edit: LineEdit
var _synced_version: int = -1


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	custom_minimum_size = Vector2(220, 0)
	var box := VBoxContainer.new()
	add_child(box)
	var title := Label.new()
	title.text = "Party (P)"
	box.add_child(title)
	status_label = Label.new()
	box.add_child(status_label)
	members_label = Label.new()
	box.add_child(members_label)
	var row := HBoxContainer.new()
	box.add_child(row)
	create_button = _button(row, "Create", "create")
	leave_button = _button(row, "Leave", "leave")
	accept_button = _button(row, "Accept", "accept")
	target_edit = LineEdit.new()
	target_edit.placeholder_text = "Player name"
	target_edit.max_length = 16
	target_edit.text_submitted.connect(func(_t: String) -> void: invite_target())
	box.add_child(target_edit)
	invite_button = Button.new()
	invite_button.text = "Invite"
	invite_button.pressed.connect(invite_target)
	box.add_child(invite_button)
	visible = false
	refresh()


func _button(parent: Node, text: String, op: String) -> Button:
	var b := Button.new()
	b.text = text
	b.pressed.connect(send_op.bind(op))
	parent.add_child(b)
	return b


func bind(s: NetSession) -> void:
	session = s


func on_message(_msg: Dictionary) -> void:
	pass


func _process(_delta: float) -> void:
	if session != null and session.store.version != _synced_version:
		refresh()


func _unhandled_key_input(event: InputEvent) -> void:
	var k := event as InputEventKey
	if k != null and k.pressed and not k.echo and k.keycode == TOGGLE_KEY:
		visible = not visible
		get_viewport().set_input_as_handled()


func refresh() -> void:
	if not is_node_ready():
		return
	var you: Dictionary = session.store.you if session != null else {}
	_synced_version = session.store.version if session != null else -1
	party_id = int(you.get("party", -1))
	is_leader = bool(you.get("leader", false))
	if you.has("members"):
		members = names_from(you["members"])
	if party_id < 0:
		members = []
	status_label.text = status_text(party_id, is_leader)
	members_label.text = "\n".join(PackedStringArray(members)) if not members.is_empty() else "-"
	create_button.disabled = party_id >= 0
	leave_button.disabled = party_id < 0


func invite_target() -> Dictionary:
	var tname: String = target_edit.text.strip_edges()
	if tname == "":
		return {}
	return send_op("invite", tname)


## Sends one party op and returns the message.
func send_op(op: String, target: String = "") -> Dictionary:
	var msg: Dictionary = party_message(op, target)
	if msg.is_empty():
		return {}
	if session != null:
		session.send(msg)
	message_sent.emit(msg)
	return msg


# ---- pure helpers ----

static func party_message(op: String, target: String = "") -> Dictionary:
	if not OPS.has(op):
		return {}
	var tgt: Variant = target if (op == "invite" and target != "") else null
	return {"t": "party", "op": op, "target": tgt}


static func names_from(v: Variant) -> Array[String]:
	var out: Array[String] = []
	if typeof(v) == TYPE_ARRAY:
		for n: Variant in v as Array:
			out.append(String(n))
	return out


static func status_text(id: int, leader: bool) -> String:
	if id < 0:
		return "No party"
	return "Party %d%s" % [id, " (you lead)" if leader else ""]
