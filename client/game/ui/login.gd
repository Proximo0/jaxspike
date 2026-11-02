class_name LoginScreen
extends Control
## Name, host and port fields plus Connect. Emits connect_requested(name, url) for valid input; show_error shows a
## failure from the connection attempt.

signal connect_requested(player_name: String, url: String)

@onready var name_edit: LineEdit = $Center/Box/NameEdit
@onready var host_edit: LineEdit = $Center/Box/HostEdit
@onready var port_edit: LineEdit = $Center/Box/PortEdit
@onready var connect_button: Button = $Center/Box/ConnectButton
@onready var error_label: Label = $Center/Box/ErrorLabel


func _ready() -> void:
	host_edit.text = LoginForm.DEFAULT_HOST
	port_edit.text = str(LoginForm.DEFAULT_PORT)
	connect_button.pressed.connect(submit)
	name_edit.text_submitted.connect(_on_name_submitted)
	name_edit.grab_focus()


## Validates the fields and emits connect_requested, or shows the validation error.
func submit() -> void:
	var r: Dictionary = LoginForm.validate(name_edit.text, host_edit.text, port_edit.text)
	if not bool(r["ok"]):
		show_error(String(r["error"]))
		return
	error_label.text = ""
	connect_requested.emit(String(r["name"]), String(r["url"]))


func show_error(msg: String) -> void:
	error_label.text = msg
	connect_button.disabled = false


func set_busy(busy: bool) -> void:
	connect_button.disabled = busy


func _on_name_submitted(_text: String) -> void:
	submit()
