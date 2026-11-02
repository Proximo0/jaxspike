class_name BuffRow
extends HBoxContainer
## One small label per active buff: "Name Lv seconds", from `you.buffs`, `welcome.buffs` and `tick_ms`.

var session: NetSession = null
var welcome: Dictionary = {}
var icons: Array[Label] = []  ## Pooled; the first `shown` are visible.
var shown: int = 0


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme


func bind(s: NetSession) -> void:
	session = s


func set_welcome(msg: Dictionary) -> void:
	welcome = msg


func _process(_delta: float) -> void:
	refresh()


func refresh() -> void:
	if session == null:
		return
	var rows: Array[Dictionary] = UiStatsCalc.buff_rows(welcome, session.store.you, maxi(session.tick_ms, 1))
	shown = rows.size()
	while icons.size() < shown:
		var l := Label.new()
		add_child(l)
		icons.append(l)
	for i: int in range(icons.size()):
		icons[i].visible = i < shown
		if i < shown:
			var r: Dictionary = rows[i]
			icons[i].text = "%s L%d %ds" % [r["name"], r["level"], ceili(float(r["seconds"]))]
