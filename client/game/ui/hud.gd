class_name Hud
extends CanvasLayer
## HP bar, kills/deaths, inventory ids, notice line and the safe-zone badge, read from the session's store.

@onready var hp_bar: ProgressBar = $Panel/Box/HpBar
@onready var hp_label: Label = $Panel/Box/HpLabel
@onready var score_label: Label = $Panel/Box/ScoreLabel
@onready var inv_label: Label = $Panel/Box/InvLabel
@onready var notice_label: Label = $Notice
@onready var safe_badge: Label = $Panel/Box/SafeBadge

var session: NetSession = null


func bind(s: NetSession) -> void:
	session = s


func _process(_delta: float) -> void:
	if session == null:
		return
	var you: Dictionary = session.store.you
	var max_hp: int = int(you.get("max_hp", 0))
	hp_bar.max_value = maxf(float(max_hp), 1.0)
	hp_bar.value = float(int(you.get("hp", 0)))
	hp_label.text = HudText.hp_text(you)
	score_label.text = HudText.score_text(you)
	inv_label.text = HudText.inv_text(you)
	notice_label.text = session.store.notice
	safe_badge.visible = bool(you.get("safe", false))
