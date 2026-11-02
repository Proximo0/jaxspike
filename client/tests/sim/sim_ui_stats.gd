extends "res://tests/sim/sim_base.gd"
## RO character UI over FakeLink: panels populate from a hand-written welcome + snap.you, and the `+` / job / skill /
## learn controls send the right messages.


static func welcome_dict() -> Dictionary:
	return {"t": "welcome", "slot": 0, "tick_ms": 120,
		"stat_names": ["str", "agi", "vit", "int", "dex", "luk"],
		"job_names": ["Novice", "Swordsman", "Mage", "Archer", "Acolyte", "Thief", "Merchant"],
		"stat_cost": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
		"buffs": ["", "Blessing", "Increase AGI"],
		"skills": {"names": ["", "Bash", "Heal"], "jobs": [], "list": [null,
			{"id": 1, "name": "Bash", "job": 1, "max_level": 10, "effect": 1, "target": 2, "element": 0, "aoe": 0, "hits": 1,
				"buff": 0, "levels": [{"sp": 8, "cast": 0, "cooldown": 10, "range": 1, "power": 1300, "buff_ticks": 0}]},
			{"id": 2, "name": "Heal", "job": 0, "max_level": 3, "effect": 3, "target": 0, "element": 0, "aoe": 0, "hits": 1,
				"buff": 0, "levels": [{"sp": 5, "cast": 8, "cooldown": 0, "range": 0, "power": 0, "buff_ticks": 0}]},
		]}}


static func snap_text(you: Dictionary) -> String:
	var base: Dictionary = {"pos": [100, 100], "hp": 150, "max_hp": 200, "kills": 0, "deaths": 0, "inv": [], "town": 0,
			"safe": false}
	base.merge(you, true)
	var ents: Array = [
		{"id": 0, "kind": 0, "pos": [100, 100], "hp": 150, "max_hp": 200, "name": "Odin"},
		{"id": 600, "kind": 2, "pos": [102, 100], "hp": 30, "max_hp": 60, "mkind": 1},
	]
	return Codec.encode({"t": "snap", "tick": 1, "you": base, "ents": ents, "gone": [], "notice": "", "full": true})


func run() -> Array:
	var rows: Array = []
	var link := FakeLink.new()
	var session := NetSession.new(link)
	session.connect_to("ws://127.0.0.1:0")
	session.tick_ms = 120  # normally set by the welcome
	var w: Dictionary = welcome_dict()

	var status: StatusPanel = (load("res://game/ui/status_panel.tscn") as PackedScene).instantiate() as StatusPanel
	var bar: SkillBar = (load("res://game/ui/skill_bar.tscn") as PackedScene).instantiate() as SkillBar
	var skills: SkillPanel = (load("res://game/ui/skill_panel.tscn") as PackedScene).instantiate() as SkillPanel
	var cast := CastBar.new()
	var buffs := BuffRow.new()
	for n: Node in [status, bar, skills, cast, buffs]:
		host.add_child(n)
	status.bind(session)
	bar.bind(session)
	skills.bind(session)
	cast.bind(session)
	buffs.bind(session)
	for n: Variant in [status, bar, skills, cast, buffs]:
		n.set_welcome(w)
	status.visible = true
	skills.visible = true
	await host.get_tree().process_frame

	var you: Dictionary = {"name": "Odin", "level": 12, "job_level": 10, "job": 0, "stats": [5, 4, 3, 2, 1, 1],
		"stat_points": 3, "skill_points": 2, "sp": 6, "max_sp": 30, "base_exp": 50, "base_exp_next": 200,
		"job_exp": 0, "job_exp_next": 0, "learned": [[2, 1], [1, 1]], "cooldowns": [[1, 5]], "casting": 2, "cast_left": 4,
		"buffs": [[2, 50, 3]]}
	link.push_inbound(snap_text(you))
	session.poll()
	status.refresh()
	await host.get_tree().process_frame

	rows.append(row("status_fields", status.name_label.text == "Odin" and status.stat_labels[0].text == "5"
			and status.job_label.text == "Novice" and status.hp_bar.max_value == 200.0 and status.sp_bar.value == 6.0,
			"%s %s" % [status.name_label.text, status.stat_labels[0].text]))
	rows.append(row("status_exp_bars", is_equal_approx(status.base_bar.value, 0.25) and status.job_bar.value == 1.0
			and status.job_bar.tooltip_text.ends_with("max"), status.job_bar.tooltip_text))
	rows.append(row("status_job_buttons", status.job_buttons.size() == 6 and not status.job_buttons[0].disabled, str(status.job_buttons.size())))
	rows.append(row("status_points_label", status.points_label.text.contains("3"), status.points_label.text))

	link.sent.clear()
	status.stat_buttons[2].pressed.emit()
	rows.append(row("allot_sends", link.sent.size() == 1 and link.sent[0].contains("\"allot\"") and link.sent[0].contains("\"stat\":2"),
			str(link.sent)))
	status.job_buttons[1].pressed.emit()
	rows.append(row("job_sends", link.sent.size() == 2 and link.sent[1].contains("\"job\"") and link.sent[1].contains("\"job\":2"),
			str(link.sent)))

	# No points: plus buttons disabled and nothing sent.
	link.push_inbound(snap_text({"stat_points": 0, "job_level": 4, "stats": [5, 4, 3, 2, 1, 1]}))
	session.poll()
	status.refresh()
	link.sent.clear()
	status.stat_buttons[0].pressed.emit()
	status.job_buttons[0].pressed.emit()
	rows.append(row("no_points_no_send", status.stat_buttons[0].disabled and status.job_buttons[0].disabled and link.sent.is_empty(),
			str(link.sent)))

	link.push_inbound(snap_text(you))
	session.poll()
	bar.refresh()
	rows.append(row("bar_slots", bar.entries.size() == 2 and bar.buttons[0].text.contains("Bash")
			and bar.buttons[1].text.contains("Heal"), str(bar.entries.size())))
	rows.append(row("bar_sp_grey", bar.buttons[0].disabled and not bar.buttons[1].disabled,
			"%s %s" % [bar.buttons[0].disabled, bar.buttons[1].disabled]))
	rows.append(row("bar_cooldown_sweep", is_equal_approx(bar.dials[0].fraction, 0.5), str(bar.dials[0].fraction)))

	link.sent.clear()
	rows.append(row("enemy_skill_needs_target", not bar.press_slot(0) and link.sent.is_empty(), ""))
	bar.targeting.select(600)
	var big: Dictionary = you.duplicate(true)
	big["sp"] = 20
	big["cooldowns"] = []
	link.push_inbound(snap_text(big))
	session.poll()
	bar.refresh()
	bar.press_slot(0)
	bar.press_slot(1)
	rows.append(row("skill_targets", link.sent.size() == 2 and link.sent[0].contains("\"id\":1") and link.sent[0].contains("\"target\":600")
			and link.sent[1].contains("\"id\":2") and link.sent[1].contains("\"target\":-1"), str(link.sent)))

	cast.refresh()
	buffs.refresh()
	rows.append(row("cast_bar_shows", cast.visible and cast.label.text == "Heal", cast.label.text))
	rows.append(row("buff_icons", buffs.shown == 1 and buffs.icons[0].text.contains("Increase AGI") and buffs.icons[0].text.contains("6s"),
			buffs.icons[0].text if not buffs.icons.is_empty() else "none"))

	skills.refresh()
	rows.append(row("skill_panel_rows", skills.rows.size() == 1 and skills.points_label.text.contains("2"), str(skills.rows.keys())))
	link.sent.clear()
	(skills.rows[2]["button"] as Button).pressed.emit()
	rows.append(row("learn_sends", link.sent.size() == 1 and link.sent[0].contains("\"learn\"") and link.sent[0].contains("\"id\":2"),
			str(link.sent)))

	for n: Node in [status, bar, skills, cast, buffs]:
		n.queue_free()
	await host.get_tree().process_frame
	return rows
