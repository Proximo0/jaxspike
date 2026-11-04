extends TestCase
## CheatPanel: every button's message matches net/protocol.py _validate_cheat, towns come from the welcome.

const WELCOME: Dictionary = {"world": {"towns": [{"idx": 0, "name": "Solmera"}, {"idx": 1, "name": "Lunavell"}]}}


func test_messages() -> void:
	assert_eq(CheatPanel.level_message(50, 0), {"t": "cheat", "op": "level", "base": 50, "job": 0})
	assert_eq(CheatPanel.level_message(120, 70), {"t": "cheat", "op": "level", "base": 99, "job": 50}, "clamped")
	assert_eq(CheatPanel.ryn_message(100000), {"t": "cheat", "op": "ryn", "amount": 100000})
	assert_eq(CheatPanel.simple_message("heal"), {"t": "cheat", "op": "heal"})
	assert_eq(CheatPanel.simple_message("kit"), {"t": "cheat", "op": "kit"})
	assert_eq(CheatPanel.simple_message("allskills"), {"t": "cheat", "op": "allskills"})
	assert_eq(CheatPanel.simple_message("nuke"), {})
	assert_eq(CheatPanel.teleport_message(1), {"t": "cheat", "op": "teleport", "town": 1})
	assert_eq(CheatPanel.godmode_message(true), {"t": "cheat", "op": "godmode", "on": true})
	assert_eq(CheatPanel.item_message(24, 1, 7), {"t": "cheat", "op": "item", "item": 24, "count": 1, "refine": 7})
	assert_eq(CheatPanel.item_message(4, 0, 12), {"t": "cheat", "op": "item", "item": 4, "count": 1, "refine": 10})
	assert_eq(CheatPanel.item_message(0, 1, 0), {})


func test_towns_and_digits() -> void:
	assert_eq(CheatPanel.towns_from(WELCOME), ["Solmera", "Lunavell"] as Array[String])
	assert_eq(CheatPanel.towns_from({}).size(), 0)
	assert_eq(CheatPanel.group_digits(100000), "100,000")
	assert_eq(CheatPanel.group_digits(999), "999")
	assert_eq(CheatPanel.group_digits(-1234567), "-1,234,567")


func test_panel_buttons_send() -> void:
	var p: CheatPanel = (load("res://game/ui/cheat_panel.tscn") as PackedScene).instantiate() as CheatPanel
	var sent: Array[Dictionary] = []
	p.message_sent.connect(func(m: Dictionary) -> void: sent.append(m))
	(Engine.get_main_loop() as SceneTree).root.add_child(p)
	assert_eq(p.title_bar.title_label.text, "Cheats (playtest) (F5)")
	assert_false(p.visible, "hidden until F5")
	p.set_welcome(WELCOME)
	var towns: Array[Node] = p.town_row.get_children().filter(func(n: Node) -> bool: return not n.is_queued_for_deletion())
	assert_eq(towns.size(), 2)
	(towns[1] as Button).pressed.emit()
	assert_eq(sent.back(), {"t": "cheat", "op": "teleport", "town": 1})
	p.toggle_godmode()
	assert_eq(sent.back(), {"t": "cheat", "op": "godmode", "on": true})
	assert_eq(p.godmode_button.text, "God mode: ON")
	p.toggle_godmode()
	assert_eq(sent.back(), {"t": "cheat", "op": "godmode", "on": false})
	p.item_spin.value = 35
	p.count_spin.value = 2
	p.refine_spin.value = 4
	p.give_item()
	assert_eq(sent.back(), {"t": "cheat", "op": "item", "item": 35, "count": 2, "refine": 4})
	var labels: Array[String] = []
	_button_texts(p, labels)
	for want: String in ["Level +10", "Level 50", "Job 50", "+100,000 Ryn", "Full heal", "Gear kit", "All skills", "Give"]:
		assert_true(labels.has(want), "button " + want)
	p.queue_free()


func _button_texts(n: Node, out: Array[String]) -> void:
	for c: Node in n.get_children():
		if c is Button:
			out.append((c as Button).text)
		_button_texts(c, out)
