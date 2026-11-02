extends "res://tests/sim/sim_base.gd"
## Items/craft/party/chat panels over FakeLink: welcome tables + a `you` with inventory, craft and party populate the
## panels, and clicks send the documented messages (read back from FakeLink.sent).

static func _welcome() -> Dictionary:
	var items: Array = [{}]
	var kinds: Array = [0, 1, 2, 3, 4, 5]
	var names: Array = ["", "Gold", "Red Potion", "Sword", "Cap", "Poring Card"]
	for i: int in range(1, 6):
		items.append({"id": i, "name": names[i], "kind": kinds[i]})
	return {"t": "welcome", "slot": 0, "items": items, "craft_items": {"100": "copper ore", "200": "copper bar"},
			"recipes": [
				{"id": 1, "inputs": [100, 100], "output": 200, "skill": 1, "level": 1, "exp": 5, "station": 6},
				{"id": 2, "inputs": [100], "output": 200, "skill": 1, "level": 5, "exp": 5, "station": -1},
			],
			"craft_skills": {"SKILL_NAMES": ["a"], "LEVEL_EXP": [0, 50, 200, 450]}, "mvp_kinds": [null, null, "Orc Lord"]}


static func _snap(tick: int, extra: Dictionary, ents: Array) -> String:
	var inv: Array = []
	for i: int in range(16):
		inv.append({"item": 0, "count": 0, "refine": 0, "cards": [0, 0, 0, 0]})
	inv[0] = {"item": 2, "count": 4, "refine": 0, "cards": [0, 0, 0, 0]}
	inv[1] = {"item": 3, "count": 1, "refine": 2, "cards": [0, 0, 0, 0]}
	inv[2] = {"item": 100, "count": 2, "refine": 0, "cards": [0, 0, 0, 0]}
	var you: Dictionary = {"pos": [10, 10], "hp": 10, "max_hp": 10, "zeny": 1234, "inv": inv,
			"equip": [1, -1, -1, -1, -1, -1, -1, -1], "party": 3, "leader": true}
	you.merge(extra, true)
	return Codec.encode({"t": "snap", "tick": tick, "you": you, "ents": ents, "gone": [], "full": true})


func _last(link: FakeLink) -> Dictionary:
	return Codec.decode(link.sent[link.sent.size() - 1]) if not link.sent.is_empty() else {}


func _same(m: Dictionary, want: Dictionary) -> bool:
	for k: String in want:
		if not m.has(k):
			return false
		if typeof(m[k]) == TYPE_FLOAT or typeof(m[k]) == TYPE_INT:
			if float(m[k]) != float(want[k]):
				return false
		elif str(m[k]) != str(want[k]):
			return false
	return true


func run() -> Array:
	var rows: Array = []
	var link := FakeLink.new()
	var session := NetSession.new(link)
	session.connect_to("ws://127.0.0.1:0")
	var inv_panel: InventoryPanel = (load("res://game/ui/inventory_panel.tscn") as PackedScene).instantiate()
	var craft_panel: CraftPanel = (load("res://game/ui/craft_panel.tscn") as PackedScene).instantiate()
	var party_panel: PartyPanel = (load("res://game/ui/party_panel.tscn") as PackedScene).instantiate()
	var chat: ChatBox = (load("res://game/ui/chat_box.tscn") as PackedScene).instantiate()
	for p: Node in [inv_panel, craft_panel, party_panel, chat]:
		host.add_child(p)
		p.call("bind", session)
	var wl: Dictionary = _welcome()
	link.push_inbound(Codec.encode(wl))
	var nodes: Array = [
		{"id": 9000, "kind": 201, "pos": [10, 11], "hp": 1},
		{"id": 9001, "kind": 206, "pos": [9, 10], "hp": 1},
	]
	link.push_inbound(_snap(1, {"craft": {"level": [1, 1, 1, 1, 1, 1], "exp": [0, 25, 0, 0, 0, 0]},
			"members": ["Odin", "Bob"]}, nodes))
	for m: Dictionary in session.poll():
		for p: Node in [inv_panel, craft_panel, party_panel, chat]:
			p.call("on_message", m)
	await host.get_tree().process_frame
	await host.get_tree().process_frame

	rows.append(row("inv_populated", inv_panel.rows.size() == 16 and inv_panel.rows[0]["name"] == "Red Potion"
			and inv_panel.slot_buttons[0].text == "4" and inv_panel.zeny_label.text == "Zeny 1234",
			"%s %s" % [inv_panel.slot_buttons[0].text, inv_panel.zeny_label.text]))
	rows.append(row("inv_equip_marker", inv_panel.slot_buttons[1].text.begins_with("E")
			and inv_panel.equip_buttons[0].text == "Weapon: Sword" and inv_panel.equip_buttons.size() == 8,
			inv_panel.equip_buttons[0].text))
	rows.append(row("inv_icon_fallback", inv_panel.slot_buttons[2].icon != null, "craft item icon"))
	rows.append(row("storage_hidden_until_sent", not inv_panel.storage_box.visible, ""))

	inv_panel.click_slot(0)
	rows.append(row("click_use", _same(_last(link), {"t": "use_item", "slot": 0}), link.sent[link.sent.size() - 1]))
	inv_panel.click_slot(1)
	rows.append(row("click_equip_toggle", _same(_last(link), {"t": "equip", "slot": 1}), link.sent[link.sent.size() - 1]))
	inv_panel.open_menu(1, Vector2i(5, 5))
	rows.append(row("menu_entries", inv_panel.menu.item_count == 3 and inv_panel.menu_actions[1] == "Refine",
			str(inv_panel.menu_actions)))
	inv_panel._on_menu_id(1)
	rows.append(row("menu_refine", _same(_last(link), {"t": "refine", "slot": 1}), link.sent[link.sent.size() - 1]))
	inv_panel._on_menu_id(2)
	rows.append(row("menu_drop", _same(_last(link), {"t": "drop", "slot": 1}), link.sent[link.sent.size() - 1]))
	inv_panel.click_equip(0)
	rows.append(row("equip_panel_click", _same(_last(link), {"t": "equip", "slot": 1}), ""))

	# Craft panel: level 1 smithing, 2 ore, furnace adjacent: recipe 1 ok, recipe 2 needs level 5.
	rows.append(row("craft_eligibility", craft_panel.statuses.size() == 2 and bool(craft_panel.statuses[1]["ok"])
			and not bool(craft_panel.statuses[2]["ok"]), str(craft_panel.statuses)))
	rows.append(row("craft_exp_bar", absf(craft_panel.skill_bars[1].value - 0.5) < 0.001,
			str(craft_panel.skill_bars[1].value)))
	rows.append(row("craft_grey", craft_panel.recipe_buttons[2].modulate == CraftPanel.GREY
			and craft_panel.recipe_buttons[1].modulate == Color.WHITE, ""))
	var before: int = link.sent.size()
	craft_panel.select_recipe(2)
	craft_panel.craft_selected()
	rows.append(row("craft_blocked_sends_nothing", link.sent.size() == before, ""))
	craft_panel.select_recipe(1)
	craft_panel.craft_selected()
	rows.append(row("craft_sends", _same(_last(link), {"t": "craft", "recipe": 1}), link.sent[link.sent.size() - 1]))
	craft_panel.gather()
	rows.append(row("gather_adjacent", _same(_last(link), {"t": "gather"}), link.sent[link.sent.size() - 1]))

	# craft/members omitted from the next snap: cached values stay.
	link.push_inbound(_snap(2, {}, [{"id": 9000, "kind": 201, "pos": [30, 30], "hp": 1}]))
	session.poll()
	await host.get_tree().process_frame
	await host.get_tree().process_frame
	before = link.sent.size()
	craft_panel.gather()
	rows.append(row("gather_far_sends_nothing", link.sent.size() == before, ""))
	rows.append(row("sticky_craft_members", absf(craft_panel.skill_bars[1].value - 0.5) < 0.001
			and party_panel.members.size() == 2, str(party_panel.members)))

	# Party panel.
	rows.append(row("party_shown", party_panel.status_label.text == "Party 3 (you lead)"
			and party_panel.members_label.text.contains("Bob"), party_panel.status_label.text))
	party_panel.target_edit.text = "Bob"
	party_panel.invite_target()
	rows.append(row("party_invite", _same(_last(link), {"t": "party", "op": "invite", "target": "Bob"}), link.sent[link.sent.size() - 1]))
	party_panel.send_op("leave")
	rows.append(row("party_leave", _same(_last(link), {"t": "party", "op": "leave"}), ""))
	party_panel.send_op("accept")
	rows.append(row("party_accept", _same(_last(link), {"t": "party", "op": "accept"}), ""))

	# Storage appears once received.
	var st: Array = []
	st.resize(100)
	st.fill(0)
	st[4] = 2
	link.push_inbound(_snap(3, {"storage": st}, []))
	session.poll()
	await host.get_tree().process_frame
	rows.append(row("storage_shown", inv_panel.storage_box.visible and inv_panel.storage_list.get_child_count() == 1, ""))
	inv_panel.take_out(4)
	rows.append(row("storage_out", _same(_last(link), {"t": "storage", "op": "out", "slot": 4}), ""))
	inv_panel.send_action("Store", 0)
	rows.append(row("storage_in", _same(_last(link), {"t": "storage", "op": "in", "slot": 0}), ""))

	# Chat.
	chat.submit("/p hello party")
	rows.append(row("chat_party_sent", _same(_last(link), {"t": "chat", "kind": "party", "text": "hello party"}), ""))
	chat.submit("/w Bob psst")
	rows.append(row("chat_whisper_sent", _same(_last(link), {"t": "chat", "kind": "whisper", "text": "psst", "to": "Bob"}), ""))
	chat.on_message({"t": "chat", "kind": "say", "from": "Bob", "text": "hi"})
	chat.on_message({"t": "mvp", "kind": 2, "killer": "Bob", "tick": 5})
	rows.append(row("chat_log", chat.lines.size() == 3 and String(chat.lines[2]["text"]).contains("Orc Lord"),
			str(chat.lines.size())))

	for p: Node in [inv_panel, craft_panel, party_panel, chat]:
		p.queue_free()
	await host.get_tree().process_frame
	return rows
