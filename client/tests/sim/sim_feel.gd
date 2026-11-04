extends "res://tests/sim/sim_base.gd"
## Team client-feel over FakeLink: a GameView fed a welcome (128 map with a town, a dungeon box, a warp and a kafra)
## and snaps, with the feel pieces wired the way the lead's game_view lines will wire them. Checks the basic info
## window, the zone banner on entry (town -> dungeon -> field), the minimap texture and markers, the path preview,
## the target ring, the smooth zoom, the chat window's System tab and the audio cues; round 3: the stat section under
## the info window (C), notices once, F2 camera_zoom live, warp camera snap, skill bar grey-out, cast bar seconds and
## RO damage numbers. With FEEL_SHOT=<png path> in
## a windowed run it also saves a screenshot of the view.

const MAP_N: int = 128
const TOWN_C: Vector2i = Vector2i(64, 64)
const DUNGEON_BOX: Array = [100, 0, 127, 20]


static func _feel_blob(raw: PackedByteArray) -> String:
	return Marshalls.raw_to_base64(raw.compress(FileAccess.COMPRESSION_DEFLATE))


static func welcome_dict() -> Dictionary:
	var terrain := PackedByteArray()
	terrain.resize(MAP_N * MAP_N)
	var pz := PackedByteArray()
	pz.resize(MAP_N * MAP_N)
	for y: int in range(MAP_N):
		for x: int in range(MAP_N):
			var i: int = y * MAP_N + x
			var t: int = 0
			if (x * 7 + y * 13) % 29 == 0:
				t = 5  # scattered trees
			if absi(x - TOWN_C.x) <= 8 and absi(y - TOWN_C.y) <= 8:
				t = 2 if (absi(x - TOWN_C.x) <= 2 or absi(y - TOWN_C.y) <= 2) else 3
				pz[i] = 1
			if x == TOWN_C.x or y == TOWN_C.y:
				t = 1 if t == 0 or t == 5 else t  # roads
			if x >= DUNGEON_BOX[0] and y <= DUNGEON_BOX[3]:
				t = 7 if (x == DUNGEON_BOX[0] or y == DUNGEON_BOX[3]) else 11
			if x >= 30 and x <= 40 and y >= 90 and y <= 100:
				t = 4  # a pond
			terrain[i] = t
	terrain[TOWN_C.y * MAP_N + TOWN_C.x + 6] = 9  # kafra
	terrain[25 * MAP_N + 110] = 8  # warp just below the dungeon wall
	return {"t": "welcome", "slot": 0, "map": MAP_N, "tick_ms": 100, "view_w": 61, "view_h": 33,
		"terrain": _feel_blob(terrain), "pz": _feel_blob(pz),
		"job_names": ["Novice", "Swordsman"],
		"world": {"towns": [{"idx": 0, "name": "Prontera", "role": "capital", "center": [TOWN_C.x, TOWN_C.y],
				"kafra": [TOWN_C.x + 6, TOWN_C.y], "station": [TOWN_C.x - 6, TOWN_C.y]}],
			"dungeons": [{"idx": 0, "name": "Prontera Culvert", "box": DUNGEON_BOX}],
			"warps": [{"from": [110, 25], "to": [110, 15]}]}}


static func snap_text(tick: int, pos: Vector2i, you_extra: Dictionary, notice: String) -> String:
	var you: Dictionary = {"pos": [pos.x, pos.y], "hp": 150, "max_hp": 200, "kills": 0, "deaths": 0, "town": 0,
		"safe": false, "level": 12, "job_level": 7, "job": 1, "sp": 6, "max_sp": 30, "base_exp": 50,
		"base_exp_next": 200, "job_exp": 10, "job_exp_next": 40, "zeny": 1234, "inv": []}
	you.merge(you_extra, true)
	var ents: Array = [
		{"id": 0, "kind": 0, "pos": [pos.x, pos.y], "hp": int(you["hp"]), "max_hp": 200, "name": "Odin"},
		{"id": 600, "kind": 2, "pos": [pos.x + 3, pos.y + 1], "hp": 30, "max_hp": 60, "mkind": 1},
		{"id": 601, "kind": 1, "pos": [pos.x - 4, pos.y + 2], "hp": 40, "max_hp": 40, "mkind": 0},
		{"id": 5000, "kind": 102, "pos": [pos.x + 1, pos.y - 2], "hp": 0, "max_hp": 0, "item": 2},
	]
	return Codec.encode({"t": "snap", "tick": tick, "you": you, "ents": ents, "gone": [], "notice": notice, "full": true})


func run() -> Array:
	var rows: Array = []
	var view: GameView = (load("res://game/world/game_view.tscn") as PackedScene).instantiate() as GameView
	host.add_child(view)
	var link := FakeLink.new()
	var session := NetSession.new(link)
	session.connect_to("ws://127.0.0.1:0")
	session.login("Odin")
	view.start(session)
	# ---- the lead's game_view lines, applied here until they land ----
	view.walk_marker.map = session.map
	var ring := TargetRing.new()
	view.add_child(ring)
	ring.bind(view.entity_layer, view.skill_bar.targeting)
	var audio := FeelAudio.new()
	view.add_child(audio)
	audio.bind(session)
	view.player_input.entity_layer = view.entity_layer
	var chat: ChatWindow = (load("res://game/ui/chat_window.tscn") as PackedScene).instantiate() as ChatWindow
	view.ui_layer.add_child(chat)
	chat.position = Vector2(8, 820)
	chat.bind(session)
	view.panels.append(chat)
	view.panels.append(view.hud)
	for p: Node in view.panels:
		if p.name == "ChatBox":
			(p as Control).visible = false  # the window replaces the box
	# ------------------------------------------------------------------
	var w: Dictionary = welcome_dict()
	link.push_inbound(Codec.encode(w))
	link.push_inbound(snap_text(1, TOWN_C, {}, "Welcome to Prontera"))
	await _feel_frames(3)
	var hud: Hud = view.hud
	rows.append(row("info_window", hud.name_label.text == "Odin" and hud.job_label.text == "Swordsman"
			and hud.level_label.text == "Base Lv. 12   Job Lv. 7" and hud.hp_label.text == "150 / 200"
			and hud.base_label.text == "25.0 %" and hud.job_exp_label.text == "25.0 %" and hud.zeny_label.text == "Zeny 1,234"
			and is_equal_approx(hud.hp_bar.value, 0.75),
			"%s|%s|%s|%s|%s" % [hud.name_label.text, hud.job_label.text, hud.hp_label.text, hud.base_label.text, hud.zeny_label.text]))
	rows.append(row("banner_town", hud.banner.title.text == "Prontera" and hud.banner.zone["kind"] == "town"
			and hud.banner.age_s >= 0.0, hud.banner.title.text))
	rows.append(row("minimap_built", hud.minimap.texture != null and hud.minimap.texture.get_width() == MAP_N
			and hud.minimap.markers.size() == 2, "markers=%d" % hud.minimap.markers.size()))
	rows.append(row("chat_system_notice", chat.visible_lines(ChatWindow.TAB_SYSTEM).size() == 1
			and chat.unread[ChatWindow.TAB_SYSTEM] == 1, str(chat.unread)))

	# Click-to-walk: marker + path preview.
	var goal: Vector2i = TOWN_C + Vector2i(5, 3)
	view.player_input.click_walk(goal)
	rows.append(row("path_preview", view.walk_marker.active and view.walk_marker.path.size() == 5
			and view.walk_marker.path[view.walk_marker.path.size() - 1] == goal, "cells=%d" % view.walk_marker.path.size()))

	# Target ring on the selected monster.
	view.skill_bar.targeting.select(600)
	await _feel_frames(2)
	rows.append(row("target_ring", ring.visible and ring.position == view.entity_layer.world_pos_of(600)
			and ring.cat == ViewMath.CAT_MONSTER, str(ring.position)))

	# Smooth zoom: two notches in, eased over frames, applied to the camera.
	view.player_input.zoom.wheel(2)
	await _feel_frames(2)
	var mid: float = view.camera.zoom.x
	for _i: int in range(600):  # converges in about 0.5 s of frame time, whatever the frame rate
		if is_equal_approx(view.camera.zoom.x, view.player_input.zoom.target):
			break
		await _feel_frames(1)
	rows.append(row("zoom_smooth", mid > 1.0 and mid < view.player_input.zoom.target
			and is_equal_approx(view.camera.zoom.x, view.player_input.zoom.target), "mid=%.3f end=%.3f" % [mid, view.camera.zoom.x]))

	# Cursor over the monster: attack.
	var hc: int = view.player_input.hover_category(view.entity_layer.world_pos_of(600))
	rows.append(row("cursor_attack", CursorState.pick(hc, 0) == CursorState.ATTACK, "cat=%d" % hc))

	# Walk a step, then warp into the dungeon with a level up: banner + audio.
	link.push_inbound(snap_text(2, TOWN_C + Vector2i(1, 0), {}, ""))
	await _feel_frames(2)
	link.push_inbound(snap_text(3, Vector2i(110, 10), {"level": 13}, ""))
	await _feel_frames(3)
	rows.append(row("banner_dungeon", hud.banner.title.text == "Prontera Culvert" and hud.banner.shown == 2, hud.banner.title.text))
	rows.append(row("audio_cues", audio.played.has(AudioCues.WARP) and audio.played.has(AudioCues.LEVEL_UP)
			and audio.streams.size() == AudioCues.ALL.size(), str(audio.played)))
	link.push_inbound(snap_text(4, Vector2i(64, 100), {"level": 13}, ""))
	await _feel_frames(2)
	rows.append(row("banner_field", hud.banner.title.text == "Prontera Field" and hud.banner.zone["kind"] == "field", hud.banner.title.text))

	# ---- round 3 (ro3-client) ----
	# Warp camera snap: the 1-tile step (snap 2) was no warp, the two jumps (snaps 3 and 4) were.
	rows.append(row("warp_snaps_counted", view.player_input.follow.snaps == 2, "snaps=%d" % view.player_input.follow.snaps))
	link.push_inbound(snap_text(5, Vector2i(65, 100), {"level": 13, "warped": true}, ""))
	await _feel_frames(1)
	var cam_tile: Vector2 = ViewMath.tile_center(Vector2(65, 100), view._tile_px())
	rows.append(row("warp_flag_snaps_camera", view.player_input.follow.snaps == 3 and view.camera.position == cam_tile,
			"snaps=%d cam=%s" % [view.player_input.follow.snaps, str(view.camera.position)]))
	link.push_inbound(snap_text(6, Vector2i(66, 100), {"level": 13}, ""))
	await _feel_frames(1)
	rows.append(row("warp_flag_one_tick", view.player_input.follow.snaps == 3, "snaps=%d" % view.player_input.follow.snaps))

	# Notices: the same text in two snaps is two System lines; one snap over many frames is one; the HUD line holds.
	var sys0: int = chat.visible_lines(ChatWindow.TAB_SYSTEM).size()
	link.push_inbound(snap_text(7, Vector2i(66, 100), {"level": 13}, "blocked"))
	await _feel_frames(4)
	link.push_inbound(snap_text(8, Vector2i(66, 100), {"level": 13}, "blocked"))
	await _feel_frames(4)
	link.push_inbound(snap_text(9, Vector2i(66, 100), {"level": 13}, ""))
	await _feel_frames(2)
	var sys1: int = chat.visible_lines(ChatWindow.TAB_SYSTEM).size()
	rows.append(row("notice_once_per_snap", sys1 == sys0 + 2 and hud.notice_label.text == "blocked",
			"lines %d -> %d hud=%s" % [sys0, sys1, hud.notice_label.text]))

	# Stat section under the info window: C shows it with the stats, C hides it again.
	link.push_inbound(snap_text(10, Vector2i(66, 100), {"level": 13, "stats": [5, 4, 3, 2, 1, 1], "stat_points": 3, "skill_points": 2}, ""))
	await _feel_frames(2)
	var key := InputEventKey.new()
	key.physical_keycode = KEY_C
	key.pressed = true
	hud.call("_unhandled_key_input", key)
	await _feel_frames(2)
	var shown: bool = hud.stats.visible and hud.stats.stat_labels[0].text == "5" and hud.stats.points_label.text.contains("3")
	var info_h: float = hud.info.size.y
	hud.call("_unhandled_key_input", key)
	await _feel_frames(2)
	rows.append(row("status_section_toggle", shown and not hud.stats.visible and hud.info.size.y < info_h,
			"stat0=%s open=%.0f closed=%.0f" % [hud.stats.stat_labels[0].text, info_h, hud.info.size.y]))

	# F2 Tuning camera_zoom applies live (eased), the wheel still works afterwards.
	Tuning.shared().set_value("camera_zoom", 1.3)
	for _i: int in range(600):
		if is_equal_approx(view.camera.zoom.x, 1.3):
			break
		await _feel_frames(1)
	var tuned_ok: bool = is_equal_approx(view.camera.zoom.x, 1.3)
	view.player_input.zoom.wheel(1)
	await _feel_frames(2)
	rows.append(row("tuning_zoom_live", tuned_ok and view.camera.zoom.x > 1.3, "zoom=%.3f" % view.camera.zoom.x))
	Tuning.shared().reset()

	# Skill bar greys out during after_delay (not during a cast); the cast bar reads cast_left as seconds.
	var cast: CastBar = null
	for p: Node in view.panels:
		if p is CastBar:
			cast = p as CastBar
	link.push_inbound(snap_text(11, Vector2i(66, 100), {"level": 13, "after_delay": 5}, ""))
	await _feel_frames(2)
	var grey: bool = view.skill_bar.greyed and view.skill_bar.buttons[0].modulate == SkillBar.GREY
	link.push_inbound(snap_text(12, Vector2i(66, 100), {"level": 13, "after_delay": 0, "casting": 2, "cast_left": 4}, ""))
	await _feel_frames(2)
	var casting_grey: bool = (not view.skill_bar.greyed and cast != null and cast.visible and cast.time_label.text.begins_with("0.")
			and cast.track.seconds_left(100) > 0.25 and cast.track.seconds_left(100) <= 0.4)
	link.push_inbound(snap_text(13, Vector2i(66, 100), {"level": 13, "after_delay": 0, "casting": 0, "cast_left": 0}, ""))
	await _feel_frames(2)
	rows.append(row("skill_bar_after_delay", grey and casting_grey and not view.skill_bar.greyed and not cast.visible,
			"grey=%s casting=%s time=%s" % [grey, casting_grey, cast.time_label.text if cast != null else "-"]))

	# RO damage numbers: a hit on the monster (white), a hit on you (red) and a miss on the monster; fed from `h`.
	var dmg_snap: Dictionary = Codec.decode(snap_text(14, Vector2i(66, 100), {"level": 13, "hp": 140}, ""))
	(dmg_snap["ents"][0] as Dictionary)["h"] = 10
	(dmg_snap["ents"][2] as Dictionary)["h"] = 7
	link.push_inbound(Codec.encode(dmg_snap))
	await _feel_frames(2)
	var miss_snap: Dictionary = Codec.decode(snap_text(15, Vector2i(66, 100), {"level": 13, "hp": 140}, ""))
	(miss_snap["ents"][2] as Dictionary)["h"] = 0
	link.push_inbound(Codec.encode(miss_snap))
	await _feel_frames(3)
	var seen: Dictionary = {}
	for p2: Dictionary in view.entity_layer.shown_pops:
		seen[[int(p2["id"]), int(p2["dmg"])]] = true
	var pending: int = view.entity_layer.pops.pending()
	rows.append(row("damage_numbers_fed", (seen.has([0, 10]) or pending > 0) and (seen.has([601, 7]) or pending > 0)
			and DamageStyle.kind_of(10, true, false) == DamageStyle.OWN and DamageStyle.kind_of(7, false, false) == DamageStyle.NORMAL
			and DamageStyle.kind_of(0, false, false) == DamageStyle.MISS and DamageStyle.text_of(0, DamageStyle.MISS) == "Miss"
			and DamageStyle.color_of(DamageStyle.CRIT) != DamageStyle.color_of(DamageStyle.NORMAL),
			"shown=%s pending=%d" % [str(seen.keys()), pending]))

	var shot: String = OS.get_environment("FEEL_SHOT")
	if shot != "" and DisplayServer.get_name() != "headless":
		link.push_inbound(snap_text(5, TOWN_C + Vector2i(-2, 1), {"level": 13}, "Welcome to Prontera"))
		hud.minimap.toggle_dots()
		view.player_input.zoom.snap(1.5)
		await _feel_frames(2)
		view.player_input.click_walk(TOWN_C + Vector2i(8, 6))
		hud.banner.zone_key = ""
		await _feel_frames(30)
		var img: Image = host.get_viewport().get_texture().get_image()
		rows.append(row("feel_shot", img.save_png(shot) == OK, shot))

	view.queue_free()
	await _feel_frames(1)
	return rows


## Waits `n` rendered (process) frames: the HUD and the view update in _process, not per physics tick.
func _feel_frames(n: int) -> void:
	for _i: int in range(n):
		await host.get_tree().process_frame
