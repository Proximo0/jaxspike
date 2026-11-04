extends "res://tests/sim/sim_base.gd"
## Login flow over FakeLink: welcome -> the map is drawn (MAP*MAP tiles), snaps -> entity nodes exist and move,
## gone removes a node, and nothing is written to user://.

const MAP_SIZE: int = 512


static func _blob(raw: PackedByteArray) -> String:
	return Marshalls.raw_to_base64(raw.compress(FileAccess.COMPRESSION_DEFLATE))


static func welcome_text() -> String:
	var terrain := PackedByteArray()
	terrain.resize(MAP_SIZE * MAP_SIZE)
	for i: int in range(terrain.size()):
		terrain[i] = (i / MAP_SIZE + i % MAP_SIZE) % 8
	var pz := PackedByteArray()
	pz.resize(MAP_SIZE * MAP_SIZE)
	return Codec.encode({"t": "welcome", "slot": 0, "map": MAP_SIZE, "tick_ms": 120, "view_w": 61, "view_h": 33,
			"terrain": _blob(terrain), "pz": _blob(pz)})


static func snap_text(tick: int, full: bool, monster_x: int, gone: Array) -> String:
	var ents: Array = [
		{"id": 0, "kind": 0, "pos": [100, 100], "hp": 150, "max_hp": 200, "name": "Odin"},
		{"id": 600, "kind": 2, "pos": [monster_x, 100], "hp": 30, "max_hp": 60, "mkind": 1},
		{"id": 5000, "kind": 102, "pos": [98, 101], "hp": 0, "max_hp": 0, "item": 2},
	]
	var you: Dictionary = {"pos": [100, 100], "hp": 150, "max_hp": 200, "kills": 2, "deaths": 1,
			"inv": [1, 2, 0], "town": 0, "safe": false}
	return Codec.encode({"t": "snap", "tick": tick, "you": you, "ents": ents, "gone": gone, "notice": "hello",
			"full": full})


func _user_files() -> int:
	var d: DirAccess = DirAccess.open("user://")
	return d.get_files().size() if d != null else 0


func run() -> Array:
	var rows: Array = []
	var files_before: int = _user_files()
	var scene: PackedScene = load("res://game/world/game_view.tscn") as PackedScene
	var view: GameView = scene.instantiate() as GameView
	host.add_child(view)
	var link := FakeLink.new()
	var session := NetSession.new(link)
	session.connect_to("ws://127.0.0.1:0")
	session.login("Odin")
	view.start(session)
	rows.append(row("login_sent", link.sent.size() == 1 and link.sent[0].contains("login"), str(link.sent)))

	link.push_inbound(welcome_text())
	await host.get_tree().process_frame
	await host.get_tree().process_frame
	var cells: int = view.map_renderer.get_used_cells().size()
	rows.append(row("map_tiles", cells == MAP_SIZE * MAP_SIZE and view.map_renderer.tile_count() == cells,
			"cells=%d drawn_ms=%d" % [cells, view.welcome_to_drawn_ms]))
	var atlas: Vector2i = view.map_renderer.get_cell_atlas_coords(Vector2i(3, 2))
	rows.append(row("map_terrain_ids", atlas == Vector2i(5, 0), "atlas=%s" % str(atlas)))

	var tuning: Tuning = Tuning.shared()
	var old_px: float = tuning.tile_px
	for px: int in [16, 48]:
		tuning.set_value("tile_px", float(px))
		view.map_renderer.build(session.map)
		await host.get_tree().process_frame
		rows.append(row("tile_px_%d" % px, view.map_renderer.get_used_cells().size() == MAP_SIZE * MAP_SIZE
				and view.map_renderer.tile_count() == MAP_SIZE * MAP_SIZE
				and is_equal_approx(view.map_renderer.drawn_tile_px(), float(px)),
				"drawn=%.1f atlas=%s" % [view.map_renderer.drawn_tile_px(), str(view.map_renderer.atlas_size())]))
	# A live change after the build swaps the atlas in place and keeps every cell.
	tuning.set_value("tile_px", 24.0)
	await host.get_tree().process_frame
	rows.append(row("tile_px_live", view.map_renderer.get_used_cells().size() == MAP_SIZE * MAP_SIZE
			and is_equal_approx(view.map_renderer.drawn_tile_px(), 24.0),
			"drawn=%.1f" % view.map_renderer.drawn_tile_px()))
	tuning.set_value("tile_px", old_px)
	await host.get_tree().process_frame

	link.push_inbound(snap_text(1, true, 102, []))
	await host.get_tree().process_frame
	await host.get_tree().process_frame
	var layer: EntityLayer = view.entity_layer
	rows.append(row("nodes_exist", layer.active_count() == 3 and layer.has_sprite(600), "n=%d" % layer.active_count()))
	var x1: float = layer.sprite_of(600).position.x
	link.push_inbound(snap_text(2, false, 103, []))
	link.push_inbound(snap_text(3, false, 104, []))
	await host.get_tree().process_frame
	await host.get_tree().process_frame
	await step(30)
	var x2: float = layer.sprite_of(600).position.x
	rows.append(row("node_moves", x2 > x1, "x %.1f -> %.1f" % [x1, x2]))
	rows.append(row("camera_follows_self", view.camera.position.distance_to(layer.world_pos_of(0)) < 0.01,
			"cam=%s" % str(view.camera.position)))

	link.push_inbound(snap_text(4, false, 104, [600]).replace("\"id\":600", "\"id\":601"))
	await host.get_tree().process_frame
	await host.get_tree().process_frame
	rows.append(row("gone_removes", not layer.has_sprite(600), "n=%d" % layer.active_count()))

	view.queue_free()
	await host.get_tree().process_frame
	rows.append(row("nothing_written", _user_files() == files_before, "files=%d" % _user_files()))
	return rows
