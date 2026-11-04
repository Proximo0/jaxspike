extends SceneTree
## Windowed review render of the card album, a card announcement and a card tooltip against a LIVE server
## (round 4, ro4-cards). Run with the server up (`venv\Scripts\python.exe -m net.server --port 8795`):
##   Godot_v4.7.2-stable_win64.exe --path client -s res://tools/cards/capture_cards.gd -- --url=ws://127.0.0.1:8795 \
##       --out=<dir>
## Logs in as a fresh character, opens the real GameView with the album added like a panel, waits for the welcome
## (welcome.cards from the server), fakes one `card` message and puts two cards into the local copy of `you` (a
## Gildscarab Card carried, an Antlet Card socketed in a carried Sword) so the album lights them and the tooltip
## shows, then writes <out>/live_album.png and <out>/live_tooltip.png. Prints one CAPTURE_CARDS line.

const TIMEOUT_MS: int = 30000

var _out: String = ""
var _url: String = "ws://127.0.0.1:8795"
var _session: NetSession = null
var _view: GameView = null
var _album: CardAlbum = null


func _initialize() -> void:
	for a: String in OS.get_cmdline_user_args():
		if a.begins_with("--url="):
			_url = a.substr(6)
		elif a.begins_with("--out="):
			_out = a.substr(6)
	if _out == "":
		print("CAPTURE_CARDS error: need --out=<dir>")
		quit(1)
		return
	DirAccess.make_dir_recursive_absolute(_out)
	_run.call_deferred()


func _run() -> void:
	_session = NetSession.new(WsLink.new())
	_session.connect_to(_url)
	var t0: int = Time.get_ticks_msec()
	while _session.link.state() != NetLink.STATE_OPEN:
		_session.link.poll_packets()
		await process_frame
		if Time.get_ticks_msec() - t0 > TIMEOUT_MS:
			_fail("no connection to %s" % _url)
			return
	_session.login("cardview%d" % (Time.get_ticks_msec() % 100000))
	_view = (load("res://game/world/game_view.tscn") as PackedScene).instantiate() as GameView
	root.add_child(_view)
	_view.start(_session)
	_album = (load("res://game/ui/card_album.tscn") as PackedScene).instantiate() as CardAlbum
	_view.ui_layer.add_child(_album)
	_view.panels.append(_album)
	_album.bind(_session)
	while _album.cards.is_empty() or _session.store.you.is_empty():
		await process_frame
		if Time.get_ticks_msec() - t0 > TIMEOUT_MS:
			_fail("no welcome.cards / snap (welcomed=%s, error=%s)" % [str(_session.welcomed), _session.last_error])
			return
	var gild: Dictionary = _card_named("Gildscarab Card")
	var antlet: Dictionary = _card_named("Antlet Card")
	_give_cards(int(gild.get("id", 0)), int(antlet.get("id", 0)))
	_album.position = Vector2(420, 120)
	_album.visible = true
	_album.refresh()
	_album.select(_album.rows.find_custom(func(r: Dictionary) -> bool: return int(r["id"]) == int(gild.get("id", -1))))
	_album.on_message({"t": "card", "item": int(gild.get("id", 0)), "kind": 18, "finder": null, "mvp": true,
		"text": "Someone found a Gildscarab Card!"})
	var inv: InventoryPanel = null
	for p: Node in _view.panels:
		if p is InventoryPanel:
			inv = p as InventoryPanel
	var files: int = 0
	files += await _shot("live_album.png")
	if inv != null:
		inv.visible = true
		inv.refresh()
		_album.visible = false
		var tip := PanelContainer.new()
		tip.theme = inv.theme
		var lab := Label.new()
		lab.text = "%s\n\n%s" % [String(inv.rows[0]["tooltip"]), String(inv.rows[1]["tooltip"])]
		tip.add_child(lab)
		_view.ui_layer.add_child(tip)
		tip.position = inv.position + Vector2(-380, 0)
		files += await _shot("live_tooltip.png")
	print("CAPTURE_CARDS done files=%d cards=%d slot=%d url=%s" % [files, _album.cards.size(), _session.slot, _url])
	_session.close()
	quit(0)


func _card_named(n: String) -> Dictionary:
	for r: Variant in _album.cards:
		if String((r as Dictionary).get("name", "")) == n:
			return r
	return {}


## Local-only change of the store's `you` (never sent): a carried card and a carried Sword with a socketed card.
func _give_cards(card: int, socketed: int) -> void:
	var you: Dictionary = _session.store.you
	var inv: Array = (you.get("inv", []) as Array).duplicate(true)
	while inv.size() < InventoryPanel.INV_SLOTS:
		inv.append({"item": 0, "count": 0, "refine": 0, "cards": [0, 0, 0, 0]})
	inv[0] = {"item": card, "count": 1, "refine": 0, "cards": [0, 0, 0, 0]}
	inv[1] = {"item": 3, "count": 1, "refine": 4, "cards": [socketed, 0, 0, 0]}
	you["inv"] = inv


func _shot(name: String) -> int:
	for i: int in range(20):
		await process_frame
	await RenderingServer.frame_post_draw
	var img: Image = root.get_texture().get_image()
	return 1 if img.save_png(_out.path_join(name)) == OK else 0


func _fail(why: String) -> void:
	print("CAPTURE_CARDS error: " + why)
	if _session != null:
		_session.close()
	quit(1)
