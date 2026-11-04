extends SceneTree
## ro5-items: windowed review render of the inventory, the shop and the card album against a LIVE server.
## Run with the server up (`venv\Scripts\python.exe -m net.server --port 8797`):
##   Godot_v4.7.2-stable_win64_console.exe --path client -s res://tools/items/capture_items.gd -- \
##       --url=ws://127.0.0.1:8797 --out=<dir>
## Logs in as a fresh character, opens the real GameView, waits for the welcome (welcome.items, cards, shops from
## the server), fills the local copy of `you` with a mix of real item ids (never sent) and stands it next to the
## first shop tile so the shop opens, then writes <out>/live_inventory.png, live_shop.png and live_album.png with
## the shared tooltip of one row drawn beside each panel. Prints one CAPTURE_ITEMS line.

const TIMEOUT_MS: int = 30000
## Real game/data/items.py ids: blade, red potion x5, gildscarab card, hat, knife, clover x3, moonstone, bow,
## blue potion x2, boots, ring, antlet card, raw fish x4, cotton shirt.
const FAKE_INV: Array = [
	[24, 1, 4, [64, 0, 0, 0]], [4, 5, 0, []], [88, 1, 0, []], [36, 1, 2, []], [22, 1, 0, []], [11, 3, 0, []],
	[21, 1, 0, []], [27, 1, 0, []], [8, 2, 0, []], [50, 1, 0, []], [51, 1, 0, []], [64, 1, 0, []], [103, 4, 0, []],
	[39, 1, 7, []],
]

var _out: String = ""
var _url: String = "ws://127.0.0.1:8797"
var _session: NetSession = null
var _view: GameView = null
var _shop_tile: Array = []
var _tip: PanelContainer = null
var _tip_label: Label = null


func _initialize() -> void:
	for a: String in OS.get_cmdline_user_args():
		if a.begins_with("--url="):
			_url = a.substr(6)
		elif a.begins_with("--out="):
			_out = a.substr(6)
	if _out == "":
		print("CAPTURE_ITEMS error: need --out=<dir>")
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
	_session.login("itemview%d" % (Time.get_ticks_msec() % 100000))
	_view = (load("res://game/world/game_view.tscn") as PackedScene).instantiate() as GameView
	root.add_child(_view)
	_view.start(_session)
	var inv: InventoryPanel = null
	var shop: ShopPanel = null
	var album: CardAlbum = null
	while inv == null or inv.welcome.is_empty() or _session.store.you.is_empty() or album == null or album.cards.is_empty():
		await process_frame
		for p: Node in _view.panels:
			if p is InventoryPanel:
				inv = p as InventoryPanel
			elif p is ShopPanel:
				shop = p as ShopPanel
			elif p is CardAlbum:
				album = p as CardAlbum
		if Time.get_ticks_msec() - t0 > TIMEOUT_MS:
			_fail("no welcome / snap (welcomed=%s, error=%s)" % [str(_session.welcomed), _session.last_error])
			return
	var tiles: Array = ShopPanel.shop_tiles(inv.welcome)
	if not tiles.is_empty():
		_shop_tile = ((tiles[0] as Dictionary).get("tile", []) as Array).duplicate()
	_tip = PanelContainer.new()
	_tip.theme = inv.theme
	_tip_label = Label.new()
	_tip.add_child(_tip_label)
	_view.ui_layer.add_child(_tip)
	var files: int = 0
	inv.visible = true
	inv.position = Vector2(1300, 200)
	_fake_you(false)
	inv.refresh()
	_show_tip(String(inv.rows[0]["tooltip"]) + "\n\n" + String(inv.rows[2]["tooltip"]), inv.position + Vector2(-430, 0))
	files += await _shot("live_inventory.png", inv, false)
	inv.visible = false
	_fake_you(true)
	shop.refresh()
	var shop_tip: String = ""
	if not shop.rows_buy.is_empty():
		shop_tip = String(shop.rows_buy[0]["tooltip"])
	if not shop.rows_sell.is_empty():
		shop_tip += "\n\n" + String(shop.rows_sell[0]["tooltip"])
	shop.position = Vector2(900, 200)
	_show_tip(shop_tip, Vector2(420, 200))
	files += await _shot("live_shop.png", shop, true)
	shop.visible = false
	shop.dismissed_town = shop.town
	album.visible = true
	album.position = Vector2(420, 120)
	album.refresh()
	album.select(album.rows.find_custom(func(r: Dictionary) -> bool: return String(r["name"]) == "Gildscarab Card"))
	_tip.visible = false
	files += await _shot("live_album.png", album, true)
	var button_ok: bool = album.hud_button != null and album.hud_button.is_visible_in_tree()
	print("CAPTURE_ITEMS done files=%d items=%d buy=%d sell=%d cards=%d button=%s url=%s" % [
		files, (inv.welcome.get("items", []) as Array).size(), shop.rows_buy.size(), shop.rows_sell.size(),
		album.cards.size(), str(button_ok), _url])
	_session.close()
	quit(0)


func _show_tip(text: String, at: Vector2) -> void:
	_tip_label.text = text
	_tip.position = at
	_tip.visible = true


## Local-only change of the store's `you` (never sent): FAKE_INV, nothing worn but the blade, and (for the shop)
## the player standing next to the first shop tile.
func _fake_you(at_shop: bool) -> void:
	var you: Dictionary = _session.store.you
	var rows: Array = []
	for r: Variant in FAKE_INV:
		var a: Array = r
		var cards: Array = [0, 0, 0, 0]
		for j: int in range((a[3] as Array).size()):
			cards[j] = (a[3] as Array)[j]
		rows.append({"item": a[0], "count": a[1], "refine": a[2], "cards": cards})
	while rows.size() < InventoryPanel.INV_SLOTS:
		rows.append({"item": 0, "count": 0, "refine": 0, "cards": [0, 0, 0, 0]})
	you["inv"] = rows
	you["equip"] = [0, -1, -1, -1, -1, -1, -1, -1]
	you["ryn"] = 12500
	if at_shop and _shop_tile.size() >= 2:
		you["pos"] = [int(_shop_tile[0]) + 1, int(_shop_tile[1])]
	_session.store.you = you


## Waits a few frames re-applying the fake `you` (snaps keep arriving), keeps `panel` visible, saves the window.
func _shot(name: String, panel: Control, at_shop: bool) -> int:
	for i: int in range(20):
		_fake_you(at_shop)
		panel.visible = true
		await process_frame
	await RenderingServer.frame_post_draw
	var img: Image = root.get_texture().get_image()
	return 1 if img.save_png(_out.path_join(name)) == OK else 0


func _fail(why: String) -> void:
	print("CAPTURE_ITEMS error: " + why)
	if _session != null:
		_session.close()
	quit(1)
