extends TestCase
## ChatWindow tab filters and party-tab routing (ChatBox's parsing is tested in test_ui_items_chat).


func test_tab_filters() -> void:
	assert_true(ChatWindow.tab_shows(ChatWindow.TAB_ALL, "say"))
	assert_true(ChatWindow.tab_shows(ChatWindow.TAB_ALL, "system"))
	assert_true(ChatWindow.tab_shows(ChatWindow.TAB_PARTY, "party"))
	assert_false(ChatWindow.tab_shows(ChatWindow.TAB_PARTY, "say"))
	assert_true(ChatWindow.tab_shows(ChatWindow.TAB_WHISPER, "whisper"))
	assert_false(ChatWindow.tab_shows(ChatWindow.TAB_SYSTEM, "whisper"))
	assert_true(ChatWindow.tab_shows(ChatWindow.TAB_SYSTEM, "system"))


func test_route_text() -> void:
	assert_eq(ChatWindow.route_text("hi", ChatWindow.TAB_PARTY), "/p hi")
	assert_eq(ChatWindow.route_text("/w Bob hi", ChatWindow.TAB_PARTY), "/w Bob hi")
	assert_eq(ChatWindow.route_text("hi", ChatWindow.TAB_ALL), "hi")
	assert_eq(ChatWindow.route_text("  ", ChatWindow.TAB_PARTY), "  ")
	assert_eq(ChatBox.parse_command(ChatWindow.route_text("go", ChatWindow.TAB_PARTY))["kind"], "party")


func test_visible_lines_and_unread() -> void:
	var c := ChatWindow.new()
	c.add_line(ChatBox.format_chat({"kind": "say", "from": "A", "text": "x"}))
	c.add_line(ChatBox.format_chat({"kind": "party", "from": "B", "text": "y"}))
	c.add_line({"text": "notice", "color": ChatBox.kind_color("system"), "kind": "system"})
	assert_eq(c.visible_lines(ChatWindow.TAB_ALL).size(), 3)
	assert_eq(c.visible_lines(ChatWindow.TAB_PARTY).size(), 1)
	assert_eq(c.unread[ChatWindow.TAB_PARTY], 1)
	assert_eq(c.unread[ChatWindow.TAB_SYSTEM], 1)
	assert_eq(c.unread[ChatWindow.TAB_WHISPER], 0)
	c.select_tab(ChatWindow.TAB_PARTY)
	assert_eq(c.unread[ChatWindow.TAB_PARTY], 0)
	c.free()


func _snap(notice: Variant) -> String:
	return Codec.encode({"t": "snap", "tick": 1, "full": true, "gone": [], "notice": notice,
		"you": {"pos": [1, 2], "hp": 3, "max_hp": 4}, "ents": []})


func _system_count(c: ChatWindow) -> int:
	return c.visible_lines(ChatWindow.TAB_SYSTEM).size()


func test_notice_logged_once_per_snap() -> void:
	var link := FakeLink.new()
	var s := NetSession.new(link)
	var c := ChatWindow.new()
	c.bind(s)
	link.push_inbound(_snap("Welcome"))
	s.poll()
	c.call("_process", 0.016)
	c.call("_process", 0.016)
	assert_eq(_system_count(c), 1)
	link.push_inbound(_snap("Welcome"))
	s.poll()
	c.call("_process", 0.016)
	c.call("_process", 0.016)
	assert_eq(_system_count(c), 2)
	link.push_inbound(_snap(null))
	s.poll()
	c.call("_process", 0.016)
	assert_eq(_system_count(c), 2)
	c.free()
