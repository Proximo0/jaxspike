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
