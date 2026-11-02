extends TestCase


func test_parse_say_party_whisper() -> void:
	assert_eq(ChatBox.parse_command("hello there"), {"t": "chat", "kind": "say", "text": "hello there", "to": null})
	assert_eq(ChatBox.parse_command("/p  go  now "), {"t": "chat", "kind": "party", "text": "go  now", "to": null})
	assert_eq(ChatBox.parse_command("/w Bob hi you"), {"t": "chat", "kind": "whisper", "text": "hi you", "to": "Bob"})
	assert_eq(ChatBox.parse_command("/unknown x")["kind"], "say")


func test_parse_rejects_empty() -> void:
	assert_true(ChatBox.parse_command("   ").is_empty())
	assert_true(ChatBox.parse_command("/p").is_empty())
	assert_true(ChatBox.parse_command("/w Bob").is_empty())
	assert_true(ChatBox.parse_command("/w").is_empty())


func test_format_chat_kinds() -> void:
	var s: Dictionary = ChatBox.format_chat({"kind": "say", "from": "A", "text": "x"})
	assert_eq(s["text"], "A: x")
	assert_eq(ChatBox.format_chat({"kind": "party", "from": "A", "text": "x"})["text"], "[Party] A: x")
	assert_eq(ChatBox.format_chat({"kind": "whisper", "from": "A", "text": "x"})["text"], "A whispers: x")
	var sys: Dictionary = ChatBox.format_chat({"kind": "system", "text": "no such player"})
	assert_eq(sys["text"], "no such player")
	var colors: Dictionary = {}
	for k: String in ["say", "party", "whisper", "system"]:
		colors[ChatBox.kind_color(k)] = true
	assert_eq(colors.size(), 4)


func test_format_mvp() -> void:
	var w: Dictionary = {"mvp_kinds": [null, null, "Orc Lord"]}
	assert_eq(ChatBox.format_mvp({"kind": 2, "killer": "Odin"}, w)["text"], "Odin slew the MVP Orc Lord!")
	assert_eq(ChatBox.format_mvp({"kind": 2, "killer": null}, w)["text"], "MVP Orc Lord was slain!")
	assert_eq(ChatBox.format_mvp({"kind": 9, "killer": null}, w)["text"], "MVP an MVP was slain!")


func test_party_message() -> void:
	assert_eq(PartyPanel.party_message("create"), {"t": "party", "op": "create", "target": null})
	assert_eq(PartyPanel.party_message("invite", "Bob"), {"t": "party", "op": "invite", "target": "Bob"})
	assert_true(PartyPanel.party_message("kick").is_empty())
	assert_eq(PartyPanel.status_text(-1, false), "No party")
	assert_eq(PartyPanel.status_text(2, true), "Party 2 (you lead)")
	assert_eq(PartyPanel.names_from(["a", "b"]), ["a", "b"] as Array[String])
