extends TestCase


func test_defaults_and_url() -> void:
	var r: Dictionary = LoginForm.validate("  Odin ", "", "")
	assert_true(bool(r["ok"]))
	assert_eq(String(r["name"]), "Odin")
	assert_eq(String(r["url"]), "ws://127.0.0.1:8765")


func test_explicit_host_port() -> void:
	var r: Dictionary = LoginForm.validate("a", "example.org", "9000")
	assert_eq(String(r["url"]), "ws://example.org:9000")


func test_errors() -> void:
	assert_false(bool(LoginForm.validate("", "h", "1")["ok"]))
	assert_false(bool(LoginForm.validate("x", "h", "abc")["ok"]))
	assert_false(bool(LoginForm.validate("x", "h", "70000")["ok"]))
	assert_false(bool(LoginForm.validate("x", "bad host", "80")["ok"]))
	assert_false(bool(LoginForm.validate("x".repeat(40), "h", "80")["ok"]))
	assert_true(String(LoginForm.validate("", "h", "1")["error"]) != "")


func test_hud_text() -> void:
	var you: Dictionary = {"hp": 80, "max_hp": 200, "kills": 3, "deaths": 1, "inv": [1, 0, 2, 0, 1]}
	assert_eq(HudText.hp_text(you), "HP 80 / 200")
	assert_eq(HudText.score_text(you), "Kills 3   Deaths 1")
	assert_eq(HudText.inv_text(you), "Items: 1 2 1")
	assert_eq(HudText.inv_text({}), "Items: none")
