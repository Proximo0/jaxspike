extends TestCase
## NoticeFeed: one event per snap, the HUD line holds HOLD_S then clears.


func test_observe_once_per_snap() -> void:
	var f := NoticeFeed.new()
	assert_eq(f.observe(1, ""), "")
	assert_eq(f.observe(2, "Welcome"), "Welcome")
	assert_eq(f.observe(2, "Welcome"), "", "same snap on a later frame counts once")
	assert_eq(f.observe(2, "Welcome"), "")


func test_same_text_in_two_snaps_counts_twice() -> void:
	var f := NoticeFeed.new()
	assert_eq(f.observe(1, "Hit"), "Hit")
	assert_eq(f.observe(2, "Hit"), "Hit")
	assert_eq(f.observe(3, ""), "")


func test_hud_line_holds_then_clears() -> void:
	var f := NoticeFeed.new()
	assert_eq(f.hud_line(0.1), "")
	f.observe(1, "Level up")
	assert_eq(f.hud_line(0.0), "Level up")
	assert_eq(f.hud_line(NoticeFeed.HOLD_S - 0.5), "Level up")
	assert_eq(f.hud_line(1.0), "", "past HOLD_S")
	assert_eq(f.hud_line(0.1), "")


func test_new_notice_restarts_hold() -> void:
	var f := NoticeFeed.new()
	f.observe(1, "A")
	f.hud_line(3.5)
	f.observe(2, "B")
	assert_eq(f.hud_line(3.5), "B")
