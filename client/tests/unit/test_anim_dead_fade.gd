extends TestCase
## DeadFade alpha: 1 while the dead clip plays and holds, then a linear fade over FADE_MS.


func test_holds_then_fades() -> void:
	var clip: float = 600.0
	assert_near(DeadFade.alpha(0.0, clip), 1.0)
	assert_near(DeadFade.alpha(clip + DeadFade.HOLD_MS - 1.0, clip), 1.0)
	assert_near(DeadFade.alpha(clip + DeadFade.HOLD_MS, clip), 1.0)
	assert_near(DeadFade.alpha(clip + DeadFade.HOLD_MS + DeadFade.FADE_MS * 0.5, clip), 0.5)
	assert_near(DeadFade.alpha(clip + DeadFade.HOLD_MS + DeadFade.FADE_MS, clip), 0.0)


func test_clamped_and_done_before_server_removal() -> void:
	assert_near(DeadFade.alpha(-50.0, 600.0), 1.0)
	assert_near(DeadFade.alpha(99999.0, 600.0), 0.0)
	assert_true(DeadFade.alpha(3000.0, 600.0) == 0.0, "gone by the 3 s removal")
	assert_near(DeadFade.alpha(10.0, -5.0), 1.0, 0.0001, "negative clip length treated as 0")
