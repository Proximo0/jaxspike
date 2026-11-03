extends TestCase
## ClipSelect and ClipTiming.


func test_act_slots_player_and_monster() -> void:
	assert_eq(ClipSelect.act_index(ClipSelect.IDLE, 0, true), 0)
	assert_eq(ClipSelect.act_index(ClipSelect.WALK, 3, true), 11)
	assert_eq(ClipSelect.act_index(ClipSelect.ATTACK, 7, true), 47)
	assert_eq(ClipSelect.act_index(ClipSelect.DEAD, 0, true), 64)
	assert_eq(ClipSelect.act_index(ClipSelect.CAST, 2, true), 98)
	assert_eq(ClipSelect.act_index(ClipSelect.ATTACK, 1, false), 17)
	assert_eq(ClipSelect.act_index(ClipSelect.DEAD, 0, false), 32)
	assert_eq(ClipSelect.clip_name(ClipSelect.SIT, false), "idle")
	assert_eq(ClipSelect.clip_name(ClipSelect.CAST, false), "attack")
	assert_eq(ClipSelect.clip_name(99, true), "idle")


func test_mirroring_rows() -> void:
	assert_eq(ClipSelect.sheet_row(0, 5), Vector2i(0, 0))
	assert_eq(ClipSelect.sheet_row(4, 5), Vector2i(4, 0))
	assert_eq(ClipSelect.sheet_row(5, 5), Vector2i(3, 1))
	assert_eq(ClipSelect.sheet_row(6, 5), Vector2i(2, 1))
	assert_eq(ClipSelect.sheet_row(7, 5), Vector2i(1, 1))
	assert_eq(ClipSelect.sheet_row(6, 8), Vector2i(6, 0))


func test_face_of_delta() -> void:
	assert_eq(ClipSelect.face_of(Vector2(0, 1)), 0)
	assert_eq(ClipSelect.face_of(Vector2(-1, 1)), 1)
	assert_eq(ClipSelect.face_of(Vector2(-1, 0)), 2)
	assert_eq(ClipSelect.face_of(Vector2(0, -1)), 4)
	assert_eq(ClipSelect.face_of(Vector2(1, 0)), 6)
	assert_eq(ClipSelect.face_of(Vector2.ZERO, 5), 5)


func test_walk_cycle_is_two_cells() -> void:
	# 8 frames over 2 cells at 150 ms/cell: 37.5 ms per frame
	assert_eq(ClipTiming.frame_at(ClipSelect.WALK, 0.0, 8, 100.0, 0.0, 150.0), 0)
	assert_eq(ClipTiming.frame_at(ClipSelect.WALK, 150.0, 8, 100.0, 0.0, 150.0), 4)
	assert_eq(ClipTiming.frame_at(ClipSelect.WALK, 300.0, 8, 100.0, 0.0, 150.0), 0)
	assert_eq(ClipTiming.frame_at(ClipSelect.WALK, 150.0, 8, 100.0, 0.0, 75.0), 0)


func test_idle_loops_on_delay() -> void:
	assert_eq(ClipTiming.frame_at(ClipSelect.IDLE, 450.0, 4, 200.0, 0.0, 150.0), 2)
	assert_eq(ClipTiming.frame_at(ClipSelect.IDLE, 850.0, 4, 200.0, 0.0, 150.0), 0)


func test_attack_fits_amotion_with_atk_at_60_percent() -> void:
	var amotion: float = 500.0
	assert_eq(ClipTiming.frame_at(ClipSelect.ATTACK, 0.0, 6, 100.0, amotion, 150.0, 3), 0)
	assert_eq(ClipTiming.frame_at(ClipSelect.ATTACK, 299.0, 6, 100.0, amotion, 150.0, 3), 2)
	assert_eq(ClipTiming.frame_at(ClipSelect.ATTACK, 300.0, 6, 100.0, amotion, 150.0, 3), 3)
	assert_eq(ClipTiming.frame_at(ClipSelect.ATTACK, 499.0, 6, 100.0, amotion, 150.0, 3), 5)
	assert_eq(ClipTiming.frame_at(ClipSelect.ATTACK, 500.0, 6, 100.0, amotion, 150.0, 3), -1)
	assert_near(ClipTiming.atk_ms(ClipSelect.ATTACK, 6, 100.0, amotion), 300.0)


func test_hurt_300_ms_and_dead_holds() -> void:
	assert_eq(ClipTiming.frame_at(ClipSelect.HURT, 100.0, 2, 150.0, 0.0, 150.0), 0)
	assert_eq(ClipTiming.frame_at(ClipSelect.HURT, 200.0, 2, 150.0, 0.0, 150.0), 1)
	assert_eq(ClipTiming.frame_at(ClipSelect.HURT, 300.0, 2, 150.0, 0.0, 150.0), -1)
	assert_eq(ClipTiming.frame_at(ClipSelect.DEAD, 100.0, 3, 150.0, 0.0, 150.0), 0)
	assert_eq(ClipTiming.frame_at(ClipSelect.DEAD, 99999.0, 3, 150.0, 0.0, 150.0), 2)
	assert_eq(ClipTiming.follow_up(ClipSelect.ATTACK), ClipSelect.STANDBY)
	assert_eq(ClipTiming.follow_up(ClipSelect.HURT), ClipSelect.IDLE)
