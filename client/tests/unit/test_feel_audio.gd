extends TestCase
## AudioCues: which feel sounds a tick calls for; the generated wavs exist and import.


func test_cues() -> void:
	var a := AudioCues.new()
	var you: Dictionary = {"pos": [10, 10], "hp": 100, "level": 1, "job_level": 1, "inv": [{"item": 5, "count": 1}]}
	assert_true(a.observe(you, {}).is_empty(), "first observation primes")
	you["pos"] = [11, 10]
	assert_true(a.observe(you, {}).is_empty(), "one step: no footstep yet")
	you["pos"] = [12, 11]
	assert_eq(a.observe(you, {}), [AudioCues.WALK] as Array[String], "every second step")
	you["hp"] = 80
	assert_eq(a.observe(you, {"a": 5, "t": 40}), [AudioCues.ATTACK, AudioCues.HIT] as Array[String])
	assert_true(a.observe(you, {"a": 5, "t": 40}).is_empty(), "same swing, same hp")
	you["level"] = 2
	you["inv"] = [{"item": 5, "count": 2}]
	assert_eq(a.observe(you, {"a": 4, "t": 41}), [AudioCues.LEVEL_UP, AudioCues.PICKUP] as Array[String])
	you["pos"] = [100, 100]
	assert_eq(a.observe(you, {}), [AudioCues.WARP] as Array[String])
	you["hp"] = 0
	assert_true(a.observe(you, {}).is_empty(), "death is not a hit sound")


func test_item_total() -> void:
	assert_eq(AudioCues.item_total([1, 0, 3]), 2)
	assert_eq(AudioCues.item_total([{"item": 1, "count": 3}, {"item": 0, "count": 9}]), 3)
	assert_eq(AudioCues.item_total(null), 0)


func test_sounds_exist() -> void:
	for cue: String in AudioCues.ALL:
		var path: String = FeelAudio.AUDIO_DIR + cue + ".wav"
		assert_true(ResourceLoader.exists(path), path)
		assert_true(load(path) is AudioStream, path + " loads")
