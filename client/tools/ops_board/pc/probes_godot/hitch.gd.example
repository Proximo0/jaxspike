extends Node
## Temporary hitch finder (untracked, delete after use). Runs presented bot fights back to back and logs every frame
## over --hitch_ms (default 25) with what happened that frame: match build time, gore events, ticks since start.

var m: Match = null
var fight: int = 0
var fights: int = 6
var last_us: int = 0
var start_us: int = 0
var hitch_ms: float = 25.0
var events: Array = []  # gore/hit events since the last frame
var frame_in_fight: int = 0
var build_ms: float = 0.0


func _ready() -> void:
	fights = int(Flags.get_arg("fights", "6"))
	hitch_ms = float(Flags.get_arg("hitch_ms", "25"))
	GfxQuality.set_level(int(Flags.get_arg("quality", "2")))
	_next()


func _next() -> void:
	if m != null:
		m.queue_free()
	fight += 1
	var t0: int = Time.get_ticks_usec()
	m = Match.new()
	add_child(m)
	m.setup({"mode": "bot_vs_bot", "seed": 4000 + fight, "presentation": true, "loop": false, "time_limit_s": 40.0})
	build_ms = (Time.get_ticks_usec() - t0) / 1000.0
	for f: Variant in m.fighters:
		(f as Node).connect("gore", func(e: Dictionary) -> void: events.append("gore:" + str(e.get("kind", "")) + ":" + str(e.get("part", ""))))
	print("HITCH fight=%d build_ms=%.1f" % [fight, build_ms])
	frame_in_fight = 0
	start_us = Time.get_ticks_usec()
	last_us = start_us


func _process(_d: float) -> void:
	var now: int = Time.get_ticks_usec()
	var ms: float = (now - last_us) / 1000.0
	last_us = now
	frame_in_fight += 1
	if ms >= hitch_ms:
		print("HITCH fight=%d frame=%d tick=%d ms=%.1f objects=%d events=%s" % [fight, frame_in_fight, m.tick, ms,
				int(Performance.get_monitor(Performance.OBJECT_COUNT)), ",".join(PackedStringArray(events.map(func(x): return str(x))))])
	events.clear()
	if m.over and m.tick > 0 or (now - start_us) > 60_000_000:
		if fight >= fights:
			get_tree().quit()
			return
		_next()
