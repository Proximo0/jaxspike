extends Node
## Temporary soak (untracked, delete after use): presented bot-vs-bot fights back to back for --soak_s seconds of
## wall time. Per fight: real fps, frame p95, and static/video memory at start and end.

var m: Match = null
var fight: int = 0
var seed_base: int = 3000
var t_start_us: int = 0
var fight_start_us: int = 0
var last_us: int = 0
var frame_ms: PackedFloat32Array = []
var mem0: float = 0.0
var vid0: float = 0.0
var soak_s: float = 600.0


func _ready() -> void:
	soak_s = float(Flags.get_arg("soak_s", "600"))
	GfxQuality.set_level(int(Flags.get_arg("quality", "2")))
	t_start_us = Time.get_ticks_usec()
	print("SOAK start static_mb=%.1f video_mb=%.1f" % [_static_mb(), _video_mb()])
	_next()


func _static_mb() -> float:
	return Performance.get_monitor(Performance.MEMORY_STATIC) / 1048576.0


func _video_mb() -> float:
	return Performance.get_monitor(Performance.RENDER_VIDEO_MEM_USED) / 1048576.0


func _next() -> void:
	if m != null:
		m.queue_free()
	fight += 1
	m = Match.new()
	add_child(m)
	m.setup({"mode": "bot_vs_bot", "seed": seed_base + fight, "presentation": true, "loop": false, "time_limit_s": 60.0})
	frame_ms = []
	fight_start_us = Time.get_ticks_usec()
	last_us = fight_start_us
	mem0 = _static_mb()
	vid0 = _video_mb()


func _process(_d: float) -> void:
	var now: int = Time.get_ticks_usec()
	frame_ms.append((now - last_us) / 1000.0)
	last_us = now
	if not (m.over and m.tick > 0) and (now - fight_start_us) < 90_000_000:
		return
	var sorted: Array = Array(frame_ms)
	sorted.sort()
	var secs: float = (now - fight_start_us) / 1e6
	print("SOAK fight=%d seed=%d ticks=%d secs=%.1f fps=%.1f p95_ms=%.2f max_ms=%.2f static_mb=%.1f->%.1f video_mb=%.1f->%.1f nodes=%d objects=%d" % [
		fight, seed_base + fight, m.tick, secs, frame_ms.size() / secs, float(sorted[int(sorted.size() * 0.95)]),
		float(sorted[-1]), mem0, _static_mb(), vid0, _video_mb(),
		int(Performance.get_monitor(Performance.OBJECT_NODE_COUNT)), int(Performance.get_monitor(Performance.OBJECT_COUNT))])
	if (now - t_start_us) / 1e6 >= soak_s:
		print("SOAK end static_mb=%.1f video_mb=%.1f fights=%d" % [_static_mb(), _video_mb(), fight])
		get_tree().quit()
		return
	_next()
