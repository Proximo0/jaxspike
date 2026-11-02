extends Node
## Temporary profiler (untracked, delete after use). Runs tests/perf/perf.tscn and, once its measured window starts,
## takes over every scripted node's _process/_physics_process so each script's time can be summed per script path.
## Also logs Performance monitors and real frame wall time per frame to a CSV.

const OUT_DIR: String = "C:/Users/crp27/AppData/Local/Temp/claude/C--Users-crp27-Downloads/cb22016c-ec31-42f0-8909-9864077e9c4b/scratchpad"

var perf: Node
var taken: Array = []  # [node, path, has_process, has_physics]
var proc_us: Dictionary = {}
var phys_us: Dictionary = {}
var proc_calls: Dictionary = {}
var rows: PackedStringArray = []
var measuring: bool = false
var frames: int = 0
var last_us: int = 0
var first: Node
var reported: bool = false
var window_us: int = 0


class Edge extends Node:
	var on_tick: Callable

	func _process(_d: float) -> void:
		on_tick.call()


func _ready() -> void:
	first = Edge.new()
	first.process_priority = -1000000
	first.on_tick = _frame_start
	add_child(first)
	perf = (load("res://tests/perf/perf.tscn") as PackedScene).instantiate()
	add_child(perf)
	process_priority = 1000000
	process_physics_priority = 1000000


func _frame_start() -> void:
	window_us = Time.get_ticks_usec()
	if not measuring and bool(perf.get("_measuring")):
		measuring = true
		_take_over(get_tree().root)
		last_us = Time.get_ticks_usec()
		rows.append("frame,wall_ms,time_process_ms,time_physics_ms,objects,nodes,draw_calls,scripts_ms,window_ms")


func _take_over(n: Node) -> void:
	if n != self and n != first and n != perf and n.get_script() != null:
		var s: Script = n.get_script()
		var hp: bool = n.is_processing() and n.has_method("_process")
		var hy: bool = n.is_physics_processing() and n.has_method("_physics_process")
		if hp or hy:
			if hp:
				n.set_process(false)
			if hy:
				n.set_physics_process(false)
			taken.append([n, s.resource_path if s.resource_path != "" else str(s), hp, hy])
	for c: Node in n.get_children():
		_take_over(c)


func _process(delta: float) -> void:
	if not measuring:
		return
	var t_all: int = Time.get_ticks_usec()
	for e: Array in taken:
		var n: Node = e[0]
		if not e[2] or not is_instance_valid(n) or not n.is_inside_tree():
			continue
		var t0: int = Time.get_ticks_usec()
		n.call("_process", delta)
		var dt: int = Time.get_ticks_usec() - t0
		proc_us[e[1]] = int(proc_us.get(e[1], 0)) + dt
		proc_calls[e[1]] = int(proc_calls.get(e[1], 0)) + 1
	var scripts_us: int = Time.get_ticks_usec() - t_all
	var now: int = Time.get_ticks_usec()
	frames += 1
	rows.append("%d,%.2f,%.2f,%.2f,%d,%d,%d,%.2f,%.2f" % [frames, (now - last_us) / 1000.0,
			Performance.get_monitor(Performance.TIME_PROCESS) * 1000.0,
			Performance.get_monitor(Performance.TIME_PHYSICS_PROCESS) * 1000.0,
			int(Performance.get_monitor(Performance.OBJECT_COUNT)), int(Performance.get_monitor(Performance.OBJECT_NODE_COUNT)),
			int(Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME)), scripts_us / 1000.0,
			(now - window_us) / 1000.0])
	last_us = now
	if frames == 290:
		_report()


func _physics_process(delta: float) -> void:
	if not measuring:
		return
	for e: Array in taken:
		var n: Node = e[0]
		if not e[3] or not is_instance_valid(n) or not n.is_inside_tree():
			continue
		var t0: int = Time.get_ticks_usec()
		n.call("_physics_process", delta)
		phys_us[e[1]] = int(phys_us.get(e[1], 0)) + Time.get_ticks_usec() - t0


func _exit_tree() -> void:
	_report()


func _report() -> void:
	if reported or frames == 0:
		return
	reported = true
	var f: FileAccess = FileAccess.open(OUT_DIR + "/prof_frames_%s.csv" % Flags.get_arg("tag", "run"), FileAccess.WRITE)
	if f != null:
		f.store_string("\n".join(rows))
	var top: Array = []
	for k: Variant in proc_us:
		top.append(["process", str(k), float(proc_us[k]) / 1000.0 / frames, int(proc_calls[k]) / frames])
	for k: Variant in phys_us:
		top.append(["physics", str(k), float(phys_us[k]) / 1000.0 / frames, 0])
	top.sort_custom(func(a: Array, b: Array) -> bool: return float(a[2]) > float(b[2]))
	print("PROF frames=%d scripted_nodes=%d" % [frames, taken.size()])
	for r: Array in top.slice(0, 25):
		print("PROF %-8s %7.3f ms/frame  nodes=%d  %s" % [r[0], r[2], r[3], r[1]])
