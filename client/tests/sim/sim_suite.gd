extends Node
## Runs the regression sims listed in SIM_FILES (each tests/sim/sim_<name>.gd extends sim_base.gd and returns
## [[name, ok, detail], ...] from run()). Prints "SIM <name> PASS|FAIL <detail>" and "SIMS passed=N failed=M".
## Part of tools/run_sims.sh. Run alone: godot --headless --path . res://tests/sim/sim_suite.tscn [-- name ...]

const SIM_FILES: Array = [
	"sim_example",
]


func _ready() -> void:
	var only: PackedStringArray = OS.get_cmdline_user_args()
	var passed: int = 0
	var failed: int = 0
	for f: String in SIM_FILES:
		if not only.is_empty() and not only.has(f):
			continue
		var scr: GDScript = load("res://tests/sim/%s.gd" % f) as GDScript
		var sim: Variant = scr.new()
		sim.host = self
		var rows: Array = await sim.run()
		for r: Array in rows:
			var verdict: String = "PASS" if r[1] else "FAIL"
			if r[1] == null:
				verdict = "INFO"
			elif r[1]:
				passed += 1
			else:
				failed += 1
			print("SIM %s %s %s" % [r[0], verdict, r[2]])
	print("SIMS passed=%d failed=%d" % [passed, failed])
	get_tree().quit(1 if failed > 0 else 0)
