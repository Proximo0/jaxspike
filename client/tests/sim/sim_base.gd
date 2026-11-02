extends RefCounted
## Shared harness for the regression sims (sim_*.gd, run by tests/sim/sim_suite.gd).
## A sim extends this, and `host` (a Node in the tree) is set before `run()`. Implement `func run() -> Array`
## returning [[name, ok, detail], ...]; ok == null reports INFO only. Everything is in sim seconds (physics
## ticks), never wall clock.

const DT: float = 1.0 / 60.0


## A port for a sim that opens sockets: a per-process block of 64 ports (20000..45600, from the OS process id)
## plus a slot offset, so two suites running at once on one machine never listen on the same port. List the slots
## in use here. Never write a literal port in a test.
static func net_port(slot: int) -> int:
	return 20000 + (OS.get_process_id() % 400) * 64 + slot


var host: Node
var world: Node3D


## An empty Node3D under the host to build the scenario in.
func sandbox() -> void:
	world = Node3D.new()
	host.add_child(world)
	await host.get_tree().physics_frame


func clear() -> void:
	if world != null:
		host.remove_child(world)
		world.free()
		world = null
	await host.get_tree().physics_frame


## Advance `ticks` physics frames.
func step(ticks: int) -> void:
	for _i: int in range(ticks):
		await host.get_tree().physics_frame


func row(name: String, ok: Variant, detail: String) -> Array:
	return [name, ok, detail]
