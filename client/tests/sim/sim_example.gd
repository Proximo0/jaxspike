extends "res://tests/sim/sim_base.gd"
## Example sim: a ball dropped on a floor comes to rest on top of it. Copy this to start a real sim.


func run() -> Array:
	await sandbox()
	var floor_body := StaticBody3D.new()
	var floor_shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = Vector3(20, 1, 20)
	floor_shape.shape = box
	floor_body.add_child(floor_shape)
	floor_body.position = Vector3(0, -0.5, 0)
	world.add_child(floor_body)
	var ball := RigidBody3D.new()
	var ball_shape := CollisionShape3D.new()
	var sphere := SphereShape3D.new()
	sphere.radius = 0.5
	ball_shape.shape = sphere
	ball.add_child(ball_shape)
	ball.position = Vector3(0, 3, 0)
	world.add_child(ball)
	await step(180)
	var y: float = ball.position.y
	var rows: Array = [row("example_ball_rests", absf(y - 0.5) < 0.05, "y=%.3f" % y)]
	await clear()
	return rows
