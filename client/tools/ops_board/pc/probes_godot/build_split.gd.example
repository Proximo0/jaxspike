extends Node
## Temporary (untracked): times the parts of Match._build separately, 4 rounds, to attribute the 300 ms match build.

func _ready() -> void:
	GfxQuality.set_level(2)
	for round: int in 4:
		var cfg: Dictionary = MatchConfig.normalize({"mode": "bot_vs_bot", "seed": 5000 + round}, 1)
		var holder := Node3D.new()
		add_child(holder)
		var t0: int = Time.get_ticks_usec()
		var p := MatchPresenter.new()
		holder.add_child(p)
		var t1: int = Time.get_ticks_usec()
		p.build("wide")
		var t2: int = Time.get_ticks_usec()
		var fs: Array = []
		var fighter_ms: Array = []
		for side: int in 2:
			var a: int = Time.get_ticks_usec()
			var f := Fighter.new()
			holder.add_child(f)
			f.global_transform = Arena.spawn_transform(side, Tuning.shared().arena_radius)
			f.look_seed = 5000 + round
			f.setup((cfg["loadouts"] as Array)[side], side, Tuning.shared())
			fs.append(f)
			fighter_ms.append("%.1f" % ((Time.get_ticks_usec() - a) / 1000.0))
		(fs[0] as Fighter).opponent = fs[1]
		(fs[1] as Fighter).opponent = fs[0]
		var t3: int = Time.get_ticks_usec()
		p.begin(fs, ["A", "B"], 0)
		var t4: int = Time.get_ticks_usec()
		print("SPLIT round=%d presenter_add=%.1f presenter_build=%.1f fighters=%s begin=%.1f total=%.1f nodes=%d" % [round,
				(t1 - t0) / 1000.0, (t2 - t1) / 1000.0, ",".join(PackedStringArray(fighter_ms)), (t4 - t3) / 1000.0,
				(t4 - t0) / 1000.0, int(Performance.get_monitor(Performance.OBJECT_NODE_COUNT))])
		# Break down the presenter build by child: time to construct is not visible, but node counts per child are.
		if round == 0:
			for c: Node in p.get_children():
				print("SPLIT child %s nodes=%d" % [c.name, _count(c)])
		await get_tree().process_frame
		holder.queue_free()
		await get_tree().process_frame
	get_tree().quit()


func _count(n: Node) -> int:
	var k: int = 1
	for c: Node in n.get_children():
		k += _count(c)
	return k
