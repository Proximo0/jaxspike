extends TestCase
## Minimap pure helpers: stride, image bytes from terrain colours, markers from welcome.world and the scan fallback.


func test_stride() -> void:
	assert_eq(Minimap.stride_for(512), 1)
	assert_eq(Minimap.stride_for(2048), 4)
	assert_eq(Minimap.stride_for(100), 1)


func test_image_bytes() -> void:
	var m := MapData.new()
	m.size = 4
	m.terrain = PackedByteArray([0, 1, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9])
	m.pz = PackedByteArray()
	m.pz.resize(16)
	var b: PackedByteArray = Minimap.image_bytes(m, 1)
	assert_eq(b.size(), 48)
	var water: Color = ViewMath.terrain_color(4)
	assert_eq(b[6], water.r8)
	assert_eq(b[8], water.b8)
	var half: PackedByteArray = Minimap.image_bytes(m, 2)
	assert_eq(half.size(), 12, "2x2 samples")
	var markers: Array[Dictionary] = Minimap.scan_markers(m)
	assert_eq(markers.size(), 2)
	assert_eq(markers[0]["pos"], Vector2i(3, 0))
	assert_eq(markers[1]["kind"], "kafra")


func test_world_markers() -> void:
	var w: Dictionary = {"warps": [{"from": [256, 77], "to": [256, 71]}], "towns": [{"name": "Prontera", "kafra": [264, 256]}]}
	var mk: Array[Dictionary] = Minimap.world_markers(w)
	assert_eq(mk.size(), 2)
	assert_eq(mk[0], {"pos": Vector2i(256, 77), "kind": "warp"})
	assert_eq(mk[1], {"pos": Vector2i(264, 256), "kind": "kafra"})
	assert_true(Minimap.world_markers({}).is_empty())
