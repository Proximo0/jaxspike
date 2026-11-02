extends Node3D
## Starter main scene: a lit floor and a box, plus the debug layer (F10 frame stamp, F4/F6-F9 render toggles).
## Replace the contents; keep the two debug nodes.


func _ready() -> void:
	add_child(FrameStamp.new())
	add_child(RenderToggles.new())
