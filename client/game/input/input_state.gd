class_name InputState
extends RefCounted
## Held-key state -> the (move, attack, eat) triple of `send_input`. `update` returns true when the triple changed
## and must be sent. Eat is level-triggered: a tap sends eat=true once on press and eat=false on release.

const MOVE_NONE: int = 0
const MOVE_NORTH: int = 1
const MOVE_SOUTH: int = 2
const MOVE_WEST: int = 3
const MOVE_EAST: int = 4

var move: int = MOVE_NONE
var attack: bool = false
var eat: bool = false
var _order: Array[int] = []  ## held directions, oldest first


## `up/down/left/right/attack_held/eat_held` are the current key states.
func update(up: bool, down: bool, left: bool, right: bool, attack_held: bool, eat_held: bool) -> bool:
	var held: Array[bool] = [false, up, down, left, right]
	var kept: Array[int] = []
	for d: int in _order:
		if held[d]:
			kept.append(d)
	for d: int in range(1, 5):
		if held[d] and not kept.has(d):
			kept.append(d)
	_order = kept
	# The newest held direction wins; a lone opposite pair cancels.
	var m: int = MOVE_NONE
	if not _order.is_empty():
		m = _order[_order.size() - 1]
	if up and down and not (left or right):
		m = MOVE_NONE
	elif left and right and not (up or down):
		m = MOVE_NONE
	var changed: bool = m != move or attack_held != attack or eat_held != eat
	move = m
	attack = attack_held
	eat = eat_held
	return changed


func reset() -> void:
	move = MOVE_NONE
	attack = false
	eat = false
	_order.clear()
