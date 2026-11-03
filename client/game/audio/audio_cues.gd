class_name AudioCues
extends RefCounted
## Which feel sounds a new server tick calls for, from the own `you` row and the own entity row. Pure: feed every
## new snap with `observe`; it returns cue names (WALK, ATTACK, HIT, LEVEL_UP, PICKUP, WARP) to play this frame.
## The first observation only primes the state (no sounds on login).

const WALK: String = "walk"
const ATTACK: String = "attack"
const HIT: String = "hit"
const LEVEL_UP: String = "level_up"
const PICKUP: String = "pickup"
const WARP: String = "warp"
const ALL: Array[String] = [WALK, ATTACK, HIT, LEVEL_UP, PICKUP, WARP]
const STEPS_PER_FOOTSTEP: int = 2  ## One footstep every two cells walked (a walk cycle is two cells).
const WARP_JUMP_TILES: int = 3  ## A position jump this long in one tick is a warp, not a step.
const ACTION_ATTACK: int = 5  ## world["anim"].action of an attack swing (round2 contract 1.2).

var primed: bool = false
var _pos: Vector2i = Vector2i.ZERO
var _hp: int = 0
var _base: int = 0
var _job: int = 0
var _items: int = 0
var _swing: int = -1  ## start tick of the last attack swing seen
var _steps: int = 0


## `you` = store.you, `own` = the own entity row (store.get_ent(slot), may be {}). Returns the cues to play.
func observe(you: Dictionary, own: Dictionary) -> Array[String]:
	var out: Array[String] = []
	if you.is_empty():
		return out
	var pos: Vector2i = Vector2i(ViewMath.to_vec(you.get("pos")))
	var hp: int = int(you.get("hp", 0))
	var base: int = int(you.get("level", 0))
	var job: int = int(you.get("job_level", 0))
	var items: int = item_total(you.get("inv"))
	var swing: int = int(own.get("t", -1)) if int(own.get("a", -1)) == ACTION_ATTACK else _swing
	if primed:
		var jump: int = maxi(absi(pos.x - _pos.x), absi(pos.y - _pos.y))
		if jump >= WARP_JUMP_TILES:
			out.append(WARP)
			_steps = 0
		elif jump > 0:
			_steps += 1
			if _steps >= STEPS_PER_FOOTSTEP:
				_steps = 0
				out.append(WALK)
		if swing != _swing and swing >= 0:
			out.append(ATTACK)
		if hp < _hp and hp > 0:
			out.append(HIT)
		if base > _base or job > _job:
			out.append(LEVEL_UP)
		if items > _items:
			out.append(PICKUP)
	primed = true
	_pos = pos
	_hp = hp
	_base = base
	_job = job
	_items = items
	_swing = swing
	return out


func reset() -> void:
	primed = false
	_steps = 0
	_swing = -1


## Total item count of `you.inv`: RO rows ({item, count}) sum their counts, v1 id lists count non-zero ids.
static func item_total(inv: Variant) -> int:
	if typeof(inv) != TYPE_ARRAY:
		return 0
	var n: int = 0
	for row: Variant in inv as Array:
		if typeof(row) == TYPE_DICTIONARY:
			if int((row as Dictionary).get("item", 0)) > 0:
				n += maxi(int((row as Dictionary).get("count", 1)), 1)
		elif int(row) > 0:
			n += 1
	return n
