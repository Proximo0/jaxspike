class_name ClipSelect
extends RefCounted
## Which clip to play for an entity's action + face (round2 contract 0 and 1.2). Pure tables, no state.
## Engine actions: 0 idle, 1 walk, 2 sit, 3 pickup, 4 standby, 5 attack, 6 hurt, 7 dead, 8 cast.
## Faces: 0 S, then clockwise 1 SW, 2 W, 3 NW, 4 N, 5 NE, 6 E, 7 SE; RO draws 0..4 and mirrors 5..7 from 3..1.

const IDLE: int = 0
const WALK: int = 1
const SIT: int = 2
const PICKUP: int = 3
const STANDBY: int = 4
const ATTACK: int = 5
const HURT: int = 6
const DEAD: int = 7
const CAST: int = 8
const ACTION_COUNT: int = 9

const NAMES: PackedStringArray = ["idle", "walk", "sit", "pickup", "standby", "attack", "hurt", "dead", "cast"]
## RO ACT action slots (first direction) per engine action. Players: idle 0, walk 8, sit 16, pick up 24,
## standby 32, attack 40, hurt 48, dead 64, cast 96. Monsters have idle 0, walk 8, attack 16, hurt 24, dead 32:
## sit/standby fall back to idle, pickup and cast to attack (RO monsters loot with their attack clip).
const PLAYER_SLOTS: PackedInt32Array = [0, 8, 16, 24, 32, 40, 48, 64, 96]
const MONSTER_SLOTS: PackedInt32Array = [0, 8, 0, 16, 0, 16, 24, 32, 16]
const MONSTER_CLIP: PackedInt32Array = [IDLE, WALK, IDLE, ATTACK, IDLE, ATTACK, HURT, DEAD, ATTACK]
## Clip a missing one falls back to (sheet without that action): sit/standby/cast/pickup -> idle or attack.
const FALLBACK: PackedInt32Array = [IDLE, IDLE, IDLE, ATTACK, IDLE, IDLE, IDLE, HURT, ATTACK]
## Unit cell deltas per face (x right, y down = south).
const FACE_DELTAS: Array[Vector2i] = [Vector2i(0, 1), Vector2i(-1, 1), Vector2i(-1, 0), Vector2i(-1, -1),
		Vector2i(0, -1), Vector2i(1, -1), Vector2i(1, 0), Vector2i(1, 1)]


static func clamp_action(action: int) -> int:
	return action if action >= 0 and action < ACTION_COUNT else IDLE


static func clamp_face(face: int) -> int:
	return posmod(face, 8)


## RO ACT index (slot + direction) for an action and face.
static func act_index(action: int, face: int, is_player: bool) -> int:
	var a: int = clamp_action(action)
	return (PLAYER_SLOTS[a] if is_player else MONSTER_SLOTS[a]) + clamp_face(face)


## Engine action the clip is drawn with (monsters reuse attack/idle for actions they have no clip for).
static func clip_action(action: int, is_player: bool) -> int:
	var a: int = clamp_action(action)
	return a if is_player else MONSTER_CLIP[a]


static func clip_name(action: int, is_player: bool) -> String:
	return NAMES[clip_action(action, is_player)]


static func fallback(action: int) -> int:
	return FALLBACK[clamp_action(action)]


## Sheet row and horizontal flip for a face: rows 5..7 mirror rows 3..1 when the sheet has only 5 rows.
static func sheet_row(face: int, rows: int) -> Vector2i:
	var f: int = clamp_face(face)
	if f <= 4 or rows >= 8:
		return Vector2i(f, 0)
	return Vector2i(8 - f, 1)  # y = 1 means flip_h


## Face of a cell delta (largest-axis sign), or `keep` for a zero delta.
static func face_of(delta: Vector2, keep: int = 0) -> int:
	var s := Vector2i(signi(roundi(delta.x)), signi(roundi(delta.y)))
	if s == Vector2i.ZERO:
		return keep
	return FACE_DELTAS.find(s)


## True for clips that loop (idle, walk, sit, standby, cast); attack, hurt, pickup and dead play once.
static func loops(action: int) -> bool:
	var a: int = clamp_action(action)
	return a == IDLE or a == WALK or a == SIT or a == STANDBY or a == CAST


## True for one-shot clips that hold their last frame afterwards (dead, sit-like poses).
static func holds_last(action: int) -> bool:
	return clamp_action(action) == DEAD
