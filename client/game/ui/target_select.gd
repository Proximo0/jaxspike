class_name TargetSelect
extends RefCounted
## Currently selected entity (left click) and the skill-target rule. Pure: positions are passed in by the caller.

const NO_TARGET: int = -1
const NEEDS_TARGET: int = -2  ## resolve() result: the skill needs a target and none is selected.

var selected_id: int = NO_TARGET


func select(id: int) -> void:
	selected_id = id


func clear() -> void:
	selected_id = NO_TARGET


## Clears the selection when the entity is no longer known (`alive` = Callable(id) -> bool).
func prune(alive: Callable) -> void:
	if selected_id != NO_TARGET and not bool(alive.call(selected_id)):
		selected_id = NO_TARGET


## Selects the entity nearest to `point` within `max_dist` among `positions` (id -> Vector2); returns the id or -1
## (and clears the selection). `skip` ids (own player, drops) are ignored. Ties go to the lower id.
func pick(point: Vector2, positions: Dictionary, max_dist: float, skip: Array = []) -> int:
	var best: int = NO_TARGET
	var best_d: float = max_dist
	var ids: Array = positions.keys()
	ids.sort()
	for id: Variant in ids:
		if skip.has(id):
			continue
		var d: float = point.distance_to(positions[id] as Vector2)
		if d <= best_d and (best == NO_TARGET or d < best_d):
			best = int(id)
			best_d = d
	selected_id = best
	return best


## The `target` to send for a skill of `target_type`: self -> -1; ally -> selected or own id; enemy/ground ->
## selected, or NEEDS_TARGET when nothing is selected.
func resolve(target_type: int, own_id: int) -> int:
	match target_type:
		UiStatsCalc.TARGET_SELF:
			return NO_TARGET
		UiStatsCalc.TARGET_ALLY:
			return selected_id if selected_id != NO_TARGET else own_id
		_:
			return selected_id if selected_id != NO_TARGET else NEEDS_TARGET
