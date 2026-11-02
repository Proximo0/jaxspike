class_name SlotPool
extends RefCounted
## Bookkeeping of a node pool: entity id -> slot index, with freed slots reused first (lowest index first).

var _slot_of: Dictionary = {}  ## id -> slot
var _free: PackedInt32Array = PackedInt32Array()
var _capacity: int = 0


## The slot for `id` (existing or newly assigned).
func acquire(id: int) -> int:
	if _slot_of.has(id):
		return int(_slot_of[id])
	var slot: int
	if _free.is_empty():
		slot = _capacity
		_capacity += 1
	else:
		var best: int = 0
		for i: int in range(1, _free.size()):
			if _free[i] < _free[best]:
				best = i
		slot = _free[best]
		_free.remove_at(best)
	_slot_of[id] = slot
	return slot


## Frees the slot of `id`. Returns it, or -1 when the id had none.
func release(id: int) -> int:
	if not _slot_of.has(id):
		return -1
	var slot: int = int(_slot_of[id])
	_slot_of.erase(id)
	_free.append(slot)
	return slot


func has_id(id: int) -> bool:
	return _slot_of.has(id)


func slot_of(id: int) -> int:
	return int(_slot_of.get(id, -1))


func active_count() -> int:
	return _slot_of.size()


## Total slots ever created (pool size).
func capacity() -> int:
	return _capacity


func ids() -> PackedInt32Array:
	var out := PackedInt32Array()
	for k: Variant in _slot_of:
		out.append(int(k))
	return out
