class_name PathPreview
extends RefCounted
## Display-only click-to-walk path: 8-connected A* over MapData with RO's 10/14 step costs and no corner cutting
## (a diagonal needs both orthogonal neighbours walkable, as the server's pathing). The server walks its own path;
## this only draws the dotted line under the walk marker, so it is capped (MAX_CELLS cells, MAX_EXPANDED nodes).

const MAX_CELLS: int = 64
const MAX_EXPANDED: int = 4096
const COST_STRAIGHT: int = 10
const COST_DIAGONAL: int = 14
## RO direction order (0 S, 1 SW, 2 W, 3 NW, 4 N, 5 NE, 6 E, 7 SE), y grows south.
const DIRS: Array[Vector2i] = [
	Vector2i(0, 1), Vector2i(-1, 1), Vector2i(-1, 0), Vector2i(-1, -1),
	Vector2i(0, -1), Vector2i(1, -1), Vector2i(1, 0), Vector2i(1, 1),
]


static func walkable(map: MapData, c: Vector2i) -> bool:
	if map == null or not map.in_bounds(c.x, c.y):
		return false
	return not ViewMath.is_blocked_terrain(map.at(c.x, c.y))


## True when a step from `c` by `d` is allowed: target walkable and, for a diagonal, both side cells walkable.
static func can_step(map: MapData, c: Vector2i, d: Vector2i) -> bool:
	if not walkable(map, c + d):
		return false
	if d.x != 0 and d.y != 0:
		return walkable(map, Vector2i(c.x + d.x, c.y)) and walkable(map, Vector2i(c.x, c.y + d.y))
	return true


## Octile distance in cost units.
static func heuristic(a: Vector2i, b: Vector2i) -> int:
	var dx: int = absi(a.x - b.x)
	var dy: int = absi(a.y - b.y)
	return COST_STRAIGHT * maxi(dx, dy) + (COST_DIAGONAL - COST_STRAIGHT) * mini(dx, dy)


## Cells from `start` (excluded) to `goal` (included), at most `max_cells` long (the first ones when the path is
## longer). Empty when start == goal, the goal is blocked or no path is found within MAX_EXPANDED expansions.
static func find(map: MapData, start: Vector2i, goal: Vector2i, max_cells: int = MAX_CELLS) -> Array[Vector2i]:
	var out: Array[Vector2i] = []
	if start == goal or not walkable(map, goal) or not map.in_bounds(start.x, start.y):
		return out
	var g: Dictionary = {start: 0}
	var came: Dictionary = {}
	var closed: Dictionary = {}
	var heap: Array = []  ## [f, seq, cell] min-heap; seq keeps ties in insertion order
	var seq: int = 0
	_path_preview_push(heap, [heuristic(start, goal), seq, start])
	var expanded: int = 0
	while not heap.is_empty() and expanded < MAX_EXPANDED:
		var top: Array = _path_preview_pop(heap)
		var c: Vector2i = top[2]
		if closed.has(c):
			continue
		if c == goal:
			var rev: Array[Vector2i] = []
			var k: Vector2i = goal
			while k != start:
				rev.append(k)
				k = came[k]
			rev.reverse()
			return rev.slice(0, max_cells)
		closed[c] = true
		expanded += 1
		var gc: int = int(g[c])
		for i: int in range(DIRS.size()):
			var d: Vector2i = DIRS[i]
			if not can_step(map, c, d):
				continue
			var n: Vector2i = c + d
			if closed.has(n):
				continue
			var cost: int = gc + (COST_DIAGONAL if (d.x != 0 and d.y != 0) else COST_STRAIGHT)
			if not g.has(n) or cost < int(g[n]):
				g[n] = cost
				came[n] = c
				seq += 1
				_path_preview_push(heap, [cost + heuristic(n, goal), seq, n])
	return out


## Drops the path's leading cells up to and including `own_pos` (the player walked them). Unchanged when the
## player is not on the path.
static func trim(path: Array[Vector2i], own_pos: Vector2i) -> Array[Vector2i]:
	var i: int = path.find(own_pos)
	if i < 0:
		return path
	return path.slice(i + 1)


static func _path_preview_less(a: Array, b: Array) -> bool:
	return int(a[0]) < int(b[0]) or (int(a[0]) == int(b[0]) and int(a[1]) < int(b[1]))


static func _path_preview_push(heap: Array, item: Array) -> void:
	heap.append(item)
	var i: int = heap.size() - 1
	while i > 0:
		var p: int = (i - 1) >> 1
		if not _path_preview_less(heap[i], heap[p]):
			break
		var t: Array = heap[i]
		heap[i] = heap[p]
		heap[p] = t
		i = p


static func _path_preview_pop(heap: Array) -> Array:
	var top: Array = heap[0]
	var last: Array = heap.pop_back()
	if heap.is_empty():
		return top
	heap[0] = last
	var i: int = 0
	var n: int = heap.size()
	while true:
		var l: int = i * 2 + 1
		var r: int = l + 1
		var m: int = i
		if l < n and _path_preview_less(heap[l], heap[m]):
			m = l
		if r < n and _path_preview_less(heap[r], heap[m]):
			m = r
		if m == i:
			break
		var t: Array = heap[i]
		heap[i] = heap[m]
		heap[m] = t
		i = m
	return top
