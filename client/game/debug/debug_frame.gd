class_name DebugFrame
extends RefCounted
## Debug frame: a ring of the last RING_SIZE frames of client state, plus pure frame-time maths.
## Usage per rendered/tick frame: begin(tick), note(key, value) any number of times, end().
## `snapshot()` / `restore(d)` round-trip the ring and, when a store is attached, the store's own
## snapshot (EntityStore.snapshot/restore, duck-typed so this file does not depend on Part C).

const RING_SIZE: int = 600
const VERSION: int = 1

var _ring: Array[Dictionary] = []
var _head: int = 0  ## index of the oldest entry once the ring is full
var _count: int = 0  ## frames ended since creation (or restored count)
var _open: Dictionary = {}
var _has_open: bool = false
var _store: Object = null


## Attach anything with snapshot() -> Dictionary and restore(Dictionary) (an EntityStore).
func attach_store(store: Object) -> void:
	_store = store


func begin(tick: int) -> void:
	_open = {"tick": tick, "frame": _count, "rx_bytes": 0, "ents": 0, "you_pos": [0, 0], "latency_ms": 0.0}
	_has_open = true


func note(key: String, value: Variant) -> void:
	if not _has_open:
		return
	_open[key] = value


func end() -> void:
	if not _has_open:
		return
	_has_open = false
	if _ring.size() < RING_SIZE:
		_ring.append(_open)
	else:
		_ring[_head] = _open
		_head = (_head + 1) % RING_SIZE
	_open = {}
	_count += 1


func size() -> int:
	return _ring.size()


## Frames ended since creation.
func total_frames() -> int:
	return _count


## The ring in order, oldest first.
func frames() -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	for i: int in range(_ring.size()):
		out.append(_ring[(_head + i) % _ring.size()])
	return out


func last() -> Dictionary:
	if _ring.is_empty():
		return {}
	return _ring[(_head + _ring.size() - 1) % _ring.size()]


func clear() -> void:
	_ring.clear()
	_head = 0
	_count = 0
	_open = {}
	_has_open = false


func snapshot() -> Dictionary:
	var frames_copy: Array = []
	for f: Dictionary in frames():
		frames_copy.append(f.duplicate(true))
	var d: Dictionary = {"version": VERSION, "count": _count, "frames": frames_copy}
	if _store != null and _store.has_method("snapshot"):
		d["store"] = _store.call("snapshot")
	return d


## Puts a snapshot back. Returns false (and changes nothing) for a malformed dictionary.
func restore(d: Dictionary) -> bool:
	if int(d.get("version", 0)) != VERSION or not d.get("frames") is Array:
		return false
	var src: Array = d["frames"]
	clear()
	var first: int = maxi(0, src.size() - RING_SIZE)
	for i: int in range(first, src.size()):
		var f: Dictionary = src[i]
		_ring.append(f.duplicate(true))
	_count = int(d.get("count", src.size()))
	if _store != null and _store.has_method("restore") and d.get("store") is Dictionary:
		_store.call("restore", d["store"])
	return true


## Nearest-rank percentile (p in 0..100) of `values`; 0.0 when empty. Does not modify `values`.
static func percentile(values: PackedFloat64Array, p: float) -> float:
	var n: int = values.size()
	if n == 0:
		return 0.0
	var sorted: PackedFloat64Array = values.duplicate()
	sorted.sort()
	var rank: int = ceili(clampf(p, 0.0, 100.0) / 100.0 * float(n))
	return sorted[clampi(rank - 1, 0, n - 1)]


## {p50, p95, max, n} of frame times in ms.
static func frame_stats(times_ms: PackedFloat64Array) -> Dictionary:
	var mx: float = 0.0
	for t: float in times_ms:
		mx = maxf(mx, t)
	return {"p50": percentile(times_ms, 50.0), "p95": percentile(times_ms, 95.0), "max": mx, "n": times_ms.size()}


## Frames per second implied by a frame time in ms (0 for a non-positive time).
static func fps_from_ms(ms: float) -> float:
	return 1000.0 / ms if ms > 0.0 else 0.0
