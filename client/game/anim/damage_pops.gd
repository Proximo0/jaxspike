class_name DamagePops
extends RefCounted
## Floating damage numbers (round2 contract 1.3). Pure, in server ms (RenderClock.render_ms).
## A number queued with `add` shows from `show_ms` (the attacker's atk frame, or at once when no attacker is
## visible) and floats up for LIFE_MS, fading over the last FADE_MS.

const LIFE_MS: float = 900.0
const FADE_MS: float = 300.0
const RISE_TILES: float = 1.2
const ATTACKER_RANGE: float = 10.0  ## cells; bows reach 9
const MAX_POPS: int = 256

var _pops: Array[Dictionary] = []  ## {id, dmg, show_ms, seq, crit}
var _seq: int = 0


func add(target_id: int, dmg: int, show_ms: float, crit: bool = false) -> void:
	_seq += 1
	_pops.append({"id": target_id, "dmg": dmg, "show_ms": show_ms, "seq": _seq, "crit": crit})
	if _pops.size() > MAX_POPS:
		_pops.remove_at(0)


## Visible numbers at `now_ms`: {id, dmg, seq, crit, rise (tiles above the head), alpha}. Drops expired ones.
func active(now_ms: float) -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	var keep: Array[Dictionary] = []
	for p: Dictionary in _pops:
		var age: float = now_ms - float(p["show_ms"])
		if age >= LIFE_MS:
			continue
		keep.append(p)
		if age < 0.0:
			continue
		var t: float = age / LIFE_MS
		var alpha: float = clampf((LIFE_MS - age) / FADE_MS, 0.0, 1.0)
		out.append({"id": p["id"], "dmg": p["dmg"], "seq": p["seq"], "crit": p["crit"], "rise": RISE_TILES * (1.0 - (1.0 - t) * (1.0 - t)),
				"alpha": alpha})
	_pops = keep
	return out


func pending() -> int:
	return _pops.size()


func clear() -> void:
	_pops.clear()


## Picks the attacker of a hit on a target at `target_cell` from candidates {id, cell (Vector2), start_ms, dur_ms}
## that are attacking: the nearest one within ATTACKER_RANGE whose swing started in [hit_ms - dur_ms, hit_ms].
## Returns the candidate, or {} when none fits (the number then shows at once).
static func pick_attacker(target_cell: Vector2, hit_ms: float, candidates: Array[Dictionary]) -> Dictionary:
	var best: Dictionary = {}
	var best_d: float = INF
	for c: Dictionary in candidates:
		var start: float = float(c["start_ms"])
		var dur: float = maxf(float(c["dur_ms"]), 1.0)
		if start > hit_ms + 0.5 or start < hit_ms - dur:
			continue
		var d: float = (c["cell"] as Vector2).distance_to(target_cell)
		if d <= ATTACKER_RANGE and d < best_d:
			best_d = d
			best = c
	return best
