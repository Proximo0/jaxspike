class_name UiStatsCalc
extends RefCounted
## Pure maths and selection rules behind the RO character UI (status panel, skill bar, skill panel, buffs).
## Inputs are the decoded `welcome` dictionary and the `snap.you` dictionary (docs/stats_wire.md, skills_wire.md).

const STAT_CAP: int = 99
const NOVICE_JOB_LEVEL: int = 10  ## Job level a Novice needs before a first-job change.
const BAR_SLOTS: int = 9
const EFFECT_PASSIVE: int = 0
const TARGET_SELF: int = 0
const TARGET_ALLY: int = 1
const TARGET_ENEMY: int = 2
const TARGET_GROUND: int = 3


## id -> skill definition dictionary, from `welcome.skills.list` (null entries skipped).
static func skill_index(welcome: Dictionary) -> Dictionary:
	var out: Dictionary = {}
	var skills: Variant = welcome.get("skills")
	if typeof(skills) != TYPE_DICTIONARY:
		return out
	var list: Variant = (skills as Dictionary).get("list")
	if typeof(list) != TYPE_ARRAY:
		return out
	for d: Variant in list:
		if typeof(d) == TYPE_DICTIONARY and (d as Dictionary).has("id"):
			out[int((d as Dictionary)["id"])] = d
	return out


## Fraction 0..1 of an exp bar. `next <= 0` means the level cap, which draws full.
static func exp_fraction(cur: int, next: int) -> float:
	if next <= 0:
		return 1.0
	return clampf(float(cur) / float(next), 0.0, 1.0)


static func exp_text(cur: int, next: int) -> String:
	if next <= 0:
		return "max"
	return "%d / %d" % [cur, next]


static func bar_text(cur: int, maxv: int) -> String:
	return "%d / %d" % [cur, maxi(maxv, 0)]


static func stat_value(you: Dictionary, i: int) -> int:
	var s: Variant = you.get("stats")
	if typeof(s) == TYPE_ARRAY and i >= 0 and i < (s as Array).size():
		return int((s as Array)[i])
	return 0


## Points the next raise of a stat at `value` costs: `welcome.stat_cost[value]`, or 1 when the table is missing.
static func stat_cost(welcome: Dictionary, value: int) -> int:
	var t: Variant = welcome.get("stat_cost")
	if typeof(t) == TYPE_ARRAY and value >= 0 and value < (t as Array).size():
		return maxi(int((t as Array)[value]), 1)
	return 1


static func can_allot(you: Dictionary, welcome: Dictionary, i: int) -> bool:
	var points: int = int(you.get("stat_points", 0))
	if points <= 0:
		return false
	var v: int = stat_value(you, i)
	if v >= STAT_CAP:
		return false
	return stat_cost(welcome, v) <= points


static func can_change_job(you: Dictionary) -> bool:
	return int(you.get("job", 0)) == 0 and int(you.get("job_level", 1)) >= NOVICE_JOB_LEVEL


## [{job, name}] for jobs 1.. from `welcome.job_names`.
static func job_choices(welcome: Dictionary) -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	var names: Variant = welcome.get("job_names")
	if typeof(names) != TYPE_ARRAY:
		return out
	for j: int in range(1, (names as Array).size()):
		out.append({"job": j, "name": String((names as Array)[j])})
	return out


static func job_name(welcome: Dictionary, job: int) -> String:
	var names: Variant = welcome.get("job_names")
	if typeof(names) == TYPE_ARRAY and job >= 0 and job < (names as Array).size():
		return String((names as Array)[job])
	return "Job %d" % job


## Level entry `levels[level-1]` of a definition (clamped), or {} for an unlearned skill.
static func level_entry(def: Dictionary, level: int) -> Dictionary:
	var lv: Variant = def.get("levels")
	if level <= 0 or typeof(lv) != TYPE_ARRAY or (lv as Array).is_empty():
		return {}
	var e: Variant = (lv as Array)[mini(level, (lv as Array).size()) - 1]
	return e if typeof(e) == TYPE_DICTIONARY else {}


static func learned_level(you: Dictionary, id: int) -> int:
	return _pair_value(you.get("learned"), id)


static func cooldown_left(you: Dictionary, id: int) -> int:
	return _pair_value(you.get("cooldowns"), id)


static func _pair_value(rows: Variant, id: int) -> int:
	if typeof(rows) == TYPE_ARRAY:
		for r: Variant in rows:
			if typeof(r) == TYPE_ARRAY and (r as Array).size() >= 2 and int((r as Array)[0]) == id:
				return int((r as Array)[1])
	return 0


## Remaining fraction (1 just after the cast, 0 ready) of a cooldown.
static func cooldown_fraction(left: int, total: int) -> float:
	if left <= 0 or total <= 0:
		return 0.0
	return clampf(float(left) / float(total), 0.0, 1.0)


## The skills on the bar: learned (level > 0) and castable (effect != passive), ascending id, at most 9.
## Entry: {id, name, level, sp, cooldown, target, cast}.
static func bar_skills(defs: Dictionary, you: Dictionary) -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	var ids: Array = defs.keys()
	ids.sort()
	for id: Variant in ids:
		var lvl: int = learned_level(you, int(id))
		var def: Dictionary = defs[id]
		if lvl <= 0 or int(def.get("effect", 0)) == EFFECT_PASSIVE:
			continue
		var e: Dictionary = level_entry(def, lvl)
		out.append({"id": int(id), "name": String(def.get("name", "")), "level": lvl, "sp": int(e.get("sp", 0)),
				"cooldown": int(e.get("cooldown", 0)), "target": int(def.get("target", TARGET_SELF)),
				"cast": int(e.get("cast", 0))})
		if out.size() >= BAR_SLOTS:
			break
	return out


static func can_afford(you: Dictionary, sp_cost: int) -> bool:
	return int(you.get("sp", 0)) >= sp_cost


## Slot index 0..8 for a bar hotkey (keys 1..9), else -1.
static func slot_for_key(keycode: int) -> int:
	if keycode >= KEY_1 and keycode <= KEY_9:
		return keycode - KEY_1
	return -1


static func can_learn(you: Dictionary, level: int, max_level: int) -> bool:
	if level >= max_level:
		return false
	# `skill_points` is omitted when the server has no stats block: learning is free then.
	return not you.has("skill_points") or int(you["skill_points"]) > 0


## Skills the learn panel lists: Novice (job 0) plus the own job, Novice first, ascending id.
## Entry: {id, name, level, max_level, can_learn, job}.
static func panel_skills(defs: Dictionary, you: Dictionary) -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	var job: int = int(you.get("job", 0))
	var ids: Array = defs.keys()
	ids.sort()
	var jobs: Array = [0] if job == 0 else [0, job]
	for pass_job: int in jobs:
		for id: Variant in ids:
			var def: Dictionary = defs[id]
			if int(def.get("job", 0)) != pass_job:
				continue
			var lvl: int = learned_level(you, int(id))
			var mx: int = int(def.get("max_level", 1))
			out.append({"id": int(id), "name": String(def.get("name", "")), "level": lvl, "max_level": mx,
					"can_learn": can_learn(you, lvl, mx), "job": pass_job})
	return out


## Buff icons: [{id, name, seconds, level}] from `you.buffs` rows [id, ticks_left, level] and `welcome.buffs`.
static func buff_rows(welcome: Dictionary, you: Dictionary, tick_ms: int) -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	var names: Variant = welcome.get("buffs")
	var b: Variant = you.get("buffs")
	if typeof(b) != TYPE_ARRAY:
		return out
	for r: Variant in b:
		if typeof(r) != TYPE_ARRAY or (r as Array).size() < 2:
			continue
		var row: Array = r
		var id: int = int(row[0])
		var ticks: int = int(row[1])
		if ticks <= 0:
			continue
		var nm: String = "Buff %d" % id
		if typeof(names) == TYPE_ARRAY and id >= 0 and id < (names as Array).size() and String((names as Array)[id]) != "":
			nm = String((names as Array)[id])
		out.append({"id": id, "name": nm, "seconds": float(ticks * tick_ms) / 1000.0,
				"level": int(row[2]) if row.size() > 2 else 1})
	return out


## True when a skill may start: no after-cast delay left (the server refuses a skill while `after_delay` > 0; a
## cast in flight is shown by the cast bar and is not a bar grey-out, so sim_ui_stats can press while casting).
static func can_act(you: Dictionary) -> bool:
	return int(you.get("after_delay", 0)) <= 0


## Share of the after-cast delay still to run (1 at its start, 0 when over), `peak` = the largest value seen.
static func after_delay_fraction(you: Dictionary, peak: int) -> float:
	if peak <= 0:
		return 0.0
	return clampf(float(int(you.get("after_delay", 0))) / float(peak), 0.0, 1.0)
