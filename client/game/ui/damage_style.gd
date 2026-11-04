class_name DamageStyle
extends RefCounted
## RO damage number look: white normal hits, yellow crits, red damage on the own character, "Miss" in grey.
## Pure statics; DamageLayer draws with these, DamagePops' timing curve is mirrored here.

enum { NORMAL, CRIT, OWN, MISS }

const COLOR_NORMAL: Color = Color(1.0, 1.0, 1.0)
const COLOR_CRIT: Color = Color(1.0, 0.9, 0.2)
const COLOR_OWN: Color = Color(1.0, 0.35, 0.3)
const COLOR_MISS: Color = Color(0.92, 0.92, 0.94)
const FONT_NORMAL_PX: float = 20.0
const FONT_CRIT_PX: float = 26.0
const OUTLINE_PX: int = 5
const RISE_TILES: float = 1.2
const DRIFT_PX: float = 6.0
const DRIFT_STEPS: int = 5  ## offsets -6, -3, 0, 3, 6 by sequence number


static func kind_of(dmg: int, own: bool, crit: bool) -> int:
	if dmg == 0:
		return MISS
	if own:
		return OWN
	return CRIT if crit else NORMAL


static func color_of(kind: int) -> Color:
	match kind:
		CRIT:
			return COLOR_CRIT
		OWN:
			return COLOR_OWN
		MISS:
			return COLOR_MISS
	return COLOR_NORMAL


static func text_of(dmg: int, kind: int) -> String:
	if kind == MISS:
		return "Miss"
	return str(absi(dmg))


static func font_px(kind: int) -> float:
	return FONT_CRIT_PX if kind == CRIT else FONT_NORMAL_PX


static func outline_px() -> int:
	return OUTLINE_PX


## Ease-out rise in tiles: 0 at age 0, RISE_TILES at the end of life (same curve as DamagePops).
static func rise_tiles(age_ms: float, life_ms: float) -> float:
	var t: float = clampf(age_ms / maxf(life_ms, 1.0), 0.0, 1.0)
	return RISE_TILES * (1.0 - (1.0 - t) * (1.0 - t))


## 1 until the last fade_ms of life, then down to 0 at life_ms.
static func alpha(age_ms: float, life_ms: float, fade_ms: float) -> float:
	return clampf((life_ms - age_ms) / maxf(fade_ms, 1.0), 0.0, 1.0)


## Deterministic horizontal offset in px per sequence number so stacked numbers do not overprint; 0 for MISS.
static func drift_px(seq: int, kind: int) -> float:
	if kind == MISS:
		return 0.0
	var step: int = posmod(seq * 3, DRIFT_STEPS)  ## 3 is coprime to 5: consecutive seqs jump around
	return (float(step) - float(DRIFT_STEPS / 2)) * (DRIFT_PX / float(DRIFT_STEPS / 2))
