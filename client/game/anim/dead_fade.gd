class_name DeadFade
extends RefCounted
## Alpha of a dead entity: the dead clip plays, the corpse holds HOLD_MS, then it fades out over FADE_MS.
## Pure: `elapsed_ms` is the time since the dead action began, `clip_ms` the length of the dead clip.

const HOLD_MS: float = 1000.0
const FADE_MS: float = 1000.0


static func alpha(elapsed_ms: float, clip_ms: float) -> float:
	var fade_start: float = maxf(clip_ms, 0.0) + HOLD_MS
	if elapsed_ms < fade_start:
		return 1.0
	return clampf(1.0 - (elapsed_ms - fade_start) / FADE_MS, 0.0, 1.0)
