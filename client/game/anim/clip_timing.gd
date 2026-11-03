class_name ClipTiming
extends RefCounted
## Frame of a clip at a given time (round2 contract 1.3). Pure: elapsed_ms is (render server ms - action start ms).
## - walk: one cycle of `frames` = WALK_CELLS cells of travel at speed_ms (so the feet match the ground speed);
## - other loops (idle, sit, standby, cast): frame delay from the sheet;
## - one-shots are fitted into dur_ms (attack: the atk frame starts at ATK_FRACTION of amotion);
##   a one-shot with dur_ms 0 uses frames x delay_ms; hurt defaults to HURT_MS.
## frame_at returns -1 once a one-shot (attack, hurt, pickup) is over, so the caller can show the follow-up
## clip (standby after attack, idle after hurt); dead holds its last frame.

const WALK_CELLS: float = 2.0
const ATK_FRACTION: float = 0.6
const HURT_MS: float = 300.0


## Length (ms) a one-shot plays for.
static func one_shot_ms(action: int, frames: int, delay_ms: float, dur_ms: float) -> float:
	if dur_ms > 0.0:
		return dur_ms
	if ClipSelect.clamp_action(action) == ClipSelect.HURT:
		return HURT_MS
	return float(maxi(frames, 1)) * maxf(delay_ms, 1.0)


## Time (ms after the start) the atk frame begins: ATK_FRACTION of the clip.
static func atk_ms(action: int, frames: int, delay_ms: float, dur_ms: float) -> float:
	return one_shot_ms(action, frames, delay_ms, dur_ms) * ATK_FRACTION


## Frame index to draw, or -1 when a one-shot is finished. `atk_frame` < 0 means frames are evenly spread.
static func frame_at(action: int, elapsed_ms: float, frames: int, delay_ms: float, dur_ms: float,
		speed_ms: float, atk_frame: int = -1) -> int:
	if frames <= 1:
		return 0 if (ClipSelect.loops(action) or ClipSelect.holds_last(action)
				or elapsed_ms < one_shot_ms(action, frames, delay_ms, dur_ms)) else -1
	var e: float = maxf(elapsed_ms, 0.0)
	var a: int = ClipSelect.clamp_action(action)
	if a == ClipSelect.WALK:
		var frame_ms: float = maxf(speed_ms, 1.0) * WALK_CELLS / float(frames)
		return int(floorf(e / frame_ms)) % frames
	if ClipSelect.loops(a):
		return int(floorf(e / maxf(delay_ms, 1.0))) % frames
	var total: float = one_shot_ms(a, frames, delay_ms, dur_ms)
	if e >= total:
		return frames - 1 if ClipSelect.holds_last(a) else -1
	if a == ClipSelect.ATTACK and atk_frame > 0 and atk_frame < frames:
		var hit: float = total * ATK_FRACTION
		if e < hit:
			return mini(int(floorf(e / hit * float(atk_frame))), atk_frame - 1)
		return mini(atk_frame + int(floorf((e - hit) / (total - hit) * float(frames - atk_frame))), frames - 1)
	return mini(int(floorf(e / total * float(frames))), frames - 1)


## Action to show after a finished one-shot: standby after attack (RO), idle otherwise.
static func follow_up(action: int) -> int:
	return ClipSelect.STANDBY if ClipSelect.clamp_action(action) == ClipSelect.ATTACK else ClipSelect.IDLE
