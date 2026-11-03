class_name RenderClock
extends RefCounted
## The client's clock in server ticks (round2 contract 1.3): render_tick = newest snap tick - interp_delay.
## Pure: the caller passes the local `now_ms` of each snap arrival and of each frame.
##
## `base_ms` maps local time to server time (server_ms = now_ms - base_ms). It follows the EARLIEST arrival seen
## (an early snap pulls it at once, a late one is jitter), creeps later by DRIFT_PER_S so a slower server clock is
## followed, and resets when snaps keep arriving more than RESET_TICKS late (pause, reconnect).
## On top of delay_ticks the clock keeps a jitter margin: the recent peak lateness of arrivals (decaying by
## MARGIN_DECAY per snap), approached at most MARGIN_SLEW ms per ms so the drawn time never jumps. Without it a
## snap that arrives late stalls the clock at the newest tick and the next one makes everything jump.

const DRIFT_PER_S: float = 2.0  ## ms per second the estimate creeps toward later arrivals
const RESET_TICKS: float = 4.0  ## an arrival this many ticks later than expected re-anchors the clock
const MARGIN_DECAY: float = 0.995  ## per snap: the peak lateness is forgotten over a few hundred snaps
const MARGIN_SLEW: float = 0.05  ## the margin moves at most 5% of elapsed time (the clock runs 0.95x..1.05x)
const MARGIN_MAX_TICKS: float = 2.0

var tick_ms: float = 120.0
var delay_ticks: float = 1.0
var newest_tick: int = -1
var base_ms: float = 0.0
var margin_ms: float = 0.0  ## jitter margin in use now
var peak_late_ms: float = 0.0  ## margin being approached
var _last_obs_ms: float = 0.0
var _last_now_ms: float = 0.0
var _seen: bool = false


func _init(new_tick_ms: float = 120.0, new_delay_ticks: float = 1.0) -> void:
	tick_ms = new_tick_ms
	delay_ticks = new_delay_ticks


## Records that the snap for `server_tick` arrived at local time `now_ms`.
func observe(server_tick: int, now_ms: float) -> void:
	var offset: float = now_ms - float(server_tick) * tick_ms
	if not _seen:
		_seen = true
		base_ms = offset
		_last_now_ms = now_ms
	else:
		base_ms += maxf(now_ms - _last_obs_ms, 0.0) * DRIFT_PER_S / 1000.0
		if offset - base_ms > RESET_TICKS * tick_ms:
			base_ms = offset
			margin_ms = 0.0
			peak_late_ms = 0.0
		elif offset < base_ms:
			base_ms = offset
	peak_late_ms = minf(maxf(offset - base_ms, peak_late_ms * MARGIN_DECAY), MARGIN_MAX_TICKS * tick_ms)
	_last_obs_ms = now_ms
	newest_tick = maxi(newest_tick, server_tick)


func has_time() -> bool:
	return _seen


## Server time (ms) being drawn at local `now_ms`: delayed by delay_ticks plus the jitter margin, never past the
## newest tick. Call with non-decreasing `now_ms` (it slews the margin).
func render_ms(now_ms: float) -> float:
	if not _seen:
		return now_ms
	var dt: float = maxf(now_ms - _last_now_ms, 0.0)
	_last_now_ms = maxf(now_ms, _last_now_ms)
	var step: float = dt * MARGIN_SLEW
	margin_ms += clampf(peak_late_ms - margin_ms, -step, step)
	var ms: float = now_ms - base_ms - delay_ticks * tick_ms - margin_ms
	return minf(ms, float(newest_tick) * tick_ms)


func render_tick(now_ms: float) -> float:
	return render_ms(now_ms) / tick_ms


## Ticks the render clock lags behind where it should be (newest - delay); positive = behind.
func drift_ticks(now_ms: float) -> float:
	return float(newest_tick) - delay_ticks - render_tick(now_ms)
