class_name FakeLink
extends NetLink
## Scripted link for tests, the lab and sims. Inbound texts are queued with an optional lag in ticks and released by
## `tick()`; outbound texts are recorded in `sent`.

var sent: Array[String] = []
var _now: int = 0
var _seq: int = 0
var _queue: Array[Dictionary] = []  ## {due, seq, text}
var _state: int = NetLink.STATE_CLOSED


## Queue one inbound text that becomes pollable after `lag_ticks` calls to tick() (0 = at once).
func push_inbound(text: String, lag_ticks: int = 0) -> void:
	_seq += 1
	_queue.append({"due": _now + maxi(lag_ticks, 0), "seq": _seq, "text": text})


func tick() -> void:
	_now += 1


func poll() -> Array[String]:
	var ready: Array[Dictionary] = []
	var rest: Array[Dictionary] = []
	for e: Dictionary in _queue:
		if int(e["due"]) <= _now:
			ready.append(e)
		else:
			rest.append(e)
	_queue = rest
	ready.sort_custom(_fake_link_before)
	var out: Array[String] = []
	for e: Dictionary in ready:
		out.append(String(e["text"]))
	return out


func send(text: String) -> void:
	if _state == NetLink.STATE_OPEN:
		sent.append(text)


func connect_to(_url: String) -> void:
	_state = NetLink.STATE_OPEN


func close() -> void:
	_state = NetLink.STATE_CLOSED


func state() -> int:
	return _state


func pending() -> int:
	return _queue.size()


func _fake_link_before(a: Dictionary, b: Dictionary) -> bool:
	if int(a["due"]) != int(b["due"]):
		return int(a["due"]) < int(b["due"])
	return int(a["seq"]) < int(b["seq"])
