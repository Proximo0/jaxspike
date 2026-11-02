class_name NetLink
extends RefCounted
## Transport interface of the net layer. Subclasses: WsLink (real socket), FakeLink (scripted), ReplayLink (recording).
## Callers poll: there are no signals. `tick()` advances links that have a logical clock (fake lag, replay).

const STATE_CLOSED: int = 0
const STATE_CONNECTING: int = 1
const STATE_OPEN: int = 2


## Inbound texts received since the last poll, oldest first.
func poll() -> Array[String]:
	return []


func send(_text: String) -> void:
	pass


func connect_to(_url: String) -> void:
	pass


func close() -> void:
	pass


## STATE_CLOSED, STATE_CONNECTING or STATE_OPEN.
func state() -> int:
	return STATE_CLOSED


## Advance the link's logical clock by one tick. No-op for real sockets.
func tick() -> void:
	pass
