class_name WsLink
extends NetLink
## NetLink over a WebSocketPeer (text frames). The only net class that touches the engine's socket layer.

const INBOUND_BUFFER_BYTES: int = 4 * 1024 * 1024
const OUTBOUND_BUFFER_BYTES: int = 64 * 1024
const MAX_QUEUED_PACKETS: int = 4096

var peer: WebSocketPeer = null
var close_code: int = -1


func connect_to(url: String) -> void:
	close()
	peer = WebSocketPeer.new()
	peer.inbound_buffer_size = INBOUND_BUFFER_BYTES
	peer.outbound_buffer_size = OUTBOUND_BUFFER_BYTES
	peer.max_queued_packets = MAX_QUEUED_PACKETS
	close_code = -1
	var err: Error = peer.connect_to_url(url)
	if err != OK:
		peer = null


func poll() -> Array[String]:
	var out: Array[String] = []
	for p: Variant in _ws_link_drain():
		if typeof(p) == TYPE_STRING:
			out.append(String(p))
	return out


func poll_packets() -> Array:
	return _ws_link_drain()


## Shared loop: Strings for text frames, PackedByteArray for binary frames.
func _ws_link_drain() -> Array:
	var out: Array = []
	if peer == null:
		return out
	peer.poll()
	while peer.get_available_packet_count() > 0:
		var pkt: PackedByteArray = peer.get_packet()
		if peer.was_string_packet():
			out.append(pkt.get_string_from_utf8())
		else:
			out.append(pkt)
	if peer.get_ready_state() == WebSocketPeer.STATE_CLOSED:
		close_code = peer.get_close_code()
	return out


func send(text: String) -> void:
	if peer != null and peer.get_ready_state() == WebSocketPeer.STATE_OPEN:
		peer.send_text(text)


func close() -> void:
	if peer != null:
		peer.close()
		peer = null


func state() -> int:
	if peer == null:
		return NetLink.STATE_CLOSED
	match peer.get_ready_state():
		WebSocketPeer.STATE_CONNECTING:
			return NetLink.STATE_CONNECTING
		WebSocketPeer.STATE_OPEN:
			return NetLink.STATE_OPEN
	return NetLink.STATE_CLOSED
