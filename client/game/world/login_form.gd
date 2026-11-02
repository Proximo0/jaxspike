class_name LoginForm
extends RefCounted
## Validation of the login fields. Returns {ok, error, name, url}.

const DEFAULT_HOST: String = "127.0.0.1"
const DEFAULT_PORT: int = 8765
const MAX_NAME: int = 16


static func validate(player_name: String, host: String, port_text: String) -> Dictionary:
	var n: String = player_name.strip_edges()
	var h: String = host.strip_edges()
	var p: String = port_text.strip_edges()
	if n.is_empty():
		return _fail("Enter a name.")
	if n.length() > MAX_NAME:
		return _fail("Name is longer than %d characters." % MAX_NAME)
	if h.is_empty():
		h = DEFAULT_HOST
	if h.contains(" ") or h.contains("/"):
		return _fail("Host is not valid.")
	var port: int = DEFAULT_PORT
	if not p.is_empty():
		if not p.is_valid_int():
			return _fail("Port must be a number.")
		port = p.to_int()
	if port < 1 or port > 65535:
		return _fail("Port must be 1..65535.")
	return {"ok": true, "error": "", "name": n, "url": url_for(h, port)}


static func url_for(host: String, port: int) -> String:
	return "ws://%s:%d" % [host, port]


static func _fail(msg: String) -> Dictionary:
	return {"ok": false, "error": msg, "name": "", "url": ""}
