class_name HudText
extends RefCounted
## Text for the HUD from a snap `you` dictionary.


static func hp_text(you: Dictionary) -> String:
	return "HP %d / %d" % [int(you.get("hp", 0)), int(you.get("max_hp", 0))]


static func score_text(you: Dictionary) -> String:
	return "Kills %d   Deaths %d" % [int(you.get("kills", 0)), int(you.get("deaths", 0))]


## Non-empty inventory item ids, e.g. "Items: 1 1 2x3". A row is an int id (v1) or a dict {item, count, ...} (RO);
## item id 0 is an empty slot.
static func inv_text(you: Dictionary) -> String:
	var parts: PackedStringArray = PackedStringArray()
	var inv: Variant = you.get("inv", [])
	if inv is Array:
		for v: Variant in inv:
			var item: int = 0
			var count: int = 1
			if v is Dictionary:
				item = int((v as Dictionary).get("item", 0))
				count = int((v as Dictionary).get("count", 1))
			elif v is int or v is float:
				item = int(v)
			if item != 0:
				parts.append(str(item) if count <= 1 else "%dx%d" % [item, count])
	if parts.is_empty():
		return "Items: none"
	return "Items: " + " ".join(parts)
