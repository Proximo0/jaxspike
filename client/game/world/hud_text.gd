class_name HudText
extends RefCounted
## Text for the HUD from a snap `you` dictionary.


static func hp_text(you: Dictionary) -> String:
	return "HP %d / %d" % [int(you.get("hp", 0)), int(you.get("max_hp", 0))]


static func score_text(you: Dictionary) -> String:
	return "Kills %d   Deaths %d" % [int(you.get("kills", 0)), int(you.get("deaths", 0))]


## Non-empty inventory item ids, e.g. "Items: 1 1 2 3". Item id 0 is an empty slot.
static func inv_text(you: Dictionary) -> String:
	var parts: PackedStringArray = PackedStringArray()
	var inv: Variant = you.get("inv", [])
	if inv is Array:
		for v: Variant in inv:
			var item: int = int(v)
			if item != 0:
				parts.append(str(item))
	if parts.is_empty():
		return "Items: none"
	return "Items: " + " ".join(parts)
