class_name ItemText
extends RefCounted
## ro5-items: the one item formatter. Display names ("+4 Blade [3]", "Gildscarab Card", "Red Potion x5"), the
## slots.json icon key of every item id (welcome.items[id].icon, else a fallback by kind and class), and the one
## tooltip builder that the inventory, the shop and the card album share. Pure: reads only the welcome message.

const KIND_ETC: int = 1
const KIND_CONSUMABLE: int = 2
const KIND_WEAPON: int = 3
const KIND_ARMOR: int = 4
const KIND_CARD: int = 5
const MAX_SOCKETS: int = 4
const LOWER_WORDS: Array[String] = ["of", "the", "and"]
## game/data/items.py WC_* order (0 = bare hands); weapon class -> icon key.
const WEAPON_ICONS: Array[String] = [
	"item.sword", "item.dagger", "item.sword", "item.bow", "item.staff", "item.mace", "item.axe",
]
## SLOT_* order (weapon, head, body, shield, garment, shoes, accessory) -> icon key.
const ARMOR_ICONS: Array[String] = [
	"item.armor", "item.hat", "item.armor", "item.shield", "item.garment", "item.boots", "item.ring",
]
const WEAPON_CLASS_NAMES: Array[String] = ["Weapon", "Dagger", "Sword", "Bow", "Staff", "Mace", "Axe"]
const ARMOR_SLOT_NAMES: Array[String] = ["Armor", "Headgear", "Armor", "Shield", "Garment", "Shoes", "Accessory"]
const ICON_ETC: String = "item.etc"
const ICON_CONSUMABLE: String = "item.potion_red"
const ICON_CARD: String = "item.card"
const REFINE_ATK_BY_LEVEL: Array[int] = [0, 2, 3, 5, 7]  ## per refine, by weapon level (ro_items.REFINE_ATK)
## ro6-ui: refine colour tiers of the rich tooltip: +1..+4 pale blue, +5..+7 gold, +8 and up orange.
const REFINE_TIER_MIN: Array[int] = [1, 5, 8]
const REFINE_TIER_COLORS: Array[Color] = [Color(0.55, 0.85, 1.0), Color(1.0, 0.82, 0.25), Color(1.0, 0.5, 0.3)]
const NAME_COLOR: Color = Color(1.0, 1.0, 1.0)


## The welcome.items row of an id ({} for 0, out of range or before the table arrives).
static func info(welcome_msg: Dictionary, id: int) -> Dictionary:
	var items: Variant = welcome_msg.get("items")
	if id > 0 and typeof(items) == TYPE_ARRAY and id < (items as Array).size():
		var r: Variant = (items as Array)[id]
		if typeof(r) == TYPE_DICTIONARY:
			return r
	return {}


static func kind(welcome_msg: Dictionary, id: int) -> int:
	return int(info(welcome_msg, id).get("kind", 0))


## 'gildscarab card' -> 'Gildscarab Card', 'bill of birds' -> 'Bill of Birds' (keeps already capitalised words).
static func title_case(s: String) -> String:
	var out: PackedStringArray = []
	for w: String in s.strip_edges().split(" ", false):
		if not out.is_empty() and LOWER_WORDS.has(w):
			out.append(w)
		else:
			out.append(w.substr(0, 1).to_upper() + w.substr(1))
	return " ".join(out)


## The plain capitalised name: welcome.items display, else its name, else welcome.craft_items, else "Item <id>".
static func base_name(welcome_msg: Dictionary, id: int) -> String:
	if id <= 0:
		return ""
	var r: Dictionary = info(welcome_msg, id)
	var disp: String = String(r.get("display", ""))
	if disp != "":
		return disp
	var nm: String = String(r.get("name", ""))
	if nm != "":
		return title_case(nm)
	var ci: Variant = welcome_msg.get("craft_items")
	if typeof(ci) == TYPE_DICTIONARY:
		var d: Dictionary = ci
		if d.has(str(id)):
			return title_case(String(d[str(id)]))
		if d.has(id):
			return title_case(String(d[id]))
	return "Item %d" % id


## Card sockets of a weapon/armor id, 0 for anything else.
static func sockets(welcome_msg: Dictionary, id: int) -> int:
	var r: Dictionary = info(welcome_msg, id)
	var k: int = int(r.get("kind", 0))
	if k != KIND_WEAPON and k != KIND_ARMOR:
		return 0
	return clampi(int(r.get("slots", 0)), 0, MAX_SOCKETS)


## RO-style display name: "+4 Blade [3]" (refine on gear, socket count on slotted gear), "Gildscarab Card".
static func display_name(welcome_msg: Dictionary, id: int, refine: int = 0) -> String:
	if id <= 0:
		return ""
	var n: String = base_name(welcome_msg, id)
	var k: int = kind(welcome_msg, id)
	if refine > 0 and (k == KIND_WEAPON or k == KIND_ARMOR):
		n = "+%d %s" % [refine, n]
	var s: int = sockets(welcome_msg, id)
	if s > 0:
		n += " [%d]" % s
	return n


## A list line: the display name plus " x<count>" for a stack of more than one.
static func stack_name(welcome_msg: Dictionary, id: int, count: int, refine: int = 0) -> String:
	var n: String = display_name(welcome_msg, id, refine)
	return n + (" x%d" % count if count > 1 else "")


## slots.json key for an item id: welcome.items[id].icon when given, else by kind (weapon class, armor slot).
static func icon_key(welcome_msg: Dictionary, id: int) -> String:
	if id <= 0:
		return ""
	var r: Dictionary = info(welcome_msg, id)
	var ic: String = String(r.get("icon", ""))
	if ic.begins_with("item."):
		return ic
	return icon_by_kind(int(r.get("kind", KIND_ETC)), int(r.get("cls", 0)))


static func icon_by_kind(k: int, cls: int) -> String:
	match k:
		KIND_WEAPON:
			return WEAPON_ICONS[cls] if cls >= 0 and cls < WEAPON_ICONS.size() else WEAPON_ICONS[0]
		KIND_ARMOR:
			return ARMOR_ICONS[cls] if cls >= 0 and cls < ARMOR_ICONS.size() else ARMOR_ICONS[0]
		KIND_CARD:
			return ICON_CARD
		KIND_CONSUMABLE:
			return ICON_CONSUMABLE
	return ICON_ETC


## icon_key for every id of welcome.items ("" for reserved rows); AssetSlots.item_icons takes it for the drops.
static func icon_table(welcome_msg: Dictionary) -> PackedStringArray:
	var out: PackedStringArray = []
	var items: Variant = welcome_msg.get("items")
	if typeof(items) != TYPE_ARRAY:
		return out
	for i: int in range((items as Array).size()):
		var r: Dictionary = info(welcome_msg, i)
		out.append("" if r.is_empty() or (int(r.get("kind", 0)) == 0 and String(r.get("icon", "")) == "")
				else icon_key(welcome_msg, i))
	return out


## ATK (weapon) or DEF (armor) including refine.
static func gear_stat(welcome_msg: Dictionary, id: int, refine: int) -> int:
	var r: Dictionary = info(welcome_msg, id)
	if int(r.get("kind", 0)) == KIND_WEAPON:
		var wl: int = clampi(int(r.get("weapon_level", 1)), 0, REFINE_ATK_BY_LEVEL.size() - 1)
		return int(r.get("atk", 0)) + refine * REFINE_ATK_BY_LEVEL[wl]
	return int(r.get("def", 0)) + refine


## The type line: "Sword, ATK 25", "Headgear, DEF 2", "Consumable, heals 45 HP", "Card", "Material".
static func type_line(welcome_msg: Dictionary, id: int, refine: int = 0) -> String:
	var r: Dictionary = info(welcome_msg, id)
	var cls: int = int(r.get("cls", 0))
	match int(r.get("kind", 0)):
		KIND_WEAPON:
			var wc: String = WEAPON_CLASS_NAMES[cls] if cls >= 0 and cls < WEAPON_CLASS_NAMES.size() else "Weapon"
			var t: String = "%s, ATK %d" % [wc, gear_stat(welcome_msg, id, refine)]
			if int(r.get("matk", 0)) > 0:
				t += ", MATK %d" % int(r["matk"])
			return t
		KIND_ARMOR:
			var sl: String = ARMOR_SLOT_NAMES[cls] if cls >= 0 and cls < ARMOR_SLOT_NAMES.size() else "Armor"
			return "%s, DEF %d" % [sl, gear_stat(welcome_msg, id, refine)]
		KIND_CONSUMABLE:
			var parts: PackedStringArray = []
			if int(r.get("heal_hp", 0)) > 0:
				parts.append("%d HP" % int(r["heal_hp"]))
			if int(r.get("heal_sp", 0)) > 0:
				parts.append("%d SP" % int(r["heal_sp"]))
			return "Consumable" + (", heals " + " and ".join(parts) if not parts.is_empty() else "")
		KIND_CARD:
			return "Card"
	return "Material"


## The welcome.cards row of a card id ({} for anything else).
static func card_row(welcome_msg: Dictionary, id: int) -> Dictionary:
	var v: Variant = welcome_msg.get("cards")
	if id <= 0 or typeof(v) != TYPE_ARRAY:
		return {}
	for r: Variant in v as Array:
		if typeof(r) == TYPE_DICTIONARY and int((r as Dictionary).get("id", 0)) == id:
			return r
	return {}


## '1 in 10,000' for a per-10000 chance ("" for 0).
static func chance_text(per_10000: int) -> String:
	if per_10000 <= 0:
		return ""
	var s: String = str(roundi(10000.0 / float(per_10000)))
	var grouped: String = ""
	while s.length() > 3:
		grouped = "," + s.substr(s.length() - 3) + grouped
		s = s.substr(0, s.length() - 3)
	return "1 in %s%s" % [s, grouped]


## The card lines of a welcome.cards row: "<Slot> card: <effects>" and where it drops.
static func card_lines(row: Dictionary) -> PackedStringArray:
	var out: PackedStringArray = []
	if row.is_empty():
		return out
	out.append("%s card: %s" % [String(row.get("slot_name", "")), String(row.get("text", ""))])
	var mon: String = String(row.get("monster", ""))
	if mon == "":
		out.append("No monster in Aurelune drops it yet.")
		return out
	var t: String = "Drops from " + mon + (" (MVP)" if bool(row.get("mvp", false)) else "")
	var ch: String = chance_text(int(row.get("chance", 0)))
	if ch != "":
		t += ", " + ch
	var where: String = String(row.get("where", ""))
	if where != "":
		t += " (%s)" % where
	out.append(t)
	return out


## The one tooltip: name line, type line (or `stat_line` in its place, the inventory's compare text), socketed
## cards with effects, card lines for a card, weight, then the caller's `extra` lines (price, compound, owned).
static func tooltip(welcome_msg: Dictionary, id: int, refine: int = 0, cards: Array = [],
		extra: PackedStringArray = PackedStringArray(), stat_line: String = "") -> String:
	if id <= 0:
		return ""
	var lines: PackedStringArray = PackedStringArray()
	lines.append(display_name(welcome_msg, id, refine))
	lines.append(stat_line if stat_line != "" else type_line(welcome_msg, id, refine))
	var bonus: String = bonus_text(welcome_msg, id)
	if bonus != "":
		lines.append("Bonus: " + bonus)
	var names: PackedStringArray = []
	for c: Variant in cards:
		var cid: int = int(c)
		if cid > 0:
			var eff: String = String(card_row(welcome_msg, cid).get("text", ""))
			names.append(base_name(welcome_msg, cid) + (" (%s)" % eff if eff != "" else ""))
	var n_sock: int = sockets(welcome_msg, id)
	if not names.is_empty():
		var t: String = "Cards: " + ", ".join(names)
		if n_sock > 0:
			t += "  (%d / %d sockets)" % [mini(names.size(), n_sock), n_sock]
		lines.append(t)
	elif n_sock > 0:
		lines.append("%d empty socket%s" % [n_sock, "" if n_sock == 1 else "s"])
	lines.append_array(card_lines(card_row(welcome_msg, id)))
	var w: int = int(info(welcome_msg, id).get("weight", 0))
	if w > 0:
		lines.append("Weight %d" % w)
	lines.append_array(extra)
	return "\n".join(lines)


# ---- ro6-ui: the rich tooltip (ItemTooltip draws this spec; the plain `tooltip` text stays the fallback) ----

## The gear bonus of welcome.items[id].bonus_text ("STR +2"), "" when none or before the server sends the field.
static func bonus_text(welcome_msg: Dictionary, id: int) -> String:
	return String(info(welcome_msg, id).get("bonus_text", ""))


## Colour of a refine level's "+N" (white for 0).
static func refine_color(refine: int) -> Color:
	var c: Color = NAME_COLOR
	for i: int in range(REFINE_TIER_MIN.size()):
		if refine >= REFINE_TIER_MIN[i]:
			c = REFINE_TIER_COLORS[i]
	return c


## Everything the rich tooltip draws, as plain data: id, icon_key, card_art (card.<snake> of a card item, else ""),
## refine_text ("+4" or ""), refine_color, name ("Blade [3]", no refine), type_line (or `stat_line`), bonus,
## cards [{id, name, text, art}] for the socketed cards, sockets, empty_sockets, card_lines, weight, extra, text
## (the plain `tooltip` string, also the fallback). {} for id 0.
static func tooltip_spec(welcome_msg: Dictionary, id: int, refine: int = 0, cards: Array = [],
		extra: PackedStringArray = PackedStringArray(), stat_line: String = "") -> Dictionary:
	if id <= 0:
		return {}
	var k: int = kind(welcome_msg, id)
	var gear: bool = k == KIND_WEAPON or k == KIND_ARMOR
	var n_sock: int = sockets(welcome_msg, id)
	var socketed: Array[Dictionary] = []
	for c: Variant in cards:
		var cid: int = int(c)
		if cid > 0:
			var row: Dictionary = card_row(welcome_msg, cid)
			socketed.append({"id": cid, "name": base_name(welcome_msg, cid), "text": String(row.get("text", "")),
					"art": String(row.get("icon", ""))})
	var own_card: Dictionary = card_row(welcome_msg, id)
	return {
		"id": id, "icon_key": icon_key(welcome_msg, id), "card_art": String(own_card.get("icon", "")),
		"refine_text": "+%d" % refine if gear and refine > 0 else "",
		"refine_color": refine_color(refine if gear else 0),
		"name": base_name(welcome_msg, id) + (" [%d]" % n_sock if n_sock > 0 else ""),
		"type_line": stat_line if stat_line != "" else type_line(welcome_msg, id, refine),
		"bonus": bonus_text(welcome_msg, id), "cards": socketed, "sockets": n_sock,
		"empty_sockets": maxi(n_sock - socketed.size(), 0), "card_lines": card_lines(own_card),
		"weight": int(info(welcome_msg, id).get("weight", 0)), "extra": extra,
		"text": tooltip(welcome_msg, id, refine, cards, extra, stat_line),
	}
