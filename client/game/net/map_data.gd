class_name MapData
extends RefCounted
## The whole map, received once in `welcome`. Row-major, index y * size + x.

var size: int = 0
var terrain: PackedByteArray = PackedByteArray()
var pz: PackedByteArray = PackedByteArray()


func is_loaded() -> bool:
	return size > 0 and terrain.size() == size * size


func in_bounds(x: int, y: int) -> bool:
	return x >= 0 and y >= 0 and x < size and y < size


## Terrain id at a tile (int8 stored in a byte: negative ids read back as 256 + id). -1 outside the map.
func at(x: int, y: int) -> int:
	if not in_bounds(x, y) or terrain.size() != size * size:
		return -1
	var b: int = terrain[y * size + x]
	return b - 256 if b > 127 else b


## True when the tile is in the protection zone.
func pz_at(x: int, y: int) -> bool:
	if not in_bounds(x, y) or pz.size() != size * size:
		return false
	return pz[y * size + x] != 0


## Fills from a `welcome` message. Returns false (and leaves the map empty) when the blobs do not decode.
func from_welcome(w: Dictionary) -> bool:
	size = 0
	terrain = PackedByteArray()
	pz = PackedByteArray()
	var n: int = int(w.get("map", 0))
	if n <= 0:
		return false
	var t: PackedByteArray = Codec.decode_map(String(w.get("terrain", "")), n * n)
	var z: PackedByteArray = Codec.decode_map(String(w.get("pz", "")), n * n)
	if t.is_empty() or z.is_empty():
		return false
	size = n
	terrain = t
	pz = z
	return true
