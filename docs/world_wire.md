# World wire: what the server must expose and the client must know about the RO world

Branch `ro/world`. Everything here is static per map seed, so it belongs in `welcome` (sent once) or in the
client's own tables, never in `snap`. Source of truth: `game/world_layout.py` (RO section) and
`game/data/terrain.py`. Tile numbers below are for `MAP = 512`; the client must use the fractions or the
`welcome` payload, never these literals, because `MAP` becomes 2048 for the first playable release.

## Terrain ids 8..11 (ids 0..7 unchanged)
| id | name | blocks | pz | suggested colour (Godot `Color`) | notes |
|---|---|---|---|---|---|
| 8 | warp | no | no | `Color(0.55, 0.25, 0.85)` violet, pulse alpha 0.6..1.0 | stepping on it teleports (server side, `game/ro_warps.py`) |
| 9 | kafra | no | yes (inside a town) | `Color(0.95, 0.80, 0.30)` gold | save point / storage NPC tile; draw a kafra sprite over it |
| 10 | station | no | yes inside towns, no elsewhere | `Color(0.40, 0.70, 0.95)` light blue | arrival pad of a warp; never teleports |
| 11 | cave | no | no | `Color(0.33, 0.28, 0.26)` dark brown | dungeon floor; dungeon walls are `T_ROCK` (7) |

The client's `ViewMath.TERRAIN_COLORS` has 8 entries and must grow to 12 (`client/game/world/view_math.gd`;
`client/game/net/map_data.gd` decodes ids as bytes, no change). Minimap: draw ids 8 and 10 as dots
regardless of zoom; tint dungeon boxes (below) so cave floor reads as "underground".

## Towns (index = `TOWNS` index = slot `% NT` home town)
| idx | name | role | centre (fraction) | kafra tile @512 | station tile @512 |
|---|---|---|---|---|---|
| 0 | Prontera | capital | (0.50, 0.50) | (264, 256) | (248, 256) |
| 1 | Geffen | mage city | (0.20, 0.22) | (110, 113) | (94, 113) |
| 2 | Payon | archer village | (0.80, 0.20) | (418, 102) | (402, 102) |
| 3 | Alberta | merchant port | (0.78, 0.80) | (407, 410) | (391, 410) |
| 4 | Morroc | desert thief town | (0.22, 0.78) | (121, 399) | (105, 399) |

Kafra = temple square's right edge + 2 tiles; station = left edge - 2 tiles (`KAFRA_OFFSET_TILES`).
Town stations are arrival pads for a future kafra teleport service; no warp targets them in v1.

## Dungeons (walled rock boxes, entered only by warp)
| idx | name | centre (fraction) | half-side | box @512 (x0, y0, x1, y1) | entrance warp -> inside station | exit warp -> outside station | kinds | MVP |
|---|---|---|---|---|---|---|---|---|
| 0 | Prontera Culvert | (0.50, 0.09) | 0.055 | (228, 18, 284, 74) | (256, 77) -> (256, 71) | (259, 71) -> (259, 77) | Thief Bug, Spore, Zombie, Skeleton | Golden Thief Bug |
| 1 | Payon Cave | (0.50, 0.91) | 0.055 | (228, 438, 284, 494) | (256, 435) -> (256, 441) | (259, 441) -> (259, 435) | Zombie, Skeleton, Soldier Skeleton, Mummy | Eddga |

A road runs from each entrance plaza to Prontera (`DUNGEON_ROADS`), opening a north and a south gate in its
wall. The ring is 2 tiles of rock; 2x2 rock pillars every 8 tiles inside.

## Warps (`WARPS`, fractions; `map_gen.warp_table()` gives the padded int32 `[N_WARPS=16, 2]` arrays)
| # | from @512 | to @512 | meaning |
|---|---|---|---|
| 0 | (256, 77) | (256, 71) | into Prontera Culvert |
| 1 | (259, 71) | (259, 77) | out of Prontera Culvert |
| 2 | (256, 435) | (256, 441) | into Payon Cave |
| 3 | (259, 441) | (259, 435) | out of Payon Cave |

Server rule (`ro_warps.apply_warps`, runs right after movement): a living player standing on a warp tile
is moved to its destination the same tick; if the destination holds a living player or an active monster the
player stays on the warp and retries next tick; lowest slot wins a shared destination; monsters never warp.
The client sees an ordinary position jump in `snap`: snap the camera (no interpolation) when a player's
position changes by more than 2 tiles in one tick, and play the warp sound on `terrain[old] == 8`.

## Field bands (walking distance to the nearest town gate, fraction of `MAP`)
| band | distance | kinds |
|---|---|---|
| 0 | 0.00 - 0.08 | Poring, Fabre, Lunatic, Drops |
| 1 | 0.08 - 0.16 | Poporing, Willow, Rocker, Spore, Thief Bug |
| 2 | 0.16 + | Wolf, Peco Peco, Hode, Bigfoot, Orc Warrior |

Monster kind ids, names, hp, level and MVP flag: `game/data/monsters.py` (`MONSTERS`); the client needs
`name` and `mvp` per kind for name plates and the MVP marker, and `size` for sprite scale.

## `welcome` additions (static, once per connection; `net/protocol.py` is lead-owned, report)
```json
"world": {
  "towns": [{"idx": 0, "name": "Prontera", "role": "capital", "center": [256, 256],
             "kafra": [264, 256], "station": [248, 256]}, ...],
  "dungeons": [{"idx": 0, "name": "Prontera Culvert", "box": [228, 18, 284, 74]}, ...],
  "warps": [{"from": [256, 77], "to": [256, 71]}, ...],
  "terrain_names": ["grass", "road", "floor", "temple", "water", "tree", "wall", "rock",
                    "warp", "kafra", "station", "cave"],
  "monsters": [{"kind": 0, "name": "Poring", "hp": 40, "level": 1, "mvp": false, "size": 0}, ...]
}
```
Server side these are plain Python from `world_layout.town_centers()`, `kafra_tiles()`,
`town_station_tiles()`, `dungeon_boxes()`, `warp_tiles()`, `TOWN_NAMES`, `TOWN_ROLES`, `DUNGEONS`,
`game.data.terrain.TERRAIN_NAMES`, `game.data.monsters.MONSTERS`.

## `snap.you` additions proposed (optional, lead decides)
- `town_name` is derivable from `town`; not needed.
- `region`: `"town" | "field0" | "field1" | "field2" | "dungeon0" | "dungeon1"` for the HUD's area label.
  Derivable client-side from `terrain`, `pz`, the dungeon boxes and the `welcome` data, so it is **not**
  proposed for the wire; the client computes it from the tile under the player.

## Client messages
None new. Warps and kafra tiles need no input: warps fire on position; kafra save/storage is a later
milestone (accounts team) and will be a `kafra` message with `{"op": "save"}` while standing on id 9.
