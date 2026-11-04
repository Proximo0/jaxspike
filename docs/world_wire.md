# World wire: what the server must expose and the client must know about the RO world

Branch `ro/world`. Everything here is static per map seed, so it belongs in `welcome` (sent once) or in the
client's own tables, never in `snap`. Source of truth: `game/world_layout.py` (RO section) and
`game/data/terrain.py`. Tile numbers below are for `MAP = 512`; the client must use the fractions or the
`welcome` payload, never these literals, because `MAP` becomes 2048 for the first playable release.

## Terrain ids 8..12 (ids 0..7 unchanged)
| id | name | blocks | pz | suggested colour (Godot `Color`) | notes |
|---|---|---|---|---|---|
| 8 | warp | no | no | `Color(0.55, 0.25, 0.85)` violet, pulse alpha 0.6..1.0 | stepping on it teleports (server side, `game/ro_warps.py`) |
| 9 | waybell | no | yes (inside a town) | `Color(0.95, 0.80, 0.30)` gold | save point / storage NPC tile; draw a waybell sprite over it |
| 10 | station | no | yes inside towns, no elsewhere | `Color(0.40, 0.70, 0.95)` light blue | arrival pad of a warp; never teleports |
| 11 | cave | no | no | `Color(0.33, 0.28, 0.26)` dark brown | dungeon floor; dungeon walls are `T_ROCK` (7) |
| 12 | shop | no | yes (inside a town) | `Color(1.00, 0.35, 0.63)` pink | NPC shop tile (ro-world2): stand on it to buy/sell (ro-items2); draw a stall sprite |

The client's `ViewMath.TERRAIN_COLORS` has 8 entries and must grow to 13 (`client/game/world/view_math.gd`;
`client/game/net/map_data.gd` decodes ids as bytes, no change). Minimap: draw ids 8 and 10 as dots
regardless of zoom; tint dungeon boxes (below) so cave floor reads as "underground".

## Towns (index = `TOWNS` index = slot `% NT` home town)
| idx | name | role | centre (fraction) | waybell tile @512 | station tile @512 |
|---|---|---|---|---|---|
| 0 | Solmera | capital | (0.50, 0.50) | (264, 256) | (248, 256) |
| 1 | Lunavell | mage city | (0.20, 0.22) | (110, 113) | (94, 113) |
| 2 | Pinehollow | archer village | (0.80, 0.20) | (418, 102) | (402, 102) |
| 3 | Tidewick | merchant port | (0.78, 0.80) | (407, 410) | (391, 410) |
| 4 | Kharsand | desert thief town | (0.22, 0.78) | (121, 399) | (105, 399) |

Waybell = temple square's right edge + 2 tiles; station = left edge - 2 tiles (`WAYBELL_OFFSET_TILES`).
Town stations are arrival pads for a future waybell teleport service; no warp targets them in v1.

## Dungeons (walled rock boxes, entered only by warp)
| idx | name | centre (fraction) | half-side | box @512 (x0, y0, x1, y1) | entrance warp -> inside station | exit warp -> outside station | kinds | MVP |
|---|---|---|---|---|---|---|---|---|
| 0 | Solmera Culvert | (0.50, 0.09) | 0.055 | (228, 18, 284, 74) | (256, 77) -> (256, 71) | (259, 71) -> (259, 77) | Pilferbug, Capling, Rotling, Clackbone | Gildscarab |
| 1 | Pinehollow Cave | (0.50, 0.91) | 0.055 | (228, 438, 284, 494) | (256, 435) -> (256, 441) | (259, 441) -> (259, 435) | Rotling, Clackbone, Clackbone Guard, Tombwrap | Scorchstripe |

A road runs from each entrance plaza to Solmera (`DUNGEON_ROADS`), opening a north and a south gate in its
wall. The ring is 2 tiles of rock; 2x2 rock pillars every 8 tiles inside.

## Warps (`WARPS`, fractions; `map_gen.warp_table()` gives the padded int32 `[N_WARPS=48, 2]` arrays; rows 0..3 below are `DUNGEON_WARPS`, rows 4.. the 28 zone-portal legs, see Zones)
| # | from @512 | to @512 | meaning |
|---|---|---|---|
| 0 | (256, 77) | (256, 71) | into Solmera Culvert |
| 1 | (259, 71) | (259, 77) | out of Solmera Culvert |
| 2 | (256, 435) | (256, 441) | into Pinehollow Cave |
| 3 | (259, 441) | (259, 435) | out of Pinehollow Cave |

Server rule (`ro_warps.apply_warps`, runs right after movement): a living player standing on a warp tile
is moved to its destination the same tick; if the destination holds a living player or an active monster the
player stays on the warp and retries next tick; lowest slot wins a shared destination; monsters never warp.
The client sees an ordinary position jump in `snap`: snap the camera (no interpolation) when a player's
position changes by more than 2 tiles in one tick, and play the warp sound on `terrain[old] == 8`.

## Field bands (walking distance to the nearest town gate, fraction of `MAP`)
| band | distance | kinds |
|---|---|---|
| 0 | 0.00 - 0.08 | Blobbin, Wrigglet, Moonbun, Emberblob |
| 1 | 0.08 - 0.16 | Vexblob, Bramblet, Fiddlehop, Capling, Pilferbug |
| 2 | 0.16 + | Snarlfang, Gallopeck, Sandgulper, Grumblebear, Grukk Brute |

Monster kind ids, names, hp, level and MVP flag: `game/data/monsters.py` (`MONSTERS`); the client needs
`name` and `mvp` per kind for name plates and the MVP marker, and `size` for sprite scale.

## `welcome` additions (static, once per connection; `net/protocol.py` is lead-owned, report)
```json
"world": {
  "towns": [{"idx": 0, "name": "Solmera", "role": "capital", "center": [256, 256],
             "waybell": [264, 256], "station": [248, 256]}, ...],
  "dungeons": [{"idx": 0, "name": "Solmera Culvert", "box": [228, 18, 284, 74]}, ...],
  "warps": [{"from": [256, 77], "to": [256, 71]}, ...],
  "terrain_names": ["grass", "road", "floor", "temple", "water", "tree", "wall", "rock",
                    "warp", "waybell", "station", "cave"],
  "monsters": [{"kind": 0, "name": "Blobbin", "hp": 40, "level": 1, "mvp": false, "size": 0}, ...]
}
```
Server side these are plain Python from `world_layout.town_centers()`, `waybell_tiles()`,
`town_station_tiles()`, `dungeon_boxes()`, `warp_tiles()`, `TOWN_NAMES`, `TOWN_ROLES`, `DUNGEONS`,
`game.data.terrain.TERRAIN_NAMES`, `game.data.monsters.MONSTERS`.

## `snap.you` additions proposed (optional, lead decides)
- `town_name` is derivable from `town`; not needed.
- `region`: `"town" | "field0" | "field1" | "field2" | "dungeon0" | "dungeon1"` for the HUD's area label.
  Derivable client-side from `terrain`, `pz`, the dungeon boxes and the `welcome` data, so it is **not**
  proposed for the wire; the client computes it from the tile under the player.

## Client messages
None new. Warps and waybell tiles need no input: warps fire on position; waybell save/storage is a later
milestone (accounts team) and will be a `waybell` message with `{"op": "save"}` while standing on id 9.

## Zones (ro-world2): RO maps as rectangles of the one big map
`world_layout.ZONES` (index = zone id): `(name, kind, ref, (level lo, level hi), music id)`; kinds `town` /
`field` / `dungeon` (`ZONE_KINDS`). Fields tile the map in a 3 x 3 grid split at `ZONE_LINES = (0.375, 0.625)`
of MAP on both axes; each split line is **one tile of `T_ROCK` across the whole map** (outside town and dungeon
boxes, which never touch a line). Towns (their wall box) and dungeons (their rock box) sit inside field cells and
win: `zone_of(pos int [N, 2]) -> int32 [N]` (pure JAX, jit/vmap) returns dungeon > town > field cell.
`zone_boxes()` gives inclusive tile boxes, `zone_box_fractions()` the same as fractions (x1, y1 exclusive).

| id | name | kind | box (fraction x0, y0, x1, y1) | box @512 inclusive | levels | music |
|---|---|---|---|---|---|---|
| 0 | solmera | town | (0.4395, 0.4395, 0.5625, 0.5625) | (225, 225, 287, 287) | 0-0 | 8 |
| 1 | lunavell | town | (0.1484, 0.1699, 0.2520, 0.2734) | (76, 87, 128, 139) | 0-0 | 13 |
| 2 | pinehollow | town | (0.7500, 0.1484, 0.8535, 0.2520) | (384, 76, 436, 128) | 0-0 | 3 |
| 3 | tidewick | town | (0.7285, 0.7500, 0.8320, 0.8535) | (373, 384, 425, 436) | 0-0 | 11 |
| 4 | kharsand | town | (0.1699, 0.7285, 0.2734, 0.8320) | (87, 373, 139, 425) | 0-0 | 18 |
| 5 | sol_sew01 | dungeon | (0.4453, 0.0352, 0.5566, 0.1465) | (228, 18, 284, 74) | 5-25 | 19 |
| 6 | pin_dun00 | dungeon | (0.4453, 0.8555, 0.5566, 0.9668) | (228, 438, 284, 494) | 15-40 | 20 |
| 7 | lun_fld00 | field | (0, 0, 0.375, 0.375) | (0, 0, 191, 191) | 8-25 | 12 |
| 8 | sol_fld01 | field | (0.375, 0, 0.625, 0.375) | (192, 0, 319, 191) | 3-15 | 9 |
| 9 | pin_fld01 | field | (0.625, 0, 1, 0.375) | (320, 0, 511, 191) | 5-20 | 4 |
| 10 | sol_fld05 | field | (0, 0.375, 0.375, 0.625) | (0, 192, 191, 319) | 3-15 | 9 |
| 11 | sol_fld08 | field | (0.375, 0.375, 0.625, 0.625) | (192, 192, 319, 319) | 1-6 | 9 |
| 12 | sol_fld06 | field | (0.625, 0.375, 1, 0.625) | (320, 192, 511, 319) | 5-18 | 9 |
| 13 | kha_fld01 | field | (0, 0.625, 0.375, 1) | (0, 320, 191, 511) | 10-30 | 17 |
| 14 | sol_fld02 | field | (0.375, 0.625, 0.625, 1) | (192, 320, 319, 511) | 3-15 | 9 |
| 15 | tid_fld01 | field | (0.625, 0.625, 1, 1) | (320, 320, 511, 511) | 5-20 | 10 |

Level bands and music ids are display data (zone name banner, BGM); monster spawns still follow `FIELD_BANDS`
by walking distance (`game/ro_spawns.py`). The split lines fall on tiles 192 and 320 at MAP 512 (768 and 1280 at
2048): the line tile belongs to the higher cell by `zone_of`, but it is rock, so nobody stands there.

### Edge portals
Every road centre line that crosses a split line gets a **portal pair** (`zone_portals() -> int32 [NP, 2, 2, 2]`,
per pair two legs `(warp tile, station tile)`). Leg 0 crosses from the low side to the high side: warp 1 tile
before the line, station `PORTAL_STEP_TILES = 2` past it, on the same road row; leg 1 mirrors it. So walking off a
field's edge along a road steps on a `T_WARP` and lands on the matching road tile of the neighbour zone, one step
from the warp back (RO-style). 14 pairs at any MAP (same fractions):

| pair | warp -> station @512 | warp -> station @512 |
|---|---|---|
| lun_fld00 <-> sol_fld01 | (191, 113) -> (194, 113) | (193, 113) -> (190, 113) |
| sol_fld05 <-> sol_fld08 | (191, 256) -> (194, 256) | (193, 256) -> (190, 256) |
| kha_fld01 <-> sol_fld02 | (191, 410) -> (194, 410) | (193, 410) -> (190, 410) |
| sol_fld01 <-> pin_fld01 | (319, 113) -> (322, 113) | (321, 113) -> (318, 113) |
| sol_fld08 <-> sol_fld06 | (319, 256) -> (322, 256) | (321, 256) -> (318, 256) |
| sol_fld02 <-> tid_fld01 | (319, 410) -> (322, 410) | (321, 410) -> (318, 410) |
| lun_fld00 <-> sol_fld05 | (102, 191) -> (102, 194) | (102, 193) -> (102, 190) |
| sol_fld01 <-> sol_fld08 | (256, 191) -> (256, 194) | (256, 193) -> (256, 190) |
| pin_fld01 <-> sol_fld06 | (399, 191) -> (399, 194) | (399, 193) -> (399, 190) |
| pin_fld01 <-> sol_fld06 | (410, 191) -> (410, 194) | (410, 193) -> (410, 190) |
| sol_fld05 <-> kha_fld01 | (102, 319) -> (102, 322) | (102, 321) -> (102, 318) |
| sol_fld05 <-> kha_fld01 | (113, 319) -> (113, 322) | (113, 321) -> (113, 318) |
| sol_fld08 <-> sol_fld02 | (256, 319) -> (256, 322) | (256, 321) -> (256, 318) |
| sol_fld06 <-> tid_fld01 | (399, 319) -> (399, 322) | (399, 321) -> (399, 318) |

Portals are ordinary warps: `ro_warps.apply_warps` moves players, monsters never cross (the rock line stops them,
like RO map edges), and `distance_fields_ro` relaxes across them, so bots walk home through portals. Town gates
are not portals: a town zone starts at its wall, and walking through a gate changes zone without a jump. The
client shows the zone name when `zone_of(you.pos)` changes (compute it client-side from `welcome.world.zones`:
dungeon box, then town box, then field box; or the optional `snap.you.zone` below).

## Town layouts (ro-world2)
Each town keeps its centred temple square (spawn), waybell (right of it) and town station (left). Added:
a paved plaza (`T_ROAD`, half-side `TEMPLE_TILES // 2 + PLAZA_MARGIN_TILES` = 12) around the temple, cross
streets wall to wall (half-width `TOWN_STREET_HALF`: Solmera 2 = 5 wide, others 1), a shop row of `T_SHOP`
tiles just north of the east street (`SHOP_ROW_TILES` = 12, 4, 4, 6, 4; `shop_tiles()` gives tiles and town),
and decoration that only replaces town floor (`town_decor()`; never a road, so gates stay open):
- **Solmera** (capital): 5-wide cross, central plaza with four 2 x 2 fountain basins (`T_WATER`,
  `town_fountains()`) at its corners, the castle block (`T_WALL`) north-west, the long shop row east.
- **Lunavell**: a 2-wide moat (`T_WATER`) ring around the plaza, bridged by the streets.
- **Pinehollow**: single trees every 4 tiles between the streets (forest village).
- **Tidewick**: a harbour basin (`T_WATER`) south-east.
- **Kharsand**: an oasis (`T_WATER`) north-east, 3 x 3 house blocks (`T_WALL`) every 7 tiles around it.

Every walkable tile inside each town is reachable from its temple (tests/test_zones.py). Shop tiles @512, first of
each row: Solmera (272, 253), Lunavell (118, 111), Pinehollow (426, 100), Tidewick (415, 408), Kharsand (129, 397).

## Waybell teleport and save points (data; the service itself is for ro-items2 / net)
`WAYBELL_TELEPORT = ((from town, to town, ryn), ...)`; `waybell_teleport_cost() -> int32 [NT, NT]` (-1 =
`WAYBELL_NO_ROUTE`, diagonal -1); arrival tile `waybell_teleport_dest() -> int32 [NT, 2]` = the destination's town
station (`T_STATION`, never a warp). Save points: `save_point_tiles() -> int32 [NT, 2]` = temple centres (what
`ro_waybell.phase` writes into `players.spawn`).

| from / to | Solmera | Lunavell | Pinehollow | Tidewick | Kharsand |
|---|---|---|---|---|---|
| Solmera | - | 1200 | 1200 | 1800 | 1200 |
| Lunavell | 1200 | - | 1200 | - | 1600 |
| Pinehollow | 1200 | 1200 | - | 1200 | - |
| Tidewick | 1800 | - | 1200 | - | - |
| Kharsand | 1200 | 1600 | - | - | - |

### Waybell teleport service (ro3-world3: in the JAX tick, `game/ro_waybell.py`)
Client message `{"t": "teleport", "town": <TOWNS index 0..4>}`, sent while the player stands within
`WAYBELL_TALK_RANGE = 2` tiles (Chebyshev) of the waybell tile (id 9) of the town they are in. Engine input
`inputs["teleport"]` int32 `[MAX_PLAYERS]`, -1 = none. Rule (`ro_waybell.teleport`): living, at the waybell, a route
in `WAYBELL_TELEPORT` from this town, `items.ryn >= fare`, and the destination's town station (`waybell_teleport_dest`,
id 10) free of living players and active monsters at the start of the waybell phase (lowest slot wins a shared
station). Then `players.pos` jumps to the station and the fare leaves `items.ryn`; otherwise nothing happens and
nothing is charged (the client may retry). The client already has everything it needs from `welcome.world`:
`towns[i].waybell` (where to stand), `waybell_teleport[].from/to/ryn/arrive` (the menu and the price), and sees the jump
as a position change of more than 2 tiles in one tick (camera snap, `you.warped` once ro3-wire lands it).
Lead lines: `state.empty_inputs`: `"teleport": jnp.full((MAX_PLAYERS,), -1, jnp.int32),  # waybell: destination town`;
`net/protocol._validate_ro`: `if t == "teleport": return {"t": t, "town": msg["town"]} if _is_int(msg.get("town"))
and 0 <= msg["town"] < len(world_layout.TOWNS) else None`; `net/server.py`: add `"teleport"` to `_RO_MESSAGES` and
`elif t == "teleport": pend["teleport"][slot] = msg["town"]` in `_on_message`. Until the input exists the phase is
save-only (`inputs.get("teleport")`).

### Minimap data (confirmed 2026-11-02, round 3)
`welcome.world.portals` (28 one-way legs `{from, to, zones}`), `welcome.world.warps` (the 4 dungeon legs then the
same 28 portal legs), `welcome.world.towns[i].waybell` / `.station`, `welcome.world.waybell_teleport[].arrive` and
`welcome.world.save_points` are all in `net/protocol.build_welcome_static` and round-trip through JSON
(`tests/test_protocol.py::test_welcome_static_tables`). The minimap draws portal `from` tiles as warp dots (id 8),
`arrive` tiles as station dots (id 10) and `towns[i].waybell` with the waybell sprite; nothing else is needed.

## `welcome.world` additions for zones (`net/protocol.py` is netperf2 / lead: lines to add)
```json
"zones": [{"id": 0, "name": "solmera", "kind": "town", "box": [225, 225, 287, 287], "levels": [0, 0],
           "music": 8}, ...],
"portals": [{"from": [191, 113], "to": [194, 113], "zones": [7, 8]}, ...],
"shops": [{"tile": [272, 253], "town": 0}, ...],
"waybell_teleport": [{"from": 0, "to": 1, "ryn": 1200, "arrive": [94, 113]}, ...],
"save_points": [[256, 256], ...]
```
Python for `world_table()` in `net/protocol.py`:
```python
zb = world_layout.zone_boxes().tolist()
zones = [{"id": i, "name": z[0], "kind": world_layout.ZONE_KINDS[z[1]], "box": zb[i], "levels": list(z[3]),
          "music": z[4]} for i, z in enumerate(world_layout.ZONES)]
psrc, pdst = world_layout.portal_warp_tiles()
pzone = np.asarray(world_layout.zone_of(jnp.asarray(np.concatenate([psrc, pdst])))).reshape(2, -1)
portals = [{"from": a, "to": b, "zones": [int(pzone[0, k]), int(pzone[1, k])]}
           for k, (a, b) in enumerate(zip(psrc.tolist(), pdst.tolist()))]
st, stown = world_layout.shop_tiles()
shops = [{"tile": t, "town": int(k)} for t, k in zip(st.tolist(), stown.tolist())]
arrive = world_layout.waybell_teleport_dest().tolist()
waybell_teleport = [{"from": a, "to": b, "ryn": z, "arrive": arrive[b]} for a, b, z in world_layout.WAYBELL_TELEPORT]
save_points = world_layout.save_point_tiles().tolist()
# then add "zones", "portals", "shops", "waybell_teleport", "save_points" to the "world" dict
```
`welcome.world.warps` already lists the portal legs too (it reads `warp_tiles()`, rows 4..).
Optional `snap.you.zone` (int, `zone_of(you.pos)`): one gather per viewer in `view.py`; the client can compute it
itself, so it is a convenience, not a requirement.
