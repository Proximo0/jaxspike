"""Terrain ids added by the RO world (ro/world). `game/config.py` is lead-owned, so they live here.

Ids 0..7 are in config (grass, road, floor, temple, water, tree, wall, rock). None of these
new ids blocks movement. Only T_WARP teleports: a player standing on it is moved to the warp's
destination, which is always a T_STATION tile (arrival pads never teleport, so no ping-pong).
Lines for the lead to copy into config.py when merging:
    T_WARP = 8      # warp pad: stepping on it teleports to its destination (game/ro_warps.py)
    T_KAFRA = 9     # kafra NPC tile inside a town: save point / storage (protection zone)
    T_STATION = 10  # arrival pad of a warp (inside dungeons, outside their entrances, in towns)
    T_CAVE = 11     # dungeon floor (walkable, never protection zone)
"""
T_WARP = 8
T_KAFRA = 9
T_STATION = 10
T_CAVE = 11
N_TERRAIN = 12

TERRAIN_NAMES = ("grass", "road", "floor", "temple", "water", "tree", "wall", "rock",
                 "warp", "kafra", "station", "cave")
assert len(TERRAIN_NAMES) == N_TERRAIN
