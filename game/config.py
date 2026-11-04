"""Shared constants for the tile MMO rules engine. Owned by the lead.

All sizes are compile-time constants. Changing one recompiles the step.
"""
MAP = 512                 # tiles per side of the single floor
CELL = 8                  # spatial grid cell size in tiles
G = MAP // CELL           # grid cells per side
NCELLS = G * G
K_CELL = 8                # max players stored per grid cell (overflow dropped and counted)

MAX_PLAYERS = 512
MAX_MONSTERS = 4096
MAX_DROPS = 2048
INV_SLOTS = 16

# Directions indexed by the move input: 0 stay, 1..8 = S, SW, W, NW, N, NE, E, SE (x right, y down; RO facing = code - 1).
# Same tuple as game/ro_move.DIR_DELTAS9; the old 4-way client codes map through ro_move.LEGACY_TO_RO.
DIR_DELTAS = ((0, 0), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))

PLAYER_MAX_HP = 40  # level-1 Novice (classic RO, docs/ro_fidelity.md E12); living players cap at ro_stats.derived(world)["max_hp"]
PLAYER_ATTACK = 10        # damage a player deals per hit
PLAYER_ATTACK_COOLDOWN = 5  # ticks between player attacks
PLAYER_RESPAWN_TICKS = 0  # respawn happens the tick after death (0 = immediate on next tick)

MONSTER_AGGRO_RANGE = 8   # tiles (manhattan). Must be <= CELL so the 3x3 grid block covers it.
MONSTER_ATTACK_COOLDOWN = 8
MONSTER_RESPAWN_TICKS = 60

# Monster kinds: index into these tables.
MONSTER_KIND_RAT = 0
MONSTER_KIND_ORC = 1
# Since the RO world merge the kind table is game/data/monsters.py (20 kinds; 0 = Blobbin keeps the rat slot,
# 1 = Grukk Brute the orc slot). Kinds per tile come from game/ro_spawns.py by field band and dungeon.
from game.data.monsters import MONSTER_DAMAGE_TUPLE as _RO_DMG, MONSTER_HP_TUPLE as _RO_HP  # noqa: E402
MONSTER_HP = _RO_HP
N_MONSTER_KINDS = len(MONSTER_HP)
# Tuned from bot soaks on the 5-town map (2026-11-02) with regen, eating and bots that retreat
# below HUNT_RETREAT_HP: (3, 9) with cooldown 8 gives 0.4-0.5 deaths and 3.3-3.4 kills per bot
# per 1000 ticks on seeds 0-2 (1024 monsters per world). Deaths come from orcs chasing a
# retreating bot; rats alone cannot kill. (1, 3) gave 0.02 deaths, (4, 12) gave 0.6.
MONSTER_DAMAGE = _RO_DMG  # per-kind max hit; (3, 9) for the first two kinds, as tuned

DROP_TTL = 200            # ticks before a ground item disappears

# Loot roll: a uniform u in [0, 1) is compared against these cumulative thresholds in order.
# coin 40%, meat 25%, sword 1%, nothing otherwise (34%).
LOOT_COIN_MAX = 0.40
LOOT_MEAT_MAX = 0.65
LOOT_SWORD_MAX = 0.66

# Item ids. 0 means "empty slot" / "no item".
ITEM_NONE = 0
ITEM_GOLD_COIN = 1
ITEM_MEAT = 2
ITEM_SWORD = 3
NUM_ITEMS = 4
# Item ids: 0..3 legacy, 4..99 game/data/items.py, 100..199 raw materials, 200..299 crafted (game/data/craft.py).
ITEM_ID_LIMIT = 300

# Terrain ids stored in world["terrain"]. Blocking set: T_WATER, T_TREE, T_WALL, T_ROCK.
T_GRASS = 0
T_ROAD = 1
T_FLOOR = 2
T_TEMPLE = 3
T_WATER = 4
T_TREE = 5
T_WALL = 6
T_ROCK = 7
# 8..11 warp, waybell, station, cave live in game/data/terrain.py; T_SHOP is the NPC shop tile inside a town (ro_shop).
T_SHOP = 12

# Survival: living players regain REGEN_HP every REGEN_TICKS ticks; eating meat heals MEAT_HEAL.
REGEN_HP = 1
REGEN_TICKS = 4
MEAT_HEAL = 40

# Bots and spawns around town gates.
HUNT_RETREAT_HP = 16      # a bot below this HP walks home to rest (40% of PLAYER_MAX_HP)
GATE_APRON_TILES = 4      # monsters never spawn within this walking distance of a gate

# Networking (milestone 2)
TICK_MS = 100  # round 2: RO walks at 150 ms/cell, so a 100 ms tick keeps the speed accumulators honest
VIEW_W = 61
VIEW_H = 33
PATH_RADIUS = 32
PATH_STUCK_TICKS = 8
SAVE_EVERY_TICKS = 2400
PROTOCOL_VERSION = 2

# Protocol v2 visibility (game/view.py): entities per viewer per tick and viewer rows.
N_ENTITIES = MAX_PLAYERS + MAX_MONSTERS + MAX_DROPS
MAX_VIEWERS = MAX_PLAYERS
VIEW_K = 256
