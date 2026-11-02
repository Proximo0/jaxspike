"""Shared constants for the tile MMO rules engine. Owned by the lead.

All sizes are compile-time constants. Changing one recompiles the step.
"""
MAP = 256                 # tiles per side of the single floor
CELL = 8                  # spatial grid cell size in tiles
G = MAP // CELL           # grid cells per side
NCELLS = G * G
K_CELL = 8                # max players stored per grid cell (overflow dropped and counted)

MAX_PLAYERS = 512
MAX_MONSTERS = 4096
MAX_DROPS = 2048
INV_SLOTS = 16

# Directions indexed by the move input: 0 stay, 1 north, 2 south, 3 west, 4 east.
DIR_DELTAS = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))

PLAYER_MAX_HP = 150
PLAYER_ATTACK = 10        # damage a player deals per hit
PLAYER_ATTACK_COOLDOWN = 5  # ticks between player attacks
PLAYER_RESPAWN_TICKS = 0  # respawn happens the tick after death (0 = immediate on next tick)

MONSTER_AGGRO_RANGE = 8   # tiles (manhattan). Must be <= CELL so the 3x3 grid block covers it.
MONSTER_ATTACK_COOLDOWN = 8
MONSTER_RESPAWN_TICKS = 60

# Monster kinds: index into these tables.
MONSTER_KIND_RAT = 0
MONSTER_KIND_ORC = 1
MONSTER_HP = (40, 120)
# Tuned from bot soaks (2026-11-02): with (2, 7) and cooldown 6 a bot died 1.5 times per
# 1000 ticks; (1, 3) with cooldown 8 gives about 0.5 deaths and 8 kills per bot per 1000 ticks
# on seeds 0-2. No healing exists yet, so incoming damage is the only lever on deaths.
MONSTER_DAMAGE = (1, 3)

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
