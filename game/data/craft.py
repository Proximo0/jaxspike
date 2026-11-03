"""M4 gathering and crafting tables. Owned by the ro-craft team (see docs/craft_brief.md).

Plain tuples turned into jnp arrays at import. Index by id. Fractions are x1000 fixed point.
Item ids: raw materials 100..199, crafted products 200..299 (config.py keeps 0..3 for M0 items;
M3 owns everything else). Node kinds 0..7, craft skills 0..5, recipes 1..N_RECIPES-1 (0 = none).
"""
import jax.numpy as jnp
import numpy as np

from game.config import MAP

# ---- sizes -------------------------------------------------------------------------------------
MAX_NODES = 2048
CRAFT_MAX_LEVEL = 50
GATHER_RANGE = 1          # manhattan distance from a player to a node it can gather from (own tile or 4-neighbour)
RECIPE_INPUTS_MAX = 4

# ---- craft skills (index into world["craft"]["level"][:, skill]) ----------------------------------
SKILL_MINING = 0
SKILL_SMITHING = 1
SKILL_FISHING = 2
SKILL_COOKING = 3
SKILL_HERBALISM = 4
SKILL_WOODCUTTING = 5
N_CRAFT_SKILLS = 6
SKILL_NAMES = ("mining", "smithing", "fishing", "cooking", "herbalism", "woodcutting")

# ---- node kinds (world["craft"]["nodes"]["kind"]) ------------------------------------------------
NODE_COPPER = 0
NODE_IRON = 1
NODE_COAL = 2
NODE_FISH = 3
NODE_HERB = 4
NODE_TREE = 5
NODE_FURNACE = 6          # station: smithing recipes
NODE_FIRE = 7             # station: cooking recipes
N_NODE_KINDS = 8
N_GATHER_KINDS = 6        # kinds 0..5 are gatherable; 6..7 are stations
NODE_NAMES = ("copper ore", "iron ore", "coal", "fishing spot", "herb", "tree", "furnace", "cooking fire")
NODE_VIEW_KIND_BASE = 400  # view kind 400 + node kind: drops are view kinds 100 + item id (100..399), game/view.py

# ---- item ids ------------------------------------------------------------------------------------
ITEM_COPPER_ORE = 100
ITEM_IRON_ORE = 101
ITEM_COAL = 102
ITEM_RAW_FISH = 103
ITEM_HERB = 104
ITEM_LOG = 105
ITEM_COPPER_BAR = 200
ITEM_IRON_BAR = 201
ITEM_STEEL_BAR = 202
ITEM_COOKED_FISH = 203
ITEM_HERB_POTION = 204
ITEM_PLANK = 205
ITEM_COPPER_DAGGER = 206
ITEM_IRON_SWORD = 207
ITEM_FISH_STEW = 208
CRAFT_ITEM_IDS = (ITEM_COPPER_ORE, ITEM_IRON_ORE, ITEM_COAL, ITEM_RAW_FISH, ITEM_HERB, ITEM_LOG,
                  ITEM_COPPER_BAR, ITEM_IRON_BAR, ITEM_STEEL_BAR, ITEM_COOKED_FISH, ITEM_HERB_POTION,
                  ITEM_PLANK, ITEM_COPPER_DAGGER, ITEM_IRON_SWORD, ITEM_FISH_STEW)
ITEM_NAMES = {
    ITEM_COPPER_ORE: "copper ore", ITEM_IRON_ORE: "iron ore", ITEM_COAL: "coal", ITEM_RAW_FISH: "raw fish",
    ITEM_HERB: "herb", ITEM_LOG: "log", ITEM_COPPER_BAR: "copper bar", ITEM_IRON_BAR: "iron bar",
    ITEM_STEEL_BAR: "steel bar", ITEM_COOKED_FISH: "cooked fish", ITEM_HERB_POTION: "herb potion",
    ITEM_PLANK: "plank", ITEM_COPPER_DAGGER: "copper dagger", ITEM_IRON_SWORD: "iron sword",
    ITEM_FISH_STEW: "fish stew",
}

# ---- per-node-kind tables, indexed by node kind ---------------------------------------------------
# Stations (kinds 6, 7) are never gathered: skill -1, product 0, chance 0.
NODE_SKILL = (SKILL_MINING, SKILL_MINING, SKILL_MINING, SKILL_FISHING, SKILL_HERBALISM, SKILL_WOODCUTTING, -1, -1)
NODE_PRODUCT = (ITEM_COPPER_ORE, ITEM_IRON_ORE, ITEM_COAL, ITEM_RAW_FISH, ITEM_HERB, ITEM_LOG, 0, 0)
NODE_LEVEL_REQ = (1, 10, 5, 1, 1, 1, 0, 0)
NODE_CHANCE_BASE = (500, 350, 400, 450, 600, 500, 0, 0)   # x1000 success chance at the required level
NODE_CHANCE_PER_LEVEL = 15                                 # x1000 added per level above the requirement
NODE_CHANCE_MAX = 950
NODE_EXP = (10, 25, 15, 10, 8, 10, 0, 0)
NODE_RESPAWN = (40, 80, 60, 1, 30, 50, 0, 0)               # ticks a gathered node stays inactive (fish: 1)
# How many nodes of each kind place_nodes tries to place; slots are contiguous per kind in this order.
NODE_COUNT = (400, 250, 250, 300, 400, 350, 5, 5)
NODE_SLOT_START = tuple(int(s) for s in np.concatenate([[0], np.cumsum(NODE_COUNT)[:-1]]))
assert sum(NODE_COUNT) <= MAX_NODES
# Stations: one furnace and one cooking fire per town, STATION_GAP tiles beyond the temple's east / west edge
# on the town's centre row (floor tiles inside the wall ring, so they sit on pz).
STATION_GAP = 2

# ---- skill levels --------------------------------------------------------------------------------
# LEVEL_EXP[L] is the total exp needed to be level L (L = 1..CRAFT_MAX_LEVEL); level = count(exp >= LEVEL_EXP) .
LEVEL_EXP = tuple(50 * (lvl - 1) ** 2 for lvl in range(1, CRAFT_MAX_LEVEL + 1))

# ---- recipes (row 0 = none) ----------------------------------------------------------------------
# (inputs x4 with 0 padding, output, skill, level req, exp, station node kind or -1 for anywhere)
_R = (
    ((0, 0, 0, 0), 0, 0, 0, 0, -1),
    ((ITEM_COPPER_ORE, ITEM_COPPER_ORE, 0, 0), ITEM_COPPER_BAR, SKILL_SMITHING, 1, 15, NODE_FURNACE),
    ((ITEM_IRON_ORE, ITEM_COAL, 0, 0), ITEM_IRON_BAR, SKILL_SMITHING, 10, 30, NODE_FURNACE),
    ((ITEM_IRON_BAR, ITEM_COAL, ITEM_COAL, 0), ITEM_STEEL_BAR, SKILL_SMITHING, 20, 60, NODE_FURNACE),
    ((ITEM_RAW_FISH, 0, 0, 0), ITEM_COOKED_FISH, SKILL_COOKING, 1, 10, NODE_FIRE),
    ((ITEM_HERB, ITEM_HERB, 0, 0), ITEM_HERB_POTION, SKILL_HERBALISM, 1, 12, -1),
    ((ITEM_LOG, 0, 0, 0), ITEM_PLANK, SKILL_WOODCUTTING, 1, 8, -1),
    ((ITEM_COPPER_BAR, ITEM_COPPER_BAR, ITEM_PLANK, 0), ITEM_COPPER_DAGGER, SKILL_SMITHING, 5, 40, NODE_FURNACE),
    ((ITEM_IRON_BAR, ITEM_IRON_BAR, ITEM_IRON_BAR, ITEM_PLANK), ITEM_IRON_SWORD, SKILL_SMITHING, 15, 90, NODE_FURNACE),
    ((ITEM_COOKED_FISH, ITEM_HERB, 0, 0), ITEM_FISH_STEW, SKILL_COOKING, 10, 35, NODE_FIRE),
)
N_RECIPES = len(_R)
RECIPE_NONE = 0
RECIPE_COPPER_BAR, RECIPE_IRON_BAR, RECIPE_STEEL_BAR, RECIPE_COOKED_FISH, RECIPE_HERB_POTION = 1, 2, 3, 4, 5
RECIPE_PLANK, RECIPE_COPPER_DAGGER, RECIPE_IRON_SWORD, RECIPE_FISH_STEW = 6, 7, 8, 9

RECIPE_INPUTS = jnp.asarray([r[0] for r in _R], jnp.int32)      # [N_RECIPES, 4]
RECIPE_OUTPUT = jnp.asarray([r[1] for r in _R], jnp.int32)      # [N_RECIPES]
RECIPE_SKILL = jnp.asarray([r[2] for r in _R], jnp.int32)
RECIPE_LEVEL = jnp.asarray([r[3] for r in _R], jnp.int32)
RECIPE_EXP = jnp.asarray([r[4] for r in _R], jnp.int32)
RECIPE_STATION = jnp.asarray([r[5] for r in _R], jnp.int32)     # node kind or -1

NODE_SKILL_A = jnp.asarray(NODE_SKILL, jnp.int32)
NODE_PRODUCT_A = jnp.asarray(NODE_PRODUCT, jnp.int32)
NODE_LEVEL_REQ_A = jnp.asarray(NODE_LEVEL_REQ, jnp.int32)
NODE_CHANCE_BASE_A = jnp.asarray(NODE_CHANCE_BASE, jnp.int32)
NODE_EXP_A = jnp.asarray(NODE_EXP, jnp.int32)
NODE_RESPAWN_A = jnp.asarray(NODE_RESPAWN, jnp.int32)
LEVEL_EXP_A = jnp.asarray(LEVEL_EXP, jnp.int32)

assert all(r[2] >= 0 and r[3] <= CRAFT_MAX_LEVEL for r in _R)
assert MAP > 0
