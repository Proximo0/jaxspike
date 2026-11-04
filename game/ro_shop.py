"""RO NPC shops: buy and sell at shop tiles for ryn, per-town stock, Discount / Overcharge. Pure JAX.

Where: a living player is "at a shop" when it stands on, or next to (8 neighbours, RO's talk range), a T_SHOP
tile (game/data/terrain.py, placed by map_gen in each town's shop row). If the terrain module has no T_SHOP (an
older world), the waybell tile and its 8 neighbours act as the shop instead. Which town's stock applies comes from
the town box the player stands in (ro_waybell.town_of_tile); outside every town nothing is sold.

Inputs (one-shot per tick, int32 [MAX_PLAYERS]):
  buy   item id to buy one unit of; 0 or -1 = none. Must be in the town's stock (SHOP_STOCK).
  sell  inventory slot to sell from; -1 = none. One unit of a stackable, the whole item otherwise; worn items and
        empty slots are refused.
Order inside the phase: sell, then buy (so selling loot can pay for a potion in the same tick).

Prices (classic): buy = BUY[id] * (100 - discount) // 100, sell = SELL[id] * (100 + overcharge) // 100 per unit,
where discount / overcharge = min(5 + 2 * level, 24) percent for a learned Discount / Overcharge level 1..10 (7 % at
1, 24 % at 10, iRO wiki), read from world["skills"]["learned"] when the skills block and the skill exist.
Ryn is capped at MAX_RYN. A buy needs the ryn and either a stack of the same stackable id below MAX_STACK (the
unit joins the lowest such slot) or an empty slot (the unit lands in the lowest one).
"""
import jax.numpy as jnp
import numpy as np

from game import ro_items, ro_waybell, world_layout
from game.config import INV_SLOTS, ITEM_NONE, MAP, MAX_PLAYERS
from game.data import items as I
from game.data import skills as skill_data
from game.data import terrain

T_SHOP = getattr(terrain, "T_SHOP", None)
SHOP_FALLBACK_WAYBELL = T_SHOP is None  # True only for terrain modules from before ro-world2's shop rows

# Per-town stock, index == world_layout.TOWNS index (Solmera, Lunavell, Pinehollow, Tidewick, Kharsand). RO flavour: the
# Solmera tool / weapon / armor dealers, Lunavell's magic shop, Pinehollow's bows, Tidewick's axes and heavy armor,
# Kharsand's daggers.
MAX_STOCK = 16
_TOOLS = (I.ITEM_RED_POTION, I.ITEM_ORANGE_POTION, I.ITEM_FLIT_FEATHER, I.ITEM_HOMING_FEATHER)
_STOCK = (
    _TOOLS + (I.ITEM_YELLOW_POTION, I.ITEM_KNIFE, I.ITEM_CUTTER, I.ITEM_MAIN_GAUCHE, I.ITEM_SWORD, I.ITEM_FALCHION,
              I.ITEM_HAT, I.ITEM_CAP, I.ITEM_COTTON_SHIRT, I.ITEM_ADVENTURER_SUIT, I.ITEM_GUARD, I.ITEM_SANDALS),
    _TOOLS + (I.ITEM_BLUE_POTION, I.ITEM_ROD, I.ITEM_WAND, I.ITEM_KNIFE, I.ITEM_HAT, I.ITEM_HOOD, I.ITEM_COTTON_SHIRT,
              I.ITEM_MANTLE, I.ITEM_SANDALS, I.ITEM_CLIP),
    _TOOLS + (I.ITEM_APPLE, I.ITEM_BOW, I.ITEM_COMPOSITE_BOW, I.ITEM_KNIFE, I.ITEM_CAP, I.ITEM_JACKET, I.ITEM_GUARD,
              I.ITEM_HOOD, I.ITEM_SANDALS, I.ITEM_MUFFLER),
    _TOOLS + (I.ITEM_YELLOW_POTION, I.ITEM_CLUB, I.ITEM_MACE, I.ITEM_AXE, I.ITEM_BATTLE_AXE, I.ITEM_CAP, I.ITEM_HELM,
              I.ITEM_ADVENTURER_SUIT, I.ITEM_BUCKLER, I.ITEM_SHIELD, I.ITEM_SHOES, I.ITEM_BOOTS),
    _TOOLS + (I.ITEM_KNIFE, I.ITEM_CUTTER, I.ITEM_MAIN_GAUCHE, I.ITEM_HAT, I.ITEM_JACKET, I.ITEM_HOOD, I.ITEM_SANDALS,
              I.ITEM_MUFFLER, I.ITEM_GLOVE, I.ITEM_BROOCH),
)
N_TOWNS = len(world_layout.TOWNS)
assert len(_STOCK) == N_TOWNS and all(len(s) <= MAX_STOCK and len(set(s)) == len(s) for s in _STOCK)
assert all(int(I.BUY[i]) > 0 for s in _STOCK for i in s), "every stocked item needs a buy price"
SHOP_STOCK = jnp.asarray([s + (ITEM_NONE,) * (MAX_STOCK - len(s)) for s in _STOCK], jnp.int32)  # [N_TOWNS, MAX_STOCK]
_IN_STOCK_NP = np.zeros((N_TOWNS, I.N_ITEMS), np.bool_)
for _t, _s in enumerate(_STOCK):
    _IN_STOCK_NP[_t, list(_s)] = True
IN_STOCK = jnp.asarray(_IN_STOCK_NP)  # bool [N_TOWNS, N_ITEMS]

SK_DISCOUNT = getattr(skill_data, "SK_DISCOUNT", None)
SK_OVERCHARGE = getattr(skill_data, "SK_OVERCHARGE", None)
_PIDX = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
_SIDX = jnp.arange(INV_SLOTS, dtype=jnp.int32)
_NEAR = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def empty_shop_inputs():
    """The two one-shot shop inputs: buy (item id, 0 = none) and sell (inventory slot, -1 = none)."""
    return {"buy": jnp.zeros((MAX_PLAYERS,), jnp.int32), "sell": jnp.full((MAX_PLAYERS,), -1, jnp.int32)}


def stock_for_client():
    """Per-town stock as plain Python for the welcome message: [{town, name, items: [id, ...]}]."""
    return [{"town": t, "name": world_layout.TOWN_NAMES[t], "items": list(s)} for t, s in enumerate(_STOCK)]


def _skill_pct(world, skill):
    """Percent from a learned merchant price skill: min(5 + 2 * level, 24) for level >= 1, else 0. int32 [P]."""
    if skill is None or "skills" not in world:
        return jnp.zeros((MAX_PLAYERS,), jnp.int32)
    lv = world["skills"]["learned"][:, skill].astype(jnp.int32)
    return jnp.where(lv > 0, jnp.minimum(5 + 2 * lv, 24), 0).astype(jnp.int32)


def discount_pct(world):
    return _skill_pct(world, SK_DISCOUNT)


def overcharge_pct(world):
    return _skill_pct(world, SK_OVERCHARGE)


def buy_price(world, item):
    """int32 [P]: ryn one unit of item id `item` [P] costs each player (Discount applied, at least 1)."""
    base = I.BUY[jnp.clip(item, 0, I.N_ITEMS - 1)]
    return jnp.maximum(base * (100 - discount_pct(world)) // 100, 1).astype(jnp.int32)


def sell_price(world, item):
    """int32 [P]: ryn one unit of item id `item` [P] fetches (Overcharge applied)."""
    base = I.SELL[jnp.clip(item, 0, I.N_ITEMS - 1)]
    return (base * (100 + overcharge_pct(world)) // 100).astype(jnp.int32)


def at_shop(world):
    """bool [P]: the player stands on or next to a shop tile (the waybell tile when T_SHOP is unknown)."""
    pos = world["players"]["pos"]
    tile = T_SHOP if T_SHOP is not None else terrain.T_WAYBELL
    near = jnp.zeros((MAX_PLAYERS,), jnp.bool_)
    for dx, dy in _NEAR:  # 9 constant offsets
        x = jnp.clip(pos[:, 0] + dx, 0, MAP - 1)
        y = jnp.clip(pos[:, 1] + dy, 0, MAP - 1)
        near = near | (world["terrain"][x, y] == tile)
    return near


def shop_town(world):
    """int32 [P]: the town whose stock the player may buy from, -1 when not at a shop or not alive."""
    p = world["players"]
    ok = p["active"] & ~p["dead"] & at_shop(world)
    return jnp.where(ok, ro_waybell.town_of_tile(p["pos"]), -1).astype(jnp.int32)


def _sell(world, inputs, town):
    pl, it = world["players"], world["items"]
    slot = inputs["sell"]
    valid = (slot >= 0) & (slot < INV_SLOTS)
    sl = jnp.clip(slot, 0, INV_SLOTS - 1)
    item = jnp.take_along_axis(pl["inv"], sl[:, None], axis=1)[:, 0]
    cnt = jnp.take_along_axis(it["inv_count"], sl[:, None], axis=1)[:, 0]
    worn = jnp.any(it["equip"] == sl[:, None], axis=1)
    do = valid & (town >= 0) & (item != ITEM_NONE) & (cnt > 0) & ~worn
    units = jnp.where(I.STACKABLE[item], 1, cnt)
    gain = jnp.where(do, sell_price(world, item) * units, 0)
    ryn = jnp.minimum(it["ryn"] + gain, I.MAX_RYN)
    mask = (_SIDX[None, :] == sl[:, None]) & do[:, None]
    new_cnt = it["inv_count"] - jnp.where(mask, units[:, None], 0)
    world = {**world, "items": {**it, "ryn": ryn, "inv_count": new_cnt}}
    return ro_items.clear_slots(world, mask & (new_cnt <= 0))


def _buy(world, inputs, town):
    pl, it = world["players"], world["items"]
    item = inputs["buy"]
    known = (item > 0) & (item < I.N_ITEMS)
    safe = jnp.clip(item, 0, I.N_ITEMS - 1)
    stocked = known & (town >= 0) & IN_STOCK[jnp.clip(town, 0, N_TOWNS - 1), safe]
    price = buy_price(world, safe)
    inv, cnt = pl["inv"], it["inv_count"]
    stack_ok = I.STACKABLE[safe][:, None] & (inv == safe[:, None]) & (cnt < I.MAX_STACK)
    empty = inv == ITEM_NONE
    has_stack, has_empty = stack_ok.any(axis=1), empty.any(axis=1)
    slot = jnp.where(has_stack, jnp.argmax(stack_ok, axis=1), jnp.argmax(empty, axis=1))
    do = stocked & (it["ryn"] >= price) & (has_stack | has_empty)
    row = jnp.where(do, _PIDX, MAX_PLAYERS)
    new_inv = inv.at[row, slot].set(safe, mode="drop")
    new_cnt = cnt.at[row, slot].add(1, mode="drop")
    ryn = it["ryn"] - jnp.where(do, price, 0)
    return {**world, "players": {**pl, "inv": new_inv}, "items": {**it, "inv_count": new_cnt, "ryn": ryn}}


def phase(world, inputs):
    """Shop phase: sell, then buy. Needs world["items"]; missing "buy"/"sell" keys mean no action."""
    none = empty_shop_inputs()
    inputs = {"buy": inputs.get("buy", none["buy"]), "sell": inputs.get("sell", none["sell"])}
    town = shop_town(world)
    world = _sell(world, inputs, town)
    return _buy(world, inputs, town)
