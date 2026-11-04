"""RO items phase: inventory counts, stack merge, zeny, use, equip, refine hook, drop, equipment totals.

Owned by INV (M3). All work is fixed-shape array ops: no Python `if` on traced values, no loops over players or
slots. `world["items"]` holds the per-slot data that the lead's `players["inv"]` (item ids) cannot:

  inv_count  int32 [MAX_PLAYERS, INV_SLOTS]      units in the slot, 0 when inv is ITEM_NONE
  inv_refine int32 [MAX_PLAYERS, INV_SLOTS]      0..MAX_REFINE
  inv_cards  int32 [MAX_PLAYERS, INV_SLOTS, 4]   card item ids per socket, 0 = empty
  equip      int32 [MAX_PLAYERS, 8]              inventory slot index worn in each equip slot, -1 = none
  zeny       int32 [MAX_PLAYERS]

Item ids stay in `world["players"]["inv"]` (loot.py writes pickups there, survival.py clears eaten MEAT).
`_normalize` repairs the parallel arrays after any such outside change.
"""
import jax.numpy as jnp

from game import ro_refine, ro_stats
from game.config import (
    DROP_TTL,
    INV_SLOTS,
    ITEM_GOLD_COIN,
    ITEM_NONE,
    MAX_DROPS,
    MAX_PLAYERS,
)
from game.data import items as item_data
from game.state import next_key

N_EQ = item_data.N_EQUIP_SLOTS
REFINE_ATK = jnp.array((0, 2, 3, 5, 7), jnp.int32)  # weapon atk per refine level, by weapon level 0..4

_PIDX = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
_SIDX = jnp.arange(INV_SLOTS, dtype=jnp.int32)


def make_state():
    """Fresh `world["items"]` block: empty counts, no refine, no cards, nothing worn (-1), no zeny."""
    return {
        "inv_count": jnp.zeros((MAX_PLAYERS, INV_SLOTS), jnp.int32),
        "inv_refine": jnp.zeros((MAX_PLAYERS, INV_SLOTS), jnp.int32),
        "inv_cards": jnp.zeros((MAX_PLAYERS, INV_SLOTS, item_data.MAX_CARDS), jnp.int32),
        "equip": jnp.full((MAX_PLAYERS, N_EQ), -1, jnp.int32),
        "zeny": jnp.zeros((MAX_PLAYERS,), jnp.int32),
    }


def empty_item_inputs():
    """The four one-shot item inputs, all -1 (no action). Each is an inventory slot index per player."""
    none = jnp.full((MAX_PLAYERS,), -1, jnp.int32)
    return {"use_item": none, "equip": none, "refine": none, "drop": none}


def item_of(world):
    """Item id per inventory slot: just `players["inv"]` [MAX_PLAYERS, INV_SLOTS]."""
    return world["players"]["inv"]


def _live(world):
    """Players that may act: active and not dead. bool [MAX_PLAYERS]."""
    p = world["players"]
    return p["active"] & ~p["dead"]


def _slot_arg(slot):
    """Return (valid, clipped) for an input slot index [MAX_PLAYERS]; out-of-range is invalid."""
    valid = (slot >= 0) & (slot < INV_SLOTS)
    return valid, jnp.clip(slot, 0, INV_SLOTS - 1)


def clear_slots(world, mask):
    """Empty the inventory slots where mask [MAX_PLAYERS, INV_SLOTS] is True.

    Zeroes id, count, refine and cards, and sets any equip pointer to a cleared slot to -1.
    Invariant after: a cleared slot has inv == 0, count == 0, refine == 0, cards == 0 and is not worn.
    """
    players, it = world["players"], world["items"]
    equip = it["equip"]
    sl = jnp.clip(equip, 0, INV_SLOTS - 1)
    hit = (equip >= 0) & jnp.take_along_axis(mask, sl, axis=1)
    new_it = {
        **it,
        "inv_count": jnp.where(mask, 0, it["inv_count"]),
        "inv_refine": jnp.where(mask, 0, it["inv_refine"]),
        "inv_cards": jnp.where(mask[..., None], 0, it["inv_cards"]),
        "equip": jnp.where(hit, -1, equip),
    }
    new_players = {**players, "inv": jnp.where(mask, ITEM_NONE, players["inv"])}
    return {**world, "players": new_players, "items": new_it}


def _normalize(world):
    """Make the parallel item arrays consistent with `players["inv"]`.

    Owns: (1) fresh pickups (id set, count 0) get count 1, empty slots get count/refine/cards 0; (2) GOLD_COIN
    slots become `count * ZENY_PER_COIN` zeny (clamped to MAX_ZENY) and are cleared; (3) stackable slots with the
    same id merge into the lowest such slot (clamped to MAX_STACK, overflow lost); (4) equip pointers to emptied or
    non-gear slots become -1 and a slot worn twice keeps only its lowest equip slot.
    Invariants after: inv == 0 <=> count == 0; zeny <= MAX_ZENY; one entry per stackable id per player.
    """
    players, it = world["players"], world["items"]
    inv = players["inv"]
    cnt = jnp.where(inv == ITEM_NONE, 0, jnp.maximum(it["inv_count"], 1))
    refine = jnp.where(inv == ITEM_NONE, 0, it["inv_refine"])
    cards = jnp.where((inv == ITEM_NONE)[..., None], 0, it["inv_cards"])
    world = {**world, "items": {**it, "inv_count": cnt, "inv_refine": refine, "inv_cards": cards}}

    # Coins -> zeny.
    coin = inv == ITEM_GOLD_COIN
    gain = jnp.sum(jnp.where(coin, cnt, 0), axis=1) * item_data.ZENY_PER_COIN
    zeny = jnp.minimum(world["items"]["zeny"] + gain, item_data.MAX_ZENY)
    world = {**world, "items": {**world["items"], "zeny": zeny}}
    world = clear_slots(world, coin)

    # Merge stackables into the lowest slot with the same id.
    inv = world["players"]["inv"]
    cnt = world["items"]["inv_count"]
    stack = item_data.STACKABLE[inv] & (inv != ITEM_NONE)
    same = stack[:, :, None] & stack[:, None, :] & (inv[:, :, None] == inv[:, None, :])
    first = jnp.argmax(same, axis=2)
    is_head = stack & (first == _SIDX[None, :])
    total = jnp.sum(jnp.where(same, cnt[:, None, :], 0), axis=2)
    new_cnt = jnp.where(is_head, jnp.minimum(total, item_data.MAX_STACK), cnt)
    world = {**world, "items": {**world["items"], "inv_count": new_cnt}}
    world = clear_slots(world, stack & ~is_head)

    # Equip pointers: must name a non-empty weapon/armor slot, no duplicates.
    inv = world["players"]["inv"]
    equip = world["items"]["equip"]
    sl = jnp.clip(equip, 0, INV_SLOTS - 1)
    kind = item_data.KIND[jnp.take_along_axis(inv, sl, axis=1)]
    ok = (equip >= 0) & ((kind == item_data.KIND_WEAPON) | (kind == item_data.KIND_ARMOR))
    earlier = jnp.tril(jnp.ones((N_EQ, N_EQ), jnp.bool_), k=-1)  # [e, e'] true for e' < e
    dup = jnp.any(
        earlier[None] & ok[:, None, :] & (equip[:, :, None] == equip[:, None, :]), axis=2
    )
    equip = jnp.where(ok & ~dup, equip, -1)
    return {**world, "items": {**world["items"], "equip": equip}}


def _use_item(world, inputs):
    """Use the slot named by inputs["use_item"] (alive players only).

    Consumable: hp += HEAL_HP capped at ro_stats.derived max_hp (HEAL_SP is
    ignored until an sp array exists), one unit consumed. Card: ro_refine.compound_card, one unit consumed only
    where it reports success. Other kinds: no-op. A slot reaching count 0 is cleared.
    """
    players, it = world["players"], world["items"]
    valid, sl = _slot_arg(inputs["use_item"])
    item = jnp.take_along_axis(players["inv"], sl[:, None], axis=1)[:, 0]
    cnt = jnp.take_along_axis(it["inv_count"], sl[:, None], axis=1)[:, 0]
    ok = valid & _live(world) & (cnt > 0)
    kind = item_data.KIND[item]

    eat = ok & (kind == item_data.KIND_CONSUMABLE)
    heal = jnp.where(eat, item_data.HEAL_HP[item], 0)
    hp = jnp.maximum(players["hp"], jnp.minimum(players["hp"] + heal, ro_stats.derived(world)["max_hp"]))
    world = {**world, "players": {**players, "hp": hp}}

    world, done = ro_refine.compound_card(world, sl, ok & (kind == item_data.KIND_CARD))
    used = eat | done
    mask = (_SIDX[None, :] == sl[:, None]) & used[:, None]
    it = world["items"]
    new_cnt = it["inv_count"] - mask.astype(jnp.int32)
    world = {**world, "items": {**it, "inv_count": new_cnt}}
    return clear_slots(world, mask & (new_cnt <= 0))


def job_of(world):
    """int32 [MAX_PLAYERS]: each player's job id (game/data/stats.py), Novice (0) for worlds without stats."""
    if "stats" not in world:
        return jnp.zeros((MAX_PLAYERS,), jnp.int32)
    return jnp.clip(world["stats"]["job"], 0, item_data.N_JOBS - 1).astype(jnp.int32)


def can_wear(world, item):
    """bool, item's shape: the player's job may wear item id `item` [MAX_PLAYERS, ...] (EQUIP_JOBS)."""
    job = job_of(world).reshape((MAX_PLAYERS,) + (1,) * (item.ndim - 1))
    return item_data.EQUIP_JOBS[jnp.clip(item, 0, item_data.N_ITEMS - 1), job]


def weapon_range(world):
    """int32 [MAX_PLAYERS]: attack range in cells of the worn weapon (1 melee and bare hands, 9 bows)."""
    equip = world["items"]["equip"][:, item_data.EQ_WEAPON]
    item = jnp.take_along_axis(world["players"]["inv"], jnp.clip(equip, 0, INV_SLOTS - 1)[:, None], axis=1)[:, 0]
    return jnp.where(equip >= 0, item_data.WEAPON_RANGE[item], 1).astype(jnp.int32)


def _equip(world, inputs):
    """Toggle-equip the weapon/armor in the slot named by inputs["equip"] (alive players only).

    A slot already worn anywhere is unequipped. Otherwise it goes to EQ_WEAPON (weapons), the equip slot equal to
    its armor slot type 1..5, or EQ_ACC1 (EQ_ACC2 when ACC1 is taken, replacing ACC2) for accessories, replacing
    the previous occupant. Items of any other kind or class are ignored, and so is wearing an item the player's job
    may not use (EQUIP_JOBS); taking one off is always allowed.
    """
    players, it = world["players"], world["items"]
    equip = it["equip"]
    valid, sl = _slot_arg(inputs["equip"])
    item = jnp.take_along_axis(players["inv"], sl[:, None], axis=1)[:, 0]
    cnt = jnp.take_along_axis(it["inv_count"], sl[:, None], axis=1)[:, 0]
    kind = item_data.KIND[item]
    cls = item_data.CLASS[item]
    ok = valid & _live(world) & (cnt > 0)

    worn = jnp.any(equip == sl[:, None], axis=1)
    is_weapon = kind == item_data.KIND_WEAPON
    is_armor = kind == item_data.KIND_ARMOR
    is_acc = is_armor & (cls == item_data.SLOT_ACCESSORY)
    acc_target = jnp.where(equip[:, item_data.EQ_ACC1] == -1, item_data.EQ_ACC1, item_data.EQ_ACC2)
    target = jnp.where(is_weapon, item_data.EQ_WEAPON, jnp.where(is_acc, acc_target, cls))
    gear = is_weapon | (is_armor & (cls >= item_data.SLOT_HEAD) & (cls <= item_data.SLOT_ACCESSORY))

    unequip = ok & gear & worn
    do = ok & gear & ~worn & can_wear(world, item)
    equip = jnp.where(unequip[:, None] & (equip == sl[:, None]), -1, equip)
    equip = equip.at[jnp.where(do, _PIDX, MAX_PLAYERS), jnp.clip(target, 0, N_EQ - 1)].set(
        sl, mode="drop"
    )
    return {**world, "items": {**it, "equip": equip}}


def _drop(world, inputs):
    """Drop one unit of the slot named by inputs["drop"] on the player's tile (alive players only).

    Free drop slots are ranked as in loot._spawn_drops: lowest free slot to the lowest dropping player; players
    beyond the free count keep their item. Stackables lose one unit; weapons/armor leave whole and their equip pointer becomes -1. When the
    drops block has the count/refine/cards fields the ground drop keeps the unit count (1 for a stackable, the
    slot's count otherwise), refine and cards; without them refine and cards are lost.
    """
    players, it, drops = world["players"], world["items"], world["drops"]
    valid, sl = _slot_arg(inputs["drop"])
    item = jnp.take_along_axis(players["inv"], sl[:, None], axis=1)[:, 0]
    cnt = jnp.take_along_axis(it["inv_count"], sl[:, None], axis=1)[:, 0]
    want = valid & _live(world) & (cnt > 0) & (item != ITEM_NONE)

    p_rank = jnp.cumsum(want) - 1
    free = ~drops["active"]
    f_rank = jnp.cumsum(free) - 1
    slot_of_rank = jnp.full((MAX_DROPS,), MAX_DROPS, jnp.int32).at[
        jnp.where(free, f_rank, MAX_DROPS)
    ].set(jnp.arange(MAX_DROPS, dtype=jnp.int32), mode="drop")
    slot = slot_of_rank.at[jnp.where(want, p_rank, MAX_DROPS)].get(mode="fill", fill_value=MAX_DROPS)
    dropped = want & (slot < MAX_DROPS)

    new_drops = {
        **drops,
        "pos": drops["pos"].at[slot].set(players["pos"], mode="drop"),
        "item": drops["item"].at[slot].set(item, mode="drop"),
        "active": drops["active"].at[slot].set(True, mode="drop"),
        "ttl": drops["ttl"].at[slot].set(DROP_TTL, mode="drop"),
    }
    if "count" in drops:
        ref = jnp.take_along_axis(it["inv_refine"], sl[:, None], axis=1)[:, 0]
        crd = jnp.take_along_axis(it["inv_cards"], sl[:, None, None], axis=1)[:, 0]
        new_drops["count"] = drops["count"].at[slot].set(jnp.where(item_data.STACKABLE[item], 1, cnt), mode="drop")
        new_drops["refine"] = drops["refine"].at[slot].set(ref, mode="drop")
        new_drops["cards"] = drops["cards"].at[slot].set(crd, mode="drop")
    world = {**world, "drops": new_drops}

    mask = (_SIDX[None, :] == sl[:, None]) & dropped[:, None]
    lose = jnp.where(item_data.STACKABLE[item], 1, cnt)
    new_cnt = it["inv_count"] - jnp.where(mask, lose[:, None], 0)
    world = {**world, "items": {**it, "inv_count": new_cnt}}
    return clear_slots(world, mask & (new_cnt <= 0))


def equipped(world):
    """Totals from worn gear, int32 per player: weapon_atk, weapon_elem, armor_def, armor_elem, card_bonus.

    weapon_atk = ATK[id] + REFINE_ATK[weapon_level] * refine; armor_def sums DEF[id] + refine over the seven armor
    equip slots; elements come from the weapon and the body item; all 0 for empty equip slots. card_bonus
    [MAX_PLAYERS, N_BONUS] sums the item's own BONUS_VALUE and the CARD_VALUE of cards in sockets below SLOTS[id],
    in one scatter-add over a flattened [MAX_PLAYERS, 8 * (1 + MAX_CARDS)] list.
    """
    players, it = world["players"], world["items"]
    equip = it["equip"]
    has = equip >= 0
    sl = jnp.clip(equip, 0, INV_SLOTS - 1)
    item = jnp.where(has, jnp.take_along_axis(players["inv"], sl, axis=1), ITEM_NONE)  # [P, 8]
    ref = jnp.where(has, jnp.take_along_axis(it["inv_refine"], sl, axis=1), 0)
    cards = jnp.where(
        has[..., None],
        jnp.take_along_axis(it["inv_cards"], sl[..., None], axis=1),
        0,
    )  # [P, 8, MAX_CARDS]

    w_item, w_ref = item[:, item_data.EQ_WEAPON], ref[:, item_data.EQ_WEAPON]
    weapon_atk = item_data.ATK[w_item] + REFINE_ATK[item_data.WEAPON_LEVEL[w_item]] * w_ref
    armor_def = jnp.sum(
        jnp.where(has, item_data.DEF[item] + ref, 0)[:, item_data.EQ_HEAD:], axis=1
    )

    sockets = (
        jnp.arange(item_data.MAX_CARDS, dtype=jnp.int32)[None, None, :]
        < item_data.SLOTS[item][..., None]
    )
    card_ok = sockets & (cards != 0)
    bonus_ids = jnp.concatenate(
        [item_data.BONUS[item][..., None], item_data.CARD_BONUS[cards]], axis=2
    )
    values = jnp.concatenate(
        [item_data.BONUS_VALUE[item][..., None], jnp.where(card_ok, item_data.CARD_VALUE[cards], 0)],
        axis=2,
    )
    width = N_EQ * (1 + item_data.MAX_CARDS)
    rows = jnp.broadcast_to(_PIDX[:, None], (MAX_PLAYERS, width))
    card_bonus = jnp.zeros((MAX_PLAYERS, item_data.N_BONUS), jnp.int32).at[
        rows, bonus_ids.reshape(MAX_PLAYERS, width)
    ].add(values.reshape(MAX_PLAYERS, width))

    return {
        "weapon_atk": weapon_atk.astype(jnp.int32),
        "weapon_elem": item_data.ELEMENT[w_item],
        "armor_def": armor_def.astype(jnp.int32),
        "armor_elem": item_data.ELEMENT[item[:, item_data.EQ_BODY]],
        "card_bonus": card_bonus,
    }


def phase(world, inputs):
    """Items phase: normalize -> use_item -> equip -> refine -> drop -> normalize.

    Needs `world["items"]` (make_state) and the four item input keys (empty_item_inputs). Refine uses a key from
    next_key, so the world's rng advances once per call. Pure; works under jit and vmap over worlds.
    """
    world = _normalize(world)
    world = _use_item(world, inputs)
    world = _equip(world, inputs)
    key, world = next_key(world)
    world = ro_refine.refine(world, inputs, key)
    world = _drop(world, inputs)
    return _normalize(world)
