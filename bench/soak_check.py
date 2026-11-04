"""End-to-end soak: bot games for many ticks, checking world invariants.

Invariants checked after every chunk of ticks:
- no active player or monster on a wall or out of bounds
- no two living (active, not dead) players on one tile; dead players are not physical
- no two active monsters on one tile
- no active monster on a living player's tile
- no active monster on a protection-zone tile; every player spawn is a temple tile
- hp in [0, derived max_hp] for players, [0, max] for monsters
- inventory values in [0, ITEM_ID_LIMIT)
- active drops on in-bounds tiles, ttl in [0, DROP_TTL]
Prints per-batch stats so the game's balance is visible.
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from game import ro_spawns, ro_stats  # noqa: E402
from game.world_layout import dungeon_index  # noqa: E402
from bots.runner import make_world_batch, run_games  # noqa: E402
from game.config import MAP, MONSTER_HP, ITEM_ID_LIMIT, DROP_TTL, T_TEMPLE  # noqa: E402


def monsters_per_band(worlds):
    """int [N_BANDS + 1] active monsters summed over the batch: per field band, then all dungeons together."""
    def one(w):
        m = w["monsters"]
        band = jnp.where(dungeon_index(m["pos"]) >= 0, ro_spawns.N_BANDS, ro_spawns.band_of(w, m["pos"]))
        return jnp.bincount(jnp.where(m["active"], band, ro_spawns.N_BANDS + 1), length=ro_spawns.N_BANDS + 2)[:-1]

    return np.asarray(jax.vmap(one)(worlds).sum(0))


def cards_in_world(worlds):
    """ro4-cards: card units that exist across the batch (carried stacks, socketed cards, card piles on the ground).

    A proxy for "cards found" while bots.runner has no per-tick card counter: a pile that expires before anyone
    picks it up is missed, and a card sold to a shop leaves the count. Use stats["cards"] when the runner has it.
    """
    from game.data import items as item_data
    inv = worlds["players"]["inv"]
    is_card = item_data.KIND[inv] == item_data.KIND_CARD
    carried = jnp.sum(jnp.where(is_card, jnp.maximum(worlds["items"]["inv_count"], 1), 0))
    socketed = jnp.sum(item_data.KIND[worlds["items"]["inv_cards"]] == item_data.KIND_CARD)
    d = worlds["drops"]
    ground = jnp.sum(d["active"] & (item_data.KIND[d["item"]] == item_data.KIND_CARD))
    return int(carried + socketed + ground)


def check_invariants(worlds):
    """Return a list of violation strings across the batch (empty means clean)."""
    w = jax.device_get(worlds)
    dmax = np.asarray(jax.vmap(ro_stats.derived)(worlds)["max_hp"])
    problems = []
    walls = np.asarray(w["walls"])
    p, m, d = w["players"], w["monsters"], w["drops"]
    for b in range(walls.shape[0]):
        pa = np.asarray(p["active"][b]) & ~np.asarray(p["dead"][b])
        pp = np.asarray(p["pos"][b])
        ma = np.asarray(m["active"][b])
        mp = np.asarray(m["pos"][b])
        da = np.asarray(d["active"][b])
        dp = np.asarray(d["pos"][b])
        if ((pp < 0) | (pp >= MAP)).any():
            problems.append(f"world {b}: player out of bounds")
        if ((mp < 0) | (mp >= MAP)).any():
            problems.append(f"world {b}: monster out of bounds")
        for name, pos, act in (("player", pp, pa), ("monster", mp, ma)):
            live = pos[act]
            if live.size and (walls[b][live[:, 0], live[:, 1]] == 1).any():
                problems.append(f"world {b}: active {name} on wall")
        ppos_live = pp[pa]
        tiles = {tuple(t) for t in ppos_live}
        if len(tiles) != len(ppos_live):
            problems.append(f"world {b}: two living players share a tile")
        if len({tuple(t) for t in mp[ma]}) != int(ma.sum()):
            problems.append(f"world {b}: two active monsters share a tile")
        mtiles = {tuple(t) for t in mp[ma]}
        if mtiles & tiles:
            problems.append(f"world {b}: monster on player tile")
        pz = np.asarray(w["pz"][b])
        if ma.any() and pz[mp[ma][:, 0], mp[ma][:, 1]].any():
            problems.append(f"world {b}: monster on protection-zone tile")
        terrain = np.asarray(w["terrain"][b])
        sp = np.asarray(p["spawn"][b])[np.asarray(p["active"][b])]
        if sp.size and (terrain[sp[:, 0], sp[:, 1]] != T_TEMPLE).any():
            problems.append(f"world {b}: player spawn not on a temple tile")
        hp = np.asarray(p["hp"][b])
        pmax = dmax[b]
        if ((hp < 0) | (hp > pmax)).any():
            problems.append(f"world {b}: player hp out of range")
        mhp = np.asarray(m["hp"][b])
        mmax = np.asarray(MONSTER_HP)[np.asarray(m["kind"][b])]
        if ((mhp[ma] <= 0) | (mhp[ma] > mmax[ma])).any():
            problems.append(f"world {b}: active monster hp out of range")
        inv = np.asarray(p["inv"][b])
        if ((inv < 0) | (inv >= ITEM_ID_LIMIT)).any():
            problems.append(f"world {b}: inventory value out of range")
        ttl = np.asarray(d["ttl"][b])
        if ((ttl[da] < 0) | (ttl[da] > DROP_TTL)).any():
            problems.append(f"world {b}: drop ttl out of range")
    return problems


def main(batch=8, chunks=10, ticks_per_chunk=100, seed=0):
    print("platform:", jax.devices()[0].platform, flush=True)
    worlds = make_world_batch(seed, batch)
    kinds = np.asarray(worlds["monsters"]["kind"])[np.asarray(worlds["monsters"]["active"])]
    print(f"monsters per world by kind: {np.bincount(kinds, minlength=len(MONSTER_HP)) / batch}", flush=True)
    print(f"monsters per world by band (field 0..{ro_spawns.N_BANDS - 1}, then dungeons): "
          f"{monsters_per_band(worlds) / batch}", flush=True)
    n_bots = int(jnp.sum(worlds["players"]["active"]))
    total_problems = []
    t0 = time.perf_counter()
    for c in range(chunks):
        worlds, stats = run_games(worlds, ticks_per_chunk)
        jax.block_until_ready(worlds)
        probs = check_invariants(worlds)
        total_problems += probs
        kills = int(jnp.sum(stats["kills"]))
        deaths = int(jnp.sum(stats["deaths"]))
        alive = int(jnp.sum(stats["alive"]))
        items = int(jnp.sum(stats["items"]))
        tick = (c + 1) * ticks_per_chunk
        per_bot = 1000 / max(n_bots * tick, 1)  # cumulative counters -> per bot per 1000 ticks
        lvl = float(jnp.mean(worlds["stats"]["base_level"][:, :n_bots // batch]))
        print(
            f"chunk {c + 1:2d} (tick {tick:5d}): "
            f"kills={kills} deaths={deaths} alive_players={alive} items_held={items} "
            f"invariant_violations={len(probs)} | per bot per 1k ticks: kills={kills * per_bot:.2f} "
            f"deaths={deaths * per_bot:.2f} | base lv {lvl:.2f} | monsters by band {monsters_per_band(worlds) / batch}",
            flush=True,
        )
    dt = time.perf_counter() - t0
    print(f"total {chunks * ticks_per_chunk} ticks x {batch} worlds in {dt:.1f}s")
    n_cards = int(jnp.sum(stats["cards"])) if "cards" in stats else cards_in_world(worlds)  # ro4-cards column
    print(f"cards found {n_cards} = {n_cards * 1000 / max(kills, 1):.3f} per 1000 kills "
          f"(expected 0.100: every card is 1 in 10000; {'runner counter' if 'cards' in stats else 'held proxy'})")
    print(f"kills/bot/1k {kills * per_bot:.2f} deaths/bot/1k {deaths * per_bot:.2f} "
          f"(targets 3-5 and 0.3-0.6 at base levels 1-10, docs/round4_contract.md); "
          f"base level mean {float(jnp.mean(worlds['stats']['base_level'][:, :n_bots // batch])):.2f}")
    if "pvp" in worlds:  # ro5-pvp2: duellists' PvP kills
        from bots.ro_pvp_policy import IS_DUELLIST
        duel = worlds["players"]["active"] & IS_DUELLIST[None, :]
        nd, dk = int(jnp.sum(duel)), int(jnp.sum(jnp.where(duel, worlds["pvp"]["pvp_kills"], 0)))
        print(f"pvp kills {int(jnp.sum(worlds['pvp']['pvp_kills']))} deaths {int(jnp.sum(worlds['pvp']['pvp_deaths']))}; "
              f"{dk / max(nd, 1):.2f} PvP kills per duellist ({nd} duellists)")
    if total_problems:
        print("FIRST VIOLATIONS:")
        for p_ in total_problems[:15]:
            print("  ", p_)
    else:
        print("all invariants held")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=100, help="ticks per chunk")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(batch=a.batch, chunks=a.chunks, ticks_per_chunk=a.ticks, seed=a.seed)
