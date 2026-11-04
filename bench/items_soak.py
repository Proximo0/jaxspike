"""Items soak: bot games with the M3 item systems wired into the tick, checking item invariants.

Harness tick (loot._spawn_drops replaced by the RO drop tables):
  movement -> monsters -> combat -> ro_drops.roll_drops + place_drops -> loot._pickup -> loot._decay
  -> survival -> ro_items.phase -> tick + 1,
with inputs {**bot_inputs, **item_inputs}.

Invariants after every chunk: the base soak_check.check_invariants (minus its NUM_ITEMS inventory check, which
predates the item table) plus
- inv ids in [0, N_ITEMS); inv_count >= 1 iff inv != 0; inv_count <= MAX_STACK; refine in 0..MAX_REFINE
- cards are 0 or a card item with socket index < SLOTS of the item holding them (and 0 in empty slots)
- equip in -1..INV_SLOTS-1, pointing at a weapon/armor of the right slot type; no slot equipped twice
- ryn in 0..MAX_RYN

"Items picked" is the running total of drops taken from the ground (drops deactivated by loot._pickup), by kind of
the item taken. Ground drops carry one unit each until the lead adds drops["count"], so this equals units that
entered inventories. Coins are listed separately (they become ryn the same tick). Rates are per bot per 1000 ticks.
Exit 0 with "soak: clean", else exit 1 with the violation list.
"""
import functools
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from bench.soak_check import check_invariants as base_check  # noqa: E402
from bots.policies import bot_inputs  # noqa: E402
from bots.ro_items_policy import item_inputs  # noqa: E402
from bots.runner import make_world_batch  # noqa: E402
from game import combat, loot, monsters, movement, ro_drops, ro_items, survival  # noqa: E402
from game.config import INV_SLOTS, MAX_PLAYERS  # noqa: E402
from game.data import items as I  # noqa: E402
from game.state import next_key  # noqa: E402

KIND_NAMES = ("etc", "consumable", "weapon", "armor", "card")
_KIND_IDS = (I.KIND_ETC, I.KIND_CONSUMABLE, I.KIND_WEAPON, I.KIND_ARMOR, I.KIND_CARD)
N_STATS = 11  # picked by kind (6, index = kind id), coins, equips, refine ok, refine broke, cards compounded


def harness_tick(world, inputs):
    """One tick with the RO drop tables and the items phase. Returns (world, stats int32 [N_STATS])."""
    world = {**world, "monsters": {**world["monsters"], "died": jnp.zeros_like(world["monsters"]["died"])}}
    world = movement.move_players(world, inputs)
    world = monsters.monster_ai(world)
    world = combat.combat(world, inputs)
    sub, world = next_key(world)
    drop_items, drop_counts, ryn = ro_drops.roll_drops(world, sub)
    world = ro_drops.place_drops(world, drop_items, drop_counts, ryn)

    before = world["drops"]
    world = loot._pickup(world)
    taken = before["active"] & ~world["drops"]["active"]
    taken_item = jnp.where(taken, before["item"], 0)
    world = loot._decay(world)
    world = survival.survival(world, inputs)

    it0, inv0 = world["items"], world["players"]["inv"]
    world = ro_items.phase(world, inputs)
    it1, inv1 = world["items"], world["players"]["inv"]

    picked = jnp.zeros((6,), jnp.int32).at[I.KIND[taken_item]].add(taken.astype(jnp.int32))
    coins = (taken_item == I.ITEM_GOLD_COIN).sum().astype(jnp.int32)
    picked = picked.at[I.KIND_ETC].add(-coins)
    equips = ((it1["equip"] != it0["equip"]) & (it1["equip"] >= 0)).sum().astype(jnp.int32)
    pidx = jnp.arange(MAX_PLAYERS)
    rslot = jnp.clip(inputs["refine"], 0, INV_SLOTS - 1)
    attempt = (inputs["refine"] >= 0) & (inv0[pidx, rslot] != 0)
    ok = attempt & (inv1[pidx, rslot] == inv0[pidx, rslot]) & (it1["inv_refine"][pidx, rslot] == it0["inv_refine"][pidx, rslot] + 1)
    broke = attempt & (inv1[pidx, rslot] == 0)
    n0 = (it0["inv_cards"] > 0).sum(axis=(1, 2))
    n1 = (it1["inv_cards"] > 0).sum(axis=(1, 2))
    compounded = jnp.maximum(n1 - n0, 0).sum().astype(jnp.int32)
    stats = jnp.concatenate([picked, jnp.stack([coins, equips, ok.sum(), broke.sum(), compounded]).astype(jnp.int32)])
    return {**world, "tick": world["tick"] + 1}, stats


def _tick_all(carry, _):
    ws, acc = carry
    keys, ws = jax.vmap(next_key)(ws)
    inputs = jax.vmap(lambda w, k: {**bot_inputs(w, k), **item_inputs(w, k)})(ws, keys)
    ws, stats = jax.vmap(harness_tick)(ws, inputs)
    return (ws, acc + stats), None


@functools.partial(jax.jit, static_argnums=2)
def run_chunk(worlds, acc, ticks):
    (worlds, acc), _ = jax.lax.scan(_tick_all, (worlds, acc), None, length=ticks)
    return worlds, acc


def add_items(worlds):
    b = worlds["tick"].shape[0]
    block = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (b,) + x.shape), ro_items.make_state())
    return {**worlds, "items": block}


def check_item_invariants(worlds):
    """Violation strings for the item block (empty means clean). Applies to every player slot."""
    w = jax.device_get(worlds)
    inv = np.asarray(w["players"]["inv"])
    cnt = np.asarray(w["items"]["inv_count"])
    ref = np.asarray(w["items"]["inv_refine"])
    cards = np.asarray(w["items"]["inv_cards"])
    eq = np.asarray(w["items"]["equip"])
    ryn = np.asarray(w["items"]["ryn"])
    kind, cls, nsock = np.asarray(I.KIND), np.asarray(I.CLASS), np.asarray(I.SLOTS)
    problems = []

    def flag(mask, text):
        bs, ps = np.nonzero(mask.reshape(mask.shape[:2] + (-1,)).any(axis=2))
        for b in np.unique(bs):
            problems.append(f"world {b}: {text} (e.g. player {int(ps[bs == b][0])})")

    flag((inv < 0) | (inv >= I.N_ITEMS), "inventory id out of range")
    inv_c = np.clip(inv, 0, I.N_ITEMS - 1)
    flag((cnt >= 1) != (inv != 0), "inv_count >= 1 differs from inv != 0")
    flag((cnt < 0) | (cnt > I.MAX_STACK), "inv_count out of range")
    flag((ref < 0) | (ref > I.MAX_REFINE), "refine out of 0..MAX_REFINE")
    flag((ref != 0) & (inv == 0), "refine on an empty slot")
    k_idx = np.arange(cards.shape[3])[None, None, None, :]
    card_c = np.clip(cards, 0, I.N_ITEMS - 1)
    bad_card = (cards != 0) & ((cards < 0) | (cards >= I.N_ITEMS) | (kind[card_c] != I.KIND_CARD)
                               | (k_idx >= nsock[inv_c][..., None]) | (inv[..., None] == 0))
    flag(bad_card, "card in a bad socket or not a card")

    flag(((eq < -1) | (eq >= INV_SLOTS)).any(axis=2), "equip out of -1..INV_SLOTS-1")
    for e in range(eq.shape[2]):
        s = np.clip(eq[:, :, e], 0, INV_SLOTS - 1)
        item = np.take_along_axis(inv, s[:, :, None], axis=2)[:, :, 0]
        used = eq[:, :, e] >= 0
        k = kind[np.clip(item, 0, I.N_ITEMS - 1)]
        c = cls[np.clip(item, 0, I.N_ITEMS - 1)]
        if e == I.EQ_WEAPON:
            good = k == I.KIND_WEAPON
        elif e <= I.EQ_SHOES:
            good = (k == I.KIND_ARMOR) & (c == e)
        else:
            good = (k == I.KIND_ARMOR) & (c == I.SLOT_ACCESSORY)
        flag(used & ((item == 0) | ~good), f"equip slot {e} points at a wrong item")
    for i in range(eq.shape[2]):
        for j in range(i + 1, eq.shape[2]):
            flag((eq[:, :, i] >= 0) & (eq[:, :, i] == eq[:, :, j]), f"inventory slot equipped twice ({i},{j})")
    flag((ryn < 0) | (ryn > I.MAX_RYN), "ryn out of range")
    return problems


def check_all(worlds):
    base = [p for p in base_check(worlds) if "inventory value out of range" not in p]
    return base + check_item_invariants(worlds)


def main(batch=4, chunks=5, ticks_per_chunk=100, seed=0):
    print("platform:", jax.devices()[0].platform, flush=True)
    worlds = add_items(make_world_batch(seed, batch))
    n_bots = int(np.asarray(worlds["players"]["active"]).sum())
    acc = jnp.zeros((batch, N_STATS), jnp.int32)
    violations = []
    t0 = time.perf_counter()
    for c in range(chunks):
        worlds, acc = run_chunk(worlds, acc, ticks_per_chunk)
        jax.block_until_ready(worlds)
        probs = check_all(worlds)
        violations += probs
        p = worlds["players"]
        print(
            f"chunk {c + 1:2d} (tick {(c + 1) * ticks_per_chunk:5d}): kills={int(p['kills'].sum())} "
            f"deaths={int(p['deaths'].sum())} alive_players={int((p['active'] & ~p['dead']).sum())} "
            f"violations={len(probs)}",
            flush=True,
        )
    dt = time.perf_counter() - t0
    ticks = chunks * ticks_per_chunk
    tot = np.asarray(acc).sum(axis=0)
    per = lambda x: x / n_bots / ticks * 1000.0  # noqa: E731
    p = jax.device_get(worlds["players"])
    it = jax.device_get(worlds["items"])
    act = np.asarray(p["active"])
    print(f"\nstats: {batch} worlds x {ticks} ticks, {n_bots} bots total; rates per bot per 1000 ticks")
    print("items picked = drops taken from the ground (one unit each), by kind of the item taken")
    print(f"  kills {per(int(p['kills'].sum())):.2f}  deaths {per(int(p['deaths'].sum())):.2f}")
    print("  picked: " + "  ".join(f"{n} {per(tot[k]):.2f}" for n, k in zip(KIND_NAMES, _KIND_IDS))
          + f"  (coins {per(tot[6]):.2f}, not counted in etc)")
    ryn = np.asarray(it["ryn"])[act]
    print(f"  ryn per bot at end: mean {ryn.mean():.1f} max {int(ryn.max())}")
    wearing = (np.asarray(it["equip"])[act] >= 0)
    print(f"  equips {per(tot[7]):.2f}; bots wearing a weapon at end {int(wearing[:, I.EQ_WEAPON].sum())}/{n_bots}, "
          f"any armor {int(wearing[:, 1:].any(axis=1).sum())}/{n_bots}")
    print(f"  refines: attempts {per(tot[8] + tot[9]):.3f} successes {per(tot[8]):.3f} breaks {per(tot[9]):.3f} "
          f"(raw {int(tot[8] + tot[9])}/{int(tot[8])}/{int(tot[9])})")
    print(f"  cards compounded {per(tot[10]):.3f} (raw {int(tot[10])})")
    print(f"total {ticks} ticks x {batch} worlds in {dt:.1f}s")
    if violations:
        print("VIOLATIONS:")
        for v in violations[:30]:
            print("  ", v)
        print(f"soak: {len(violations)} violations")
        return 1
    print("soak: clean")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=100, help="ticks per chunk")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.exit(main(batch=a.batch, chunks=a.chunks, ticks_per_chunk=a.ticks, seed=a.seed))
