"""M5 soak: bot-played worlds with the party, MVP and kafra phases attached in a test harness.

The lead-owned step.py and combat.py are not edited here. This harness wraps `game.step.step`:
it replays combat's player targeting to feed `ro_mvp.record_damage` (the lead wires that call into
combat.py at merge), then runs the three M5 phases in the contract's order. Bots use the hunting policy
plus `bots.ro_party_policy.party_inputs`. Reports party sizes, MVP kills per 1000 ticks and the standard
invariants (MVP slots are excluded from the per-kind hp check, their kind is outside config's tables).
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from bench.soak_check import check_invariants  # noqa: E402
from bots.policies import bot_inputs  # noqa: E402
from bots.ro_party_policy import party_inputs  # noqa: E402
from bots.runner import make_world_batch  # noqa: E402
from game import ro_kafra, ro_mvp, ro_party  # noqa: E402
from game.config import MAX_PLAYERS, PLAYER_ATTACK  # noqa: E402
from game.spatial import build_grid, nearest_in_block  # noqa: E402
from game.state import next_key  # noqa: E402
from game.step import step  # noqa: E402


def attach(worlds):
    """Add the M5 blocks to a batch of worlds and place the MVPs."""
    b = worlds["tick"].shape[0]
    blocks = {"party": ro_party.make_state(), "mvp": ro_mvp.make_state()}
    blocks = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (b,) + a.shape), blocks)
    return jax.vmap(ro_mvp.place_mvps)({**worlds, **blocks})


def _player_damage(world, inputs):
    """Replay combat's player targeting: (attacker slots, monster slots, damage) for this tick's hits."""
    p, m = world["players"], world["monsters"]
    alive = p["active"] & ~p["dead"]
    table, _ = build_grid(m["pos"], m["active"])
    tgt, dist = nearest_in_block(table, p["pos"], m["pos"])
    hit = alive & inputs["attack"] & (p["attack_cd"] == 0) & (dist <= 1) & m["active"][tgt]
    pidx = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
    return pidx, jnp.where(hit, tgt, -1).astype(jnp.int32), jnp.where(hit, PLAYER_ATTACK, 0).astype(jnp.int32)


def m5_step(world, inputs):
    a, mi, d = _player_damage(world, inputs)
    world = ro_mvp.record_damage(world, a, mi, d)
    world = step(world, inputs)
    world = ro_party.phase(world, inputs)
    world = ro_mvp.phase(world)
    return ro_kafra.phase(world, inputs)


def run(worlds, ticks):
    def tick(ws, _):
        keys, ws = jax.vmap(next_key)(ws)
        k1, k2 = jax.vmap(lambda k: tuple(jax.random.split(k)))(keys)
        inputs = jax.vmap(bot_inputs)(ws, k1)
        inputs = {**inputs, **jax.vmap(party_inputs)(ws, k2)}
        return jax.vmap(m5_step)(ws, inputs), None

    final, _ = jax.lax.scan(tick, worlds, None, length=ticks)
    return final


def m5_invariants(worlds):
    """Standard invariants with MVP slots hidden, plus M5 checks."""
    w = jax.device_get(worlds)
    is_mvp = np.asarray(jax.vmap(ro_mvp.is_mvp)(worlds))
    m = w["monsters"]
    masked = {**w, "monsters": {**m, "active": np.asarray(m["active"]) & ~is_mvp,
                                "kind": np.where(is_mvp, 0, np.asarray(m["kind"]))}}
    problems = check_invariants(masked)
    party = w["party"]
    for b in range(is_mvp.shape[0]):
        pid = np.asarray(party["party_id"][b])
        act = np.asarray(w["players"]["active"][b])
        counts = np.bincount(pid[pid >= 0], minlength=ro_party.MAX_PARTIES)
        if (counts > ro_party.PARTY_MAX_MEMBERS).any():
            problems.append(f"world {b}: party over {ro_party.PARTY_MAX_MEMBERS} members")
        if not np.array_equal(counts, np.asarray(party["member_count"][b])):
            problems.append(f"world {b}: member_count out of sync")
        if (pid[~act] >= 0).any():
            problems.append(f"world {b}: inactive player in a party")
        leaders = np.asarray(party["leader"][b]) & (pid >= 0)
        if not np.array_equal(np.bincount(pid[leaders], minlength=ro_party.MAX_PARTIES) > 0, counts > 0):
            problems.append(f"world {b}: party without exactly one leader")
        mvp = w["mvp"]
        live_mvp = is_mvp[b] & np.asarray(m["active"][b])
        if live_mvp.any():
            mpos = np.asarray(m["pos"][b])[live_mvp]
            walls = np.asarray(w["walls"][b])
            if ((mpos < 0) | (mpos >= walls.shape[0])).any():
                problems.append(f"world {b}: mvp out of bounds")
            elif (walls[mpos[:, 0], mpos[:, 1]] != 0).any():
                problems.append(f"world {b}: active mvp on wall")
            others = np.asarray(m["pos"][b])[np.asarray(m["active"][b]) & ~is_mvp[b]]
            pp = np.asarray(w["players"]["pos"][b])[act & ~np.asarray(w["players"]["dead"][b])]
            taken = {tuple(t) for t in others} | {tuple(t) for t in pp}
            if any(tuple(t) in taken for t in mpos):
                problems.append(f"world {b}: mvp shares a tile")
            hp = np.asarray(m["hp"][b])[live_mvp]
            if ((hp <= 0) | (hp > max(ro_mvp.MVP_HP))).any():
                problems.append(f"world {b}: mvp hp out of range")
        if (np.asarray(mvp["next_spawn"][b]) < 0).any():
            problems.append(f"world {b}: mvp next_spawn negative")
    return problems


def main(batch=4, chunks=5, ticks=100, seed=0):
    print("platform:", jax.devices()[0].platform, flush=True)
    worlds = attach(make_world_batch(seed, batch))
    total = []
    events_seen = 0
    t0 = time.perf_counter()
    for c in range(chunks):
        worlds = run(worlds, ticks)
        jax.block_until_ready(worlds)
        probs = m5_invariants(worlds)
        total += probs
        pid = np.asarray(worlds["party"]["party_id"])
        sizes = np.asarray(worlds["party"]["member_count"])
        in_party = int((pid >= 0).sum())
        n_parties = int((sizes > 0).sum())
        mean_size = float(sizes[sizes > 0].mean()) if n_parties else 0.0
        ev = np.asarray(worlds["mvp"]["event"])
        events_seen = int((ev[..., 0] >= 0).sum())
        heads = int(np.asarray(worlds["mvp"]["event_head"]).sum())
        kills = int(worlds["players"]["kills"].sum())
        slots = np.asarray(worlds["mvp"]["mvp_slot"])[0]  # same slots in every world
        live_slots = slots[slots >= 0]
        mvp_hp = np.asarray(worlds["monsters"]["hp"])[:, live_slots].sum() if live_slots.size else 0
        mvp_dmg = np.asarray(worlds["mvp"]["damage_ledger"])[:, live_slots].sum() if live_slots.size else 0
        print(f"chunk {c + 1:2d} (tick {(c + 1) * ticks:5d}): kills={kills} players_in_party={in_party} "
              f"parties={n_parties} mean_size={mean_size:.1f} max_size={int(sizes.max())} "
              f"mvp_kills_total={heads} mvp_hp_sum={int(mvp_hp)} mvp_dmg_taken={int(mvp_dmg)} "
              f"invariant_violations={len(probs)}", flush=True)
    dt = time.perf_counter() - t0
    tt = chunks * ticks
    print(f"total {tt} ticks x {batch} worlds in {dt:.1f}s; mvp kills per 1000 ticks per world: "
          f"{heads / batch / tt * 1000:.2f} (events in ring {events_seen})")
    if total:
        print("FIRST VIOLATIONS:")
        for p_ in total[:15]:
            print("  ", p_)
    else:
        print("all invariants held")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.batch, a.chunks, a.ticks, a.seed)
