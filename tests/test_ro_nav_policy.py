"""Tests for bots/ro_nav_policy.py: dungeon entrance fields and the retreat / rest / dive / hunt-inside modes."""
import jax
import jax.numpy as jnp
import numpy as np

from bots import ro_nav_policy as nav
from game import ro_stats
from game.config import MAP, MAX_PLAYERS
from game.data.monsters import HP as MON_HP, KIND_BY_NAME
from game.map_gen import DIST_UNREACHABLE
from game.state import make_empty_world
from game.world_layout import dungeon_boxes, dungeon_pads, portal_warp_tiles

KEY = jax.random.PRNGKey(3)
ENT = np.asarray(dungeon_pads())[:, 0]  # [ND, 2] entrance warp tiles
S, SW, W, NW, N, NE, E, SE = range(1, 9)


def _cheb(x, y):
    xs, ys = jnp.meshgrid(jnp.arange(MAP), jnp.arange(MAP), indexing="ij")
    return jnp.maximum(jnp.abs(xs - x), jnp.abs(ys - y))


def world(pos, level=1, hp_pct=100, slot=0, field=True, pz=(), monsters=(), kinds=None):
    """One active player at `pos` with the given base level and HP percent of its derived max, with the stats
    bots.ro_stats_policy allots a fresh Novice's 48 points to (STR 18, VIT 4)."""
    w = make_empty_world(KEY)
    st = {**w["stats"], "base_level": w["stats"]["base_level"].at[slot].set(level),
          "str": w["stats"]["str"].at[slot].set(18), "vit": w["stats"]["vit"].at[slot].set(4)}
    w = {**w, "stats": st}
    p = w["players"]
    p = {**p, "pos": p["pos"].at[slot].set(jnp.array(pos, jnp.int32)), "active": p["active"].at[slot].set(True)}
    w = {**w, "players": p}
    mx = int(ro_stats.derived(w)["max_hp"][slot])
    w["players"] = {**p, "hp": p["hp"].at[slot].set(mx * hp_pct // 100)}
    for x, y in pz:
        w["pz"] = w["pz"].at[x, y].set(True)
    if monsters:
        m = w["monsters"]
        for i, (x, y) in enumerate(monsters):
            k = 0 if kinds is None else kinds[i]
            m = {**m, "pos": m["pos"].at[i].set(jnp.array([x, y])), "active": m["active"].at[i].set(True),
                 "kind": m["kind"].at[i].set(k), "hp": m["hp"].at[i].set(int(MON_HP[k]))}  # full hp: not provoked
        w["monsters"] = m
    if field:
        w["dist_dungeon"] = jnp.stack([_cheb(int(x), int(y)) for x, y in ENT]).astype(jnp.int16)
    # home-gate field: distance to the map centre's west edge stands in for dist_town (all towns share it)
    gate = _cheb(10, MAP // 2).astype(jnp.int16)
    w["dist_town"] = jnp.broadcast_to(gate, w["dist_town"].shape)
    return w


_nav = jax.jit(nav.nav_moves)


def run(w, base=0):
    move, sit, _, _ = _nav(w, KEY, jnp.full((MAX_PLAYERS,), base, jnp.int32))
    return move, sit


def test_dungeon_fields_on_open_ground_are_chebyshev_and_respect_walls():
    w = make_empty_world(KEY)
    f = np.asarray(jax.jit(nav.dungeon_fields)(w["terrain"]))
    x, y = ENT[0]
    assert f.shape[0] == len(ENT) and f[0, x, y] == 0
    assert f[0, x + 3, y + 5] == 5 and f[0, x - 7, y + 2] == 7
    # a wall column with one gap: the far side's distance goes around through the gap, no corner cutting
    t = w["terrain"]
    from game.config import T_WALL
    t = t.at[x + 2, y - 20:y + 20].set(T_WALL).at[x + 2, y + 20].set(t[0, 0])
    f = np.asarray(jax.jit(nav.dungeon_fields)(t))
    assert f[0, x + 2, y] == DIST_UNREACHABLE
    assert f[0, x + 3, y] > 20


ROAM_HOME = SW  # beyond 2 * HUNT_RADIUS north-east of the test gate (10, MAP / 2): SW, W, NW tie, lowest code


def test_low_level_bot_roams_instead_of_diving():
    x, y = ENT[0]
    move, _ = run(world((x + 40, y + 40), level=nav.DUNGEON_MIN_LEVEL - 1), base=W)
    assert int(move[0]) == ROAM_HOME


def test_high_level_healthy_bot_walks_down_the_dungeon_field():
    x, y = ENT[0]
    move, _ = run(world((x + 40, y + 40), level=nav.DUNGEON_MIN_LEVEL), base=E)
    assert int(move[0]) == NW
    move, _ = run(world((x, y + 30), level=nav.DUNGEON_MIN_LEVEL), base=E)
    assert int(move[0]) in (NW, N, NE)  # all three lower the field to 29; ties go to the lowest code


def test_bot_on_the_entrance_warp_stands_for_the_warp_phase():
    x, y = ENT[0]
    move, _ = run(world((x, y), level=20), base=E)
    assert int(move[0]) == 0


def test_no_dungeon_field_means_no_dive():
    x, y = ENT[0]
    move, _ = run(world((x + 40, y + 40), level=30, field=False), base=E)
    assert int(move[0]) == ROAM_HOME


def test_gatherer_slots_never_dive():
    x, y = ENT[0]
    move, _ = run(world((x + 40, y + 40), level=30, slot=1), base=E)
    assert int(move[1]) != NW  # never down the dungeon field; the base move is ignored (it roams)


def test_bot_below_dive_pct_does_not_dive():
    x, y = ENT[0]
    move, _ = run(world((x + 40, y + 40), level=30, hp_pct=nav.DIVE_PCT - 5), base=E)
    assert int(move[0]) == ROAM_HOME


def test_hurt_bot_walks_home_even_at_high_level():
    """Hurt with an aggressive monster close by: no field rest, walk home."""
    x, y = ENT[0]
    bug = KIND_BY_NAME["Pilferbug"]
    move, sit = run(world((x + 40, y + 40), level=30, hp_pct=nav.RETREAT_PCT - 5, monsters=[(x + 42, y + 40)],
                          kinds=[bug]), base=E)
    assert int(move[0]) in (W, NW, SW) and not bool(sit[0])  # down the home field toward x = 10


def test_hurt_bot_sits_in_a_safe_field_spot_and_stands_when_rested():
    from game import ro_anim
    w = world((100, 100), hp_pct=nav.RETREAT_PCT - 5, monsters=[(101, 100)])  # a passive Blobbin at full hp
    move, sit, attack, _ = _nav(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(move[0]) == 0 and bool(sit[0]) and not bool(attack[0])  # sits down (the input is a toggle)
    a = w["anim"]
    w["anim"] = {**a, "action": a["action"].at[0].set(ro_anim.SIT)}
    move, sit = run(w)
    assert int(move[0]) == 0 and not bool(sit[0])  # already sitting: no toggle, stays down
    p = w["players"]
    mx = int(ro_stats.derived(w)["max_hp"][0])
    w["players"] = {**p, "hp": p["hp"].at[0].set(mx * (nav.RETREAT_PCT + 10) // 100)}
    move, sit = run(w)
    assert int(move[0]) == 0 and not bool(sit[0])  # above RETREAT_PCT but below REST_PCT: keeps sitting
    w["players"] = {**p, "hp": p["hp"].at[0].set(mx)}
    move, sit = run(w)
    assert bool(sit[0])  # rested: stands up (toggle) and hunts


def test_prey_choice_by_hit_and_cost():
    """A fresh Novice chases a Blobbin but not a Fiddlehop (VIT 18 eats its 19 ATK) beside it; an adjacent passive
    Fiddlehop is left alone, a provoked one is fought back."""
    hop = KIND_BY_NAME["Fiddlehop"]
    lv = 7  # within LEVEL_MARGIN of the level-9 Fiddlehop: only the cost test keeps the bot off it
    w = world((100, 100), level=lv, monsters=[(104, 100)], kinds=[hop])
    _, _, attack, fight = _nav(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert not bool(fight[0]) and not bool(attack[0])
    hit, cost = nav.prey_cost(w, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(cost[0]) > int(ro_stats.derived(w)["max_hp"][0])  # cannot be won from full hp
    w = world((100, 100), monsters=[(104, 100)])
    mx = int(ro_stats.derived(w)["max_hp"][0])
    hit, cost = nav.prey_cost(w, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(hit[0]) == 80 and 0 < int(cost[0]) * nav.COST_SAFETY_PCT < (100 - nav.FLOOR_PCT) * mx
    _, _, _, fight = _nav(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert bool(fight[0])
    w = world((100, 100), level=lv, monsters=[(101, 100)], kinds=[hop])
    _, _, attack, _ = _nav(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert not bool(attack[0])
    m = w["monsters"]
    w["monsters"] = {**m, "hp": m["hp"].at[0].add(-10)}
    _, _, attack, _ = _nav(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert bool(attack[0])


def test_resting_bot_on_pz_stands_and_sits():
    move, sit = run(world((100, 100), hp_pct=50, pz=[(100, 100)]), base=E)
    assert int(move[0]) == 0 and bool(sit[0])
    move, sit = run(world((100, 100), hp_pct=100, pz=[(100, 100)]), base=E)
    assert int(move[0]) == S and not bool(sit[0])  # rested: leaves town down the gate field


def test_bot_inside_a_dungeon_hunts_instead_of_walking_out():
    x0, y0, x1, y1 = dungeon_boxes()[0]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    move, _ = run(world((cx, cy), level=20, monsters=[(cx + 3, cy)]), base=W)
    assert int(move[0]) == E  # chases the monster, not the base "walk back to the gate"
    move, _ = run(world((cx, cy), level=20, monsters=[(cx + 1, cy)]), base=W)
    assert int(move[0]) == 0  # adjacent: stand and fight
    move, _ = run(world((cx, cy), level=20), base=W)
    assert 1 <= int(move[0]) <= 8  # nothing near: keeps a wander heading


def test_dead_and_inactive_slots_do_not_move():
    w = world((100, 100), level=20)
    w["players"] = {**w["players"], "dead": w["players"]["dead"].at[0].set(True)}
    move, sit = run(w, base=E)
    assert int(move[0]) == 0 and not bool(sit[0])
    assert int(move[5]) == 0


def test_jit_vmap_shapes():
    x, y = ENT[0]
    w = world((x + 40, y + 40), level=20)
    ws = jax.tree.map(lambda a: jnp.stack([a, a]), w)
    move, sit, _, _ = jax.jit(jax.vmap(nav.nav_moves, in_axes=(0, 0, 0)))(
        ws, jax.random.split(KEY, 2), jnp.zeros((2, MAX_PLAYERS), jnp.int32))
    assert move.shape == (2, MAX_PLAYERS) and move.dtype == jnp.int32 and sit.dtype == jnp.bool_
    assert int(move[0, 0]) == int(move[1, 0]) == NW


def test_chase_steps_around_a_wall_and_attacks_when_adjacent():
    w = world((100, 100), monsters=[(104, 100)])
    w["walls"] = w["walls"].at[101, 99:102].set(1)  # a wall east (diagonals would cut its corners)
    move, _, attack, _ = _nav(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(move[0]) in (S, SW, W, NW, N) and not bool(attack[0])  # boxed in: a random free step, never 0
    w = world((100, 100), monsters=[(104, 100)])
    w["walls"] = w["walls"].at[101, 100].set(1).at[100, 99].set(1)  # wall east and north: only south goes round
    move, _, _, _ = _nav(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(move[0]) in (S, SW, W, NW, N, NE, E, SE) and int(move[0]) not in (E, NE, SE, N)
    move, _, attack, _ = _nav(world((100, 100), monsters=[(101, 100)]), KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(move[0]) == 0 and bool(attack[0])


def test_chase_never_steps_onto_a_living_player():
    w = world((100, 100), monsters=[(103, 100)])
    p = w["players"]
    w["players"] = {**p, "pos": p["pos"].at[2].set(jnp.array([101, 100])), "active": p["active"].at[2].set(True)}
    move, _, _, _ = _nav(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(move[0]) in (NE, SE)


def test_hurt_bot_does_not_attack():
    move, _, attack, _ = _nav(world((100, 100), hp_pct=nav.RETREAT_PCT - 10, monsters=[(101, 100)]), KEY,
                           jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert not bool(attack[0])


def test_monsters_far_above_the_bots_level_are_not_chased():
    from game.data.monsters import LEVEL
    strong = int(np.argmax(np.asarray(LEVEL)))
    w = world((100, 100), level=1, monsters=[(104, 100)])
    w["monsters"] = {**w["monsters"], "kind": w["monsters"]["kind"].at[0].set(strong)}
    move, _, _, _ = _nav(w, KEY, jnp.full((MAX_PLAYERS,), W, jnp.int32))
    assert int(move[0]) == S  # roams down the gate field instead of chasing east


def test_bot_that_wants_potions_walks_to_its_shop_row():
    goal = np.asarray(nav.SHOP_OF_TOWN[0])
    start = (int(goal[0]) - 6, int(goal[1]))
    w = world(start, pz=[start])
    shop = jnp.zeros((MAX_PLAYERS,), jnp.bool_).at[0].set(True)
    move, _, _, _ = jax.jit(nav.nav_moves)(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32), shop)
    assert int(move[0]) in (NE, E, SE)
    w = world((int(goal[0]) - 1, int(goal[1])), pz=[(int(goal[0]) - 1, int(goal[1]))])
    move, _, _, _ = jax.jit(nav.nav_moves)(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32), shop)
    assert int(move[0]) == 0  # next to the shop: stop


def test_roam_walks_away_from_the_gate_inside_half_the_hunt_radius():
    from bots.policies import HUNT_RADIUS
    move, _ = run(world((10 + HUNT_RADIUS // 4, MAP // 2)), base=0)
    assert int(move[0]) in (NE, E, SE)  # away from the gate at (10, MAP / 2), never back toward it


def test_dungeon_fields_relax_across_the_zone_portals():
    """A wall along the whole x = 192 rock line cuts the map; the Culvert stays reachable through the portal."""
    from game.config import T_WALL
    w = make_empty_world(KEY)
    src, dst = (np.asarray(t) for t in portal_warp_tiles())
    line = 192 if MAP == 512 else int(0.375 * MAP)
    legs = [(tuple(s), tuple(d)) for s, d in zip(src.tolist(), dst.tolist()) if s[0] == line - 1]
    assert legs, "no portal leg crosses the first vertical zone line"
    t = w["terrain"].at[line, :].set(T_WALL)
    f = np.asarray(jax.jit(nav.dungeon_fields)(t))
    x, y = ENT[0]
    assert x > line
    assert f[0, line, y] == DIST_UNREACHABLE
    far = f[0, line - 40, y]
    assert far < DIST_UNREACHABLE
    (sx, sy), (dx, dy) = legs[0]
    assert f[0, sx, sy] == f[0, dx, dy] + 1  # the warp source is one step from its destination


def test_dive_picks_the_nearest_dungeon_the_bot_is_high_enough_for():
    x1, y1 = ENT[1]
    assert nav.DUNGEON_DIVE_LEVEL[1] > nav.DUNGEON_DIVE_LEVEL[0] == nav.DUNGEON_MIN_LEVEL
    move, _ = run(world((x1 - 40, y1 + 40), level=nav.DUNGEON_DIVE_LEVEL[1]), base=W)
    assert int(move[0]) == NE  # Pinehollow Cave is next door and allowed
    move, _ = run(world((x1 - 40, y1 + 40), level=nav.DUNGEON_DIVE_LEVEL[0]), base=W)
    assert int(move[0]) != NE  # too low for Pinehollow Cave: walks toward the Culvert instead


def test_ring_by_level_sends_higher_bots_further_out():
    from bots.policies import HUNT_RADIUS
    pos = (10 + 3 * HUNT_RADIUS // 4, MAP // 2)  # past band 0's lo, inside band 1's lo
    move, _ = run(world(pos, level=nav.BAND1_LEVEL), base=0)
    assert int(move[0]) in (NE, E, SE)  # walks away from the gate
    assert nav.RING_BY_LEVEL[1][0] > 3 * HUNT_RADIUS // 4 > nav.RING_BY_LEVEL[0][0]
    # ro5-balance: from BAND1_OUTER_LEVEL the ring reaches band 1's outer half; below it the same spot is past ring_hi
    assert nav.RING_BY_LEVEL[2][0] > nav.RING_BY_LEVEL[1][0] and nav.RING_BY_LEVEL[2][1] > nav.RING_BY_LEVEL[1][1]
    lo2, hi1 = nav.RING_BY_LEVEL[2][0], nav.RING_BY_LEVEL[1][1]
    move, _ = run(world((10 + lo2 - 3, MAP // 2), level=nav.BAND1_OUTER_LEVEL), base=0)
    assert int(move[0]) in (NE, E, SE)  # short of the outer ring: still walking out
    move, _ = run(world((10 + hi1 + 3, MAP // 2), level=nav.BAND1_OUTER_LEVEL - 1), base=0)
    assert int(move[0]) in (NW, W, SW)  # beyond the inner band-1 ring: turns back toward the gate


def test_walking_away_prefers_a_zone_portal_tile():
    p = np.asarray(portal_warp_tiles()[0])[0]
    px, py = int(p[0]), int(p[1])
    w = world((px - 1, py))
    w["dist_town"] = jnp.broadcast_to(_cheb(px - 10, py).astype(jnp.int16), w["dist_town"].shape)
    move, _, _, _ = _nav(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(move[0]) == E  # E, NE and SE all raise the field by one; the portal bonus breaks the tie


def test_roaming_never_steps_onto_a_dungeon_entrance():
    x, y = ENT[0]
    w = world((x - 1, y), level=1)
    w["dist_town"] = jnp.broadcast_to(_cheb(x - 10, y).astype(jnp.int16), w["dist_town"].shape)
    move, _, _, _ = _nav(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(move[0]) in (NE, SE)  # away from the gate, but not east onto the warp


def test_party_member_follows_its_leader_in_the_same_zone():
    def with_party(w, leader, member):
        pt = w["party"]
        pid = pt["party_id"].at[leader].set(0).at[member].set(0)
        return {**w, "party": {**pt, "party_id": pid, "leader": pt["leader"].at[leader].set(True),
                               "member_count": pt["member_count"].at[0].set(2)}}
    w = world((100, 100), slot=5)
    p = w["players"]
    w["players"] = {**p, "pos": p["pos"].at[0].set(jnp.array([120, 100])), "active": p["active"].at[0].set(True)}
    move, _, _, _ = _nav(with_party(w, 0, 5), KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(move[5]) == E
    w["players"] = {**w["players"], "pos": w["players"]["pos"].at[0].set(jnp.array([300, 100]))}  # other zone
    move, _, _, _ = _nav(with_party(w, 0, 5), KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(move[5]) != E or int(move[5]) == int(run(w, base=0)[0][5])
    w["players"] = {**w["players"], "pos": w["players"]["pos"].at[0].set(jnp.array([103, 100]))}  # close enough
    move, _, _, _ = _nav(with_party(w, 0, 5), KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(move[5]) == int(run(w, base=0)[0][5])


def test_bot_left_on_a_warp_tile_waits_for_the_warp_phase():
    from game.data.terrain import T_WARP
    w = world((100, 100))
    w["terrain"] = w["terrain"].at[100, 100].set(T_WARP)
    w["tick"] = jnp.int32(1)
    move, _, _, _ = _nav(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(move[0]) == 0
    w["tick"] = jnp.int32(nav.WARP_WAIT_TICKS)
    move, _, _, _ = _nav(w, KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert int(move[0]) != 0


def test_fight_mask_covers_chasing_and_adjacent_bots():
    _, _, attack, fight = _nav(world((100, 100), monsters=[(101, 100)]), KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert bool(attack[0]) and bool(fight[0])
    _, _, attack, fight = _nav(world((100, 100)), KEY, jnp.zeros((MAX_PLAYERS,), jnp.int32))
    assert not bool(attack[0]) and not bool(fight[0])
