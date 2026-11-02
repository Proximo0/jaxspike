"""Tests for game/ro_craft.py (M4 gathering and crafting phase)."""
import jax
import jax.numpy as jnp

from game import ro_craft as rc
from game.config import INV_SLOTS, MAP, MAX_PLAYERS
from game.data import craft as C
from game.state import empty_inputs, make_empty_world


def inv_of(*rows):
    inv = jnp.zeros((MAX_PLAYERS, INV_SLOTS), jnp.int32)
    for i, row in enumerate(rows):
        inv = inv.at[i, : len(row)].set(jnp.asarray(row, jnp.int32))
    return inv


def vec(i, item):
    return jnp.zeros((MAX_PLAYERS,), jnp.int32).at[i].set(item)


def make_world(players=(), nodes=(), seed=0):
    """Empty world with craft block; players = [(slot, x, y)], nodes = [(slot, kind, x, y)] (placed, active)."""
    w = make_empty_world(jax.random.PRNGKey(seed))
    cr = rc.make_state()
    pl = w["players"]
    for i, x, y in players:
        pl = {**pl, "pos": pl["pos"].at[i].set(jnp.array([x, y])), "active": pl["active"].at[i].set(True)}
    n = cr["nodes"]
    for s, k, x, y in nodes:
        n = {**n, "pos": n["pos"].at[s].set(jnp.array([x, y])), "kind": n["kind"].at[s].set(k),
             "placed": n["placed"].at[s].set(True), "active": n["active"].at[s].set(True)}
    return {**w, "players": pl, "craft": {**cr, "nodes": n}}


def with_inv(w, inv):
    return {**w, "players": {**w["players"], "inv": inv}}


def with_level(w, skill, lvl):
    cr = w["craft"]
    return {**w, "craft": {**cr, "level": cr["level"].at[:, skill].set(lvl)}}


def gin(*gatherers):
    g = jnp.zeros((MAX_PLAYERS,), jnp.bool_)
    for i in gatherers:
        g = g.at[i].set(True)
    return {**empty_inputs(), "gather": g, "craft": jnp.zeros((MAX_PLAYERS,), jnp.int32)}


def cin(i, recipe):
    return {**empty_inputs(), "gather": jnp.zeros((MAX_PLAYERS,), jnp.bool_), "craft": vec(i, recipe)}


_gather = jax.jit(rc.gather)
_craft = jax.jit(rc.craft)
_phase = jax.jit(rc.phase)


def gather_outcomes(w, inputs, n=24):
    """Run gather with n different rng seeds; return the resulting worlds."""
    return [_gather({**w, "rng": jax.random.key_data(jax.random.PRNGKey(s))}, inputs) for s in range(n)]


def never_gathers(w, inputs, who=0):
    return not any(int(o["craft"]["gathers"][who]) for o in gather_outcomes(w, inputs))


# ---- inventory helpers --------------------------------------------------------------------------
def test_add_item_first_empty_and_none_and_full():
    inv = inv_of([5, 0, 7], [1] * INV_SLOTS)
    out = rc.add_item(inv, vec(0, 9).at[1].set(9))
    assert out[0, :3].tolist() == [5, 9, 7]
    assert (out[1] == inv[1]).all()                      # full inventory unchanged
    assert (rc.add_item(inv, jnp.zeros((MAX_PLAYERS,), jnp.int32)) == inv).all()


def test_has_room_count_remove():
    inv = inv_of([4, 4, 0, 4], [1] * INV_SLOTS)
    assert rc.has_room(inv).tolist()[:2] == [True, False]
    assert rc.count_item(inv, vec(0, 4))[0] == 3
    assert rc.count_item(inv, jnp.zeros((MAX_PLAYERS,), jnp.int32))[0] == 0
    out = rc.remove_item(inv, vec(0, 4))
    assert out[0, :4].tolist() == [0, 4, 0, 4]            # lowest matching slot cleared
    assert (rc.remove_item(inv, vec(0, 77)) == inv).all()
    assert (rc.remove_item(inv, jnp.zeros((MAX_PLAYERS,), jnp.int32)) == inv).all()


def test_level_from_exp_boundaries():
    e = jnp.array([0, 49, 50, 199, 200, 10**9])
    assert rc.level_from_exp(e).tolist() == [1, 1, 2, 2, 3, C.CRAFT_MAX_LEVEL]
    assert rc.level_from_exp(jnp.zeros((3, 6), jnp.int32)).shape == (3, 6)


# ---- gather -----------------------------------------------------------------------------------------
def test_gather_success_and_failure_outcomes():
    w = make_world([(0, 10, 10)], [(0, C.NODE_COPPER, 11, 10)])
    w = with_inv(w, inv_of([3, 0, 8]))
    outs = gather_outcomes(w, gin(0))
    wins = [o for o in outs if int(o["craft"]["gathers"][0]) == 1]
    fails = [o for o in outs if int(o["craft"]["gathers"][0]) == 0]
    assert wins and fails                                 # chance 500/1000: both outcomes happen
    o = wins[0]
    assert o["players"]["inv"][0, :3].tolist() == [3, C.ITEM_COPPER_ORE, 8]   # first empty slot
    assert not bool(o["craft"]["nodes"]["active"][0])
    assert int(o["craft"]["nodes"]["respawn_in"][0]) == C.NODE_RESPAWN[C.NODE_COPPER]
    assert int(o["craft"]["exp"][0, C.SKILL_MINING]) == C.NODE_EXP[C.NODE_COPPER]
    f = fails[0]
    assert bool(f["craft"]["nodes"]["active"][0]) and (f["players"]["inv"] == w["players"]["inv"]).all()
    assert int(f["craft"]["exp"].sum()) == 0


def test_gather_own_tile_level_up_and_no_input():
    w = make_world([(0, 10, 10)], [(0, C.NODE_HERB, 10, 10)])
    w = {**w, "craft": {**w["craft"], "exp": w["craft"]["exp"].at[0, C.SKILL_HERBALISM].set(45)}}
    outs = gather_outcomes(w, gin(0))
    assert any(int(o["craft"]["level"][0, C.SKILL_HERBALISM]) == 2 for o in outs)   # 45 + 8 >= 50
    quiet = _gather(w, gin())
    assert bool(quiet["craft"]["nodes"]["active"][0]) and int(quiet["craft"]["gathers"].sum()) == 0


def test_gather_blocked_cases():
    base = make_world([(0, 10, 10)], [(0, C.NODE_IRON, 11, 10)])
    w = with_level(base, C.SKILL_MINING, 50)
    assert not never_gathers(w, gin(0))                                    # control: works
    assert never_gathers(base, gin(0))                                     # iron needs mining 10
    assert never_gathers(with_inv(w, inv_of([1] * INV_SLOTS)), gin(0))     # full inventory
    pl = w["players"]
    assert never_gathers({**w, "players": {**pl, "dead": pl["dead"].at[0].set(True)}}, gin(0))
    assert never_gathers({**w, "players": {**pl, "active": pl["active"].at[0].set(False)}}, gin(0))
    far = make_world([(0, 10, 10)], [(0, C.NODE_COPPER, 12, 10)])
    assert never_gathers(far, gin(0))                                      # out of range
    off = make_world([(0, 10, 10)], [(0, C.NODE_COPPER, 11, 10)])
    n = off["craft"]["nodes"]
    off = {**off, "craft": {**off["craft"], "nodes": {**n, "active": jnp.zeros_like(n["active"])}}}
    assert never_gathers(off, gin(0))                                      # inactive node
    assert never_gathers(make_world([(0, 10, 10)], [(0, C.NODE_FURNACE, 11, 10)]), gin(0))   # station


def test_gather_two_players_lowest_slot_wins():
    w = make_world([(3, 10, 10), (5, 12, 10)], [(0, C.NODE_COPPER, 11, 10)])
    outs = gather_outcomes(w, gin(3, 5))
    assert any(int(o["craft"]["gathers"][3]) for o in outs)
    for o in outs:
        assert int(o["craft"]["gathers"][5]) == 0
        assert int(o["craft"]["gathers"].sum()) <= 1
        assert int(o["craft"]["exp"][5].sum()) == 0 and (o["players"]["inv"][5] == 0).all()


def test_adjacent_node_picks_lowest_active_gatherable():
    w = make_world([(0, 10, 10)], [(7, C.NODE_TREE, 10, 11), (2, C.NODE_FISH, 9, 10), (1, C.NODE_FURNACE, 11, 10)])
    idx, kind, ok = rc.adjacent_node(w)
    assert (int(idx[0]), int(kind[0]), bool(ok[0])) == (2, C.NODE_FISH, True)
    assert (int(idx[1]), int(kind[1]), bool(ok[1])) == (C.MAX_NODES, 0, False)


# ---- respawn ----------------------------------------------------------------------------------------
def test_respawn_countdown_and_stations():
    w = make_world([], [(0, C.NODE_COPPER, 5, 5), (1, C.NODE_FURNACE, 8, 8)])
    n = w["craft"]["nodes"]
    n = {**n, "active": n["active"].at[0].set(False), "respawn_in": n["respawn_in"].at[0].set(3)}
    w = {**w, "craft": {**w["craft"], "nodes": n}}
    rs = jax.jit(rc.respawn)
    for expect_active, expect_r in [(False, 2), (False, 1), (True, 0), (True, 0)]:
        w = rs(w)
        assert bool(w["craft"]["nodes"]["active"][0]) is expect_active
        assert int(w["craft"]["nodes"]["respawn_in"][0]) == expect_r
        assert bool(w["craft"]["nodes"]["active"][1])                     # station never deactivates
    assert int(w["craft"]["nodes"]["active"].sum()) == 2                  # unplaced slots stay inactive


# ---- craft ------------------------------------------------------------------------------------------
def furnace_world(inv_row):
    w = make_world([(0, 10, 10)], [(0, C.NODE_FURNACE, 11, 10)])
    return with_inv(w, inv_of(inv_row))


def crafts(w, recipe):
    return int(_craft(w, cin(0, recipe))["craft"]["crafts"][0])


def test_craft_consumes_exact_inputs_with_multiplicity():
    w = furnace_world([C.ITEM_COPPER_ORE, 2, C.ITEM_COPPER_ORE, 3])
    o = _craft(w, cin(0, C.RECIPE_COPPER_BAR))
    assert o["players"]["inv"][0, :4].tolist() == [C.ITEM_COPPER_BAR, 2, 0, 3]   # output in first empty slot
    assert int(o["craft"]["crafts"][0]) == 1
    assert int(o["craft"]["exp"][0, C.SKILL_SMITHING]) == int(C.RECIPE_EXP[C.RECIPE_COPPER_BAR])


def test_craft_blocked_cases():
    assert crafts(furnace_world([C.ITEM_COPPER_ORE]), C.RECIPE_COPPER_BAR) == 0      # needs two ore
    two = furnace_world([C.ITEM_COPPER_ORE, C.ITEM_COPPER_ORE])
    assert crafts(two, C.RECIPE_COPPER_BAR) == 1
    assert crafts(two, C.RECIPE_NONE) == 0
    assert crafts(two, C.RECIPE_IRON_BAR) == 0                                       # wrong inputs
    iron = furnace_world([C.ITEM_IRON_ORE, C.ITEM_COAL])
    assert crafts(iron, C.RECIPE_IRON_BAR) == 0                                      # smithing 10 needed
    assert crafts(with_level(iron, C.SKILL_SMITHING, 10), C.RECIPE_IRON_BAR) == 1
    pl = two["players"]
    far = {**two, "players": {**pl, "pos": pl["pos"].at[0].set(jnp.array([20, 20]))}}
    assert crafts(far, C.RECIPE_COPPER_BAR) == 0                                     # no station nearby
    fire = with_inv(make_world([(0, 10, 10)], [(0, C.NODE_FIRE, 11, 10)]), inv_of([C.ITEM_COPPER_ORE] * 2))
    assert crafts(fire, C.RECIPE_COPPER_BAR) == 0                                    # wrong station
    assert crafts({**two, "players": {**pl, "dead": pl["dead"].at[0].set(True)}}, C.RECIPE_COPPER_BAR) == 0
    assert crafts(two, 99) == 0                                                      # clipped to a real id, no inputs


def test_craft_station_only_when_needed():
    w = with_inv(make_world([(0, 10, 10)], []), inv_of([C.ITEM_HERB, C.ITEM_HERB]))
    o = _craft(w, cin(0, C.RECIPE_HERB_POTION))
    assert o["players"]["inv"][0, :2].tolist() == [C.ITEM_HERB_POTION, 0]
    cc = rc.can_craft(w)
    assert bool(cc[0, C.RECIPE_HERB_POTION]) and not bool(cc[0, C.RECIPE_COPPER_BAR])
    assert not bool(cc[:, 0].any())
    assert bool(rc.has_inputs(w)[0, C.RECIPE_HERB_POTION])


def test_has_inputs_ignores_station():
    w = with_inv(make_world([(0, 10, 10)], []), inv_of([C.ITEM_COPPER_ORE, C.ITEM_COPPER_ORE]))
    assert bool(rc.has_inputs(w)[0, C.RECIPE_COPPER_BAR]) and not bool(rc.can_craft(w)[0, C.RECIPE_COPPER_BAR])
    assert bool(rc.adjacent_station(w, jnp.full((MAX_PLAYERS,), -1, jnp.int32)).all())


# ---- phase, jit, vmap ---------------------------------------------------------------------------------
def test_phase_gather_and_craft_same_tick_and_vmap():
    w = make_world([(0, 10, 10)], [(0, C.NODE_FURNACE, 10, 11), (1, C.NODE_COPPER, 9, 10)])
    w = with_inv(w, inv_of([C.ITEM_COPPER_ORE, C.ITEM_COPPER_ORE]))
    inputs = {**gin(0), "craft": vec(0, C.RECIPE_COPPER_BAR)}
    outs = [_phase({**w, "rng": jax.random.key_data(jax.random.PRNGKey(s))}, inputs) for s in range(16)]
    assert all(int(o["craft"]["crafts"][0]) == 1 for o in outs)
    both = [o for o in outs if int(o["craft"]["gathers"][0]) == 1]
    assert both
    assert sorted(both[0]["players"]["inv"][0, :3].tolist()) == sorted([0, C.ITEM_COPPER_ORE, C.ITEM_COPPER_BAR])
    bw = jax.tree.map(lambda a: jnp.stack([a, a]), w)
    bi = jax.tree.map(lambda a: jnp.stack([a, a]), inputs)
    vo = jax.vmap(rc.phase)(bw, bi)
    assert vo["craft"]["crafts"].shape == (2, MAX_PLAYERS)
    assert vo["craft"]["crafts"][:, 0].tolist() == [1, 1]


def test_phase_on_real_terrain_world_without_inputs_is_noop():
    from game import map_gen

    k_world, k_walls, k_nodes = jax.random.split(jax.random.PRNGKey(0), 3)
    terrain = map_gen.make_terrain(k_walls)
    w = {**make_empty_world(k_world), "terrain": terrain, "walls": map_gen.blocking(terrain),
         "pz": map_gen.protection_zone(terrain)}
    w = rc.with_craft(w, k_nodes)
    nodes = w["craft"]["nodes"]
    assert int(nodes["placed"].sum()) > 0 and w["craft"]["dist_node"].shape == (C.N_NODE_KINDS, MAP, MAP)
    assert w["craft"]["dist_node"].dtype == jnp.int16
    inputs = {**empty_inputs(), "gather": jnp.zeros((MAX_PLAYERS,), jnp.bool_),
              "craft": jnp.zeros((MAX_PLAYERS,), jnp.int32)}
    out = _phase(w, inputs)
    assert jax.tree.structure(out) == jax.tree.structure(w)
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(w)):
        assert a.shape == b.shape and a.dtype == b.dtype
    assert all((a == b).all() for a, b in zip(jax.tree.leaves(out["players"]), jax.tree.leaves(w["players"])))
    assert (out["craft"]["nodes"]["active"] == nodes["active"]).all()


def test_reexports():
    for name in ("place_nodes", "node_fields", "node_at", "nodes_near", "station_tiles"):
        assert callable(getattr(rc, name))
