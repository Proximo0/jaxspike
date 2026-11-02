"""Tests for bots/ro_craft_policy.py (gatherer bot policy)."""
import jax
import jax.numpy as jnp

from bots import policies
from bots.ro_craft_policy import GATHER_EVERY, craft_bot_inputs
from game import ro_craft as rc
from game.config import INV_SLOTS, MAP, MAX_PLAYERS
from game.data import craft as C
from game.map_gen import DIST_UNREACHABLE
from game.state import make_empty_world

C0 = MAP // 2  # working area around the map centre
G = 1          # slot 1 is a gatherer of kind 0 (copper)
KEY = jax.random.PRNGKey(7)
_policy = jax.jit(craft_bot_inputs)


def manhattan(nx, ny):
    xs, ys = jnp.meshgrid(jnp.arange(MAP), jnp.arange(MAP), indexing="ij")
    return (jnp.abs(xs - nx) + jnp.abs(ys - ny)).astype(jnp.int16)


def make_world(player, nodes=(), inv=(), fields=(), pz=(), inactive=()):
    """Empty world: slot G active at `player`; nodes [(slot, kind, x, y)]; fields {kind: (x, y)} manhattan fields."""
    w = make_empty_world(jax.random.PRNGKey(0))
    cr = rc.make_state()
    pl = {**w["players"], "pos": w["players"]["pos"].at[G].set(jnp.array(player)),
          "active": w["players"]["active"].at[G].set(True)}
    if inv:
        pl["inv"] = pl["inv"].at[G, : len(inv)].set(jnp.asarray(inv, jnp.int32))
    n = cr["nodes"]
    for s, k, x, y in nodes:
        n = {**n, "pos": n["pos"].at[s].set(jnp.array([x, y])), "kind": n["kind"].at[s].set(k),
             "placed": n["placed"].at[s].set(True), "active": n["active"].at[s].set(s not in inactive)}
    dn = cr["dist_node"]
    for k, (x, y) in dict(fields).items():
        dn = dn.at[k].set(manhattan(x, y))
    pzm = w["pz"]
    for x, y in pz:
        pzm = pzm.at[x, y].set(True)
    return {**w, "players": pl, "pz": pzm, "craft": {**cr, "nodes": n, "dist_node": dn}}


def run(w):
    return _policy(w, KEY)


def test_gatherer_next_to_active_node_gathers_and_stands_still():
    w = make_world((C0, C0), nodes=[(0, C.NODE_COPPER, C0 + 1, C0)], fields={0: (C0 + 1, C0)})
    out = run(w)
    assert bool(out["gather"][G]) and int(out["move"][G]) == 0 and int(out["craft"][G]) == 0


def test_non_gatherers_never_gather_or_craft():
    w = make_world((C0, C0), nodes=[(0, C.NODE_COPPER, C0 + 1, C0)], fields={0: (C0 + 1, C0)})
    pl = w["players"]
    pos = pl["pos"].at[0].set(jnp.array([C0, C0 + 1])).at[2].set(jnp.array([C0 + 2, C0]))
    w = {**w, "players": {**pl, "pos": pos, "active": pl["active"].at[0].set(True).at[2].set(True),
                          "inv": pl["inv"].at[0, :2].set(C.ITEM_COPPER_ORE)}}
    out = run(w)
    for i in range(MAX_PLAYERS):
        if i % GATHER_EVERY != 1:
            assert not bool(out["gather"][i]) and int(out["craft"][i]) == 0


def test_walks_downhill_ties_to_lowest_direction():
    w = make_world((C0 - 10, C0), nodes=[(0, C.NODE_COPPER, C0, C0)], fields={0: (C0, C0)})
    assert int(run(w)["move"][G]) == 4  # east is the only lower neighbour
    w = make_world((C0 - 10, C0 - 10), nodes=[(0, C.NODE_COPPER, C0, C0)], fields={0: (C0, C0)})
    out = run(w)
    assert int(out["move"][G]) == 2 and not bool(out["gather"][G])  # south (2) and east (4) tie: lowest code


def test_never_steps_onto_pz_toward_a_node():
    w = make_world((C0 - 10, C0 - 10), nodes=[(0, C.NODE_COPPER, C0, C0)], fields={0: (C0, C0)},
                   pz=[(C0 - 10, C0 - 9)])  # the south neighbour is town
    assert int(run(w)["move"][G]) == 4


def test_leaves_town_downhill_when_standing_on_pz():
    w = make_world((C0 - 10, C0), nodes=[(0, C.NODE_COPPER, C0, C0)], fields={0: (C0, C0)},
                   pz=[(C0 - 10, C0), (C0 - 9, C0)])
    assert int(run(w)["move"][G]) == 4


def test_crafts_with_two_copper_ore_next_to_furnace():
    w = make_world((C0, C0), nodes=[(1500, C.NODE_FURNACE, C0 + 1, C0)], fields={0: (C0 + 30, C0), C.NODE_FURNACE: (C0 + 1, C0)},
                   inv=[C.ITEM_COPPER_ORE, C.ITEM_COPPER_ORE])
    out = run(w)
    assert int(out["craft"][G]) == C.RECIPE_COPPER_BAR and int(out["move"][G]) == 0 and not bool(out["gather"][G])


def test_walks_toward_furnace_only_with_a_full_bag_of_ore():
    kw = dict(nodes=[(1500, C.NODE_FURNACE, C0, C0)], fields={0: (C0 - 30, C0), C.NODE_FURNACE: (C0, C0)},
              pz=[(C0 - 9, C0)])  # towns are fine when walking to a station
    out = run(make_world((C0 - 10, C0), inv=[C.ITEM_COPPER_ORE] * INV_SLOTS, **kw))
    assert int(out["move"][G]) == 4 and int(out["craft"][G]) == 0 and not bool(out["gather"][G])
    # With room left it keeps gathering (west, down the copper field) instead of commuting with two ore.
    out = run(make_world((C0 - 10, C0), inv=[C.ITEM_COPPER_ORE, C.ITEM_COPPER_ORE], **kw))
    assert int(out["move"][G]) == 3 and int(out["craft"][G]) == 0


def test_depleted_node_or_full_inventory_falls_back_to_base_move():
    base = jax.jit(policies.bot_inputs)
    w = make_world((C0, C0), nodes=[(0, C.NODE_COPPER, C0 + 1, C0)], fields={0: (C0 + 1, C0)}, inactive=(0,))
    out = run(w)
    assert not bool(out["gather"][G]) and int(out["move"][G]) == int(base(w, KEY)["move"][G])
    full = make_world((C0 - 10, C0), nodes=[(0, C.NODE_COPPER, C0, C0)], fields={0: (C0, C0)},
                      inv=[C.ITEM_COPPER_BAR] * INV_SLOTS)  # nothing craftable from bars alone
    out = run(full)
    assert not bool(out["gather"][G]) and int(out["move"][G]) == int(base(full, KEY)["move"][G])


def test_unreachable_field_uses_base_move():
    w = make_world((C0, C0), nodes=[(0, C.NODE_COPPER, C0 + 5, C0)])
    w["craft"]["dist_node"] = jnp.full_like(w["craft"]["dist_node"], DIST_UNREACHABLE)
    out = run(w)
    assert int(out["move"][G]) == int(jax.jit(policies.bot_inputs)(w, KEY)["move"][G]) and not bool(out["gather"][G])


def test_low_level_iron_gatherer_works_copper():
    # slot 5 is the iron gatherer (5 // 4 == 1)
    w = make_world((C0 - 10, C0), nodes=[(0, C.NODE_COPPER, C0, C0)], fields={0: (C0, C0), 1: (C0 - 30, C0)})
    pl = w["players"]
    w = {**w, "players": {**pl, "pos": pl["pos"].at[5].set(jnp.array([C0 - 10, C0])), "active": pl["active"].at[5].set(True)}}
    assert int(run(w)["move"][5]) == 4  # level 1 < iron requirement: heads for the copper field, not iron


def test_shapes_dtypes_jit_and_vmap():
    w1 = make_world((C0, C0), nodes=[(0, C.NODE_COPPER, C0 + 1, C0)], fields={0: (C0 + 1, C0)})
    w2 = make_world((C0 - 10, C0), nodes=[(0, C.NODE_COPPER, C0, C0)], fields={0: (C0, C0)})
    out = run(w1)
    assert out["gather"].shape == (MAX_PLAYERS,) and out["gather"].dtype == jnp.bool_
    assert out["craft"].shape == (MAX_PLAYERS,) and out["craft"].dtype == jnp.int32
    assert out["move"].shape == (MAX_PLAYERS,) and out["move"].dtype == jnp.int32
    assert out["attack"].dtype == jnp.bool_ and out["eat"].dtype == jnp.bool_
    ws = jax.tree.map(lambda a, b: jnp.stack([a, b]), w1, w2)
    outs = jax.jit(jax.vmap(craft_bot_inputs))(ws, jnp.stack([KEY, KEY]))
    assert outs["gather"].shape == (2, MAX_PLAYERS)
    assert bool(outs["gather"][0, G]) and not bool(outs["gather"][1, G]) and int(outs["move"][1, G]) == 4


def test_sidesteps_a_living_player_on_the_downhill_tile():
    # South (2) and east (4) tie; a living player on the south tile makes east the only lower free neighbour,
    # and a dead one does not block. Two gatherers on different fields could otherwise deadlock face to face.
    w = make_world((C0 - 10, C0 - 10), nodes=[(0, C.NODE_COPPER, C0, C0)], fields={0: (C0, C0)})
    pl = w["players"]
    pl = {**pl, "pos": pl["pos"].at[2].set(jnp.array([C0 - 10, C0 - 9])), "active": pl["active"].at[2].set(True)}
    assert int(run({**w, "players": pl})["move"][G]) == 4
    pl = {**pl, "dead": pl["dead"].at[2].set(True)}
    assert int(run({**w, "players": pl})["move"][G]) == 2


def test_crafts_any_held_recipe_not_only_its_own_chain():
    # A copper gatherer holding two herbs brews a potion (no station needed) instead of carrying them around.
    w = make_world((C0, C0), inv=[C.ITEM_HERB, C.ITEM_HERB], fields={0: (C0 + 20, C0)})
    out = run(w)
    assert int(out["craft"][G]) == C.RECIPE_HERB_POTION and int(out["move"][G]) == 0
