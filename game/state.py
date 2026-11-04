"""World state schema and constructor. Owned by the lead.

The world is a plain nested dict (a JAX pytree). Every phase reads and writes
only these keys. Shapes are fixed. Inactive slots keep their last values but
their `active` flag is False, and phases must ignore them.

world = {
  "walls":   int8  [MAP, MAP]                 1 = blocked
  "terrain": int8  [MAP, MAP]                 T_* ids from config
  "pz":      bool  [MAP, MAP]                 protection zone: monsters cannot hit players here
  "dist_town": int16 [NT, MAP, MAP]           walking distance to each town's gate aprons (map_gen.distance_fields)
  "tick":    int32 []
  "rng":     uint32 [2]                       JAX PRNG key, advanced by next_key()
  "players": {
      "pos":        int32 [MAX_PLAYERS, 2]
      "hp":         int32 [MAX_PLAYERS]
      "active":     bool  [MAX_PLAYERS]        logged in (may be dead)
      "dead":       bool  [MAX_PLAYERS]        hp reached 0 this or an earlier tick, awaiting respawn
      "spawn":      int32 [MAX_PLAYERS, 2]
      "attack_cd":  int32 [MAX_PLAYERS]        ticks until next attack allowed
      "inv":        int32 [MAX_PLAYERS, INV_SLOTS]  item ids, ITEM_NONE = empty
      "kills":      int32 [MAX_PLAYERS]
      "deaths":     int32 [MAX_PLAYERS]
  }
  "monsters": {
      "pos":        int32 [MAX_MONSTERS, 2]
      "hp":         int32 [MAX_MONSTERS]
      "kind":       int32 [MAX_MONSTERS]
      "active":     bool  [MAX_MONSTERS]       alive and on the map
      "spawn":      int32 [MAX_MONSTERS, 2]
      "respawn_in": int32 [MAX_MONSTERS]       >0 while dead, counts down to respawn
      "attack_cd":  int32 [MAX_MONSTERS]
      "died":       bool  [MAX_MONSTERS]       set by combat on the tick it dies, cleared each tick
  }
  "drops": {
      "pos":    int32 [MAX_DROPS, 2]
      "item":   int32 [MAX_DROPS]               ITEM_NONE if slot unused
      "active": bool  [MAX_DROPS]
      "ttl":    int32 [MAX_DROPS]
      "count":  int32 [MAX_DROPS]               units in the pile (a coin pile's count is its ryn)
      "refine": int32 [MAX_DROPS]
      "cards":  int32 [MAX_DROPS, 4]
  }
}

inputs (per tick, built by bots or the network layer):
  {"move": int32 [MAX_PLAYERS] in 0..8 (config.DIR_DELTAS), "attack": bool [MAX_PLAYERS], "eat": bool [MAX_PLAYERS]}
  plus the RO one-shot keys of empty_inputs(). Round-2 blocks: "move" (game/ro_move.py), "anim" (game/ro_anim.py),
  "monai" (game/ro_monster_modes.py).
"""
import jax
import jax.numpy as jnp

from game.config import (
    INV_SLOTS,
    MAP,
    MAX_DROPS,
    MAX_MONSTERS,
    MAX_PLAYERS,
    PLAYER_MAX_HP,
    T_GRASS,
)
from game.data.items import EQ_BODY, EQ_WEAPON, ITEM_COTTON_SHIRT, ITEM_KNIFE
from game.world_layout import TOWNS

STARTER_KIT = (ITEM_KNIFE, ITEM_COTTON_SHIRT)  # RO Novice starter gear (ro6-balance): inventory slots 0 and 1, worn


def starter_items(it):
    """ro_items block with STARTER_KIT worn: one unit in slots 0 and 1, weapon slot -> 0, body slot -> 1."""
    return {**it, "inv_count": it["inv_count"].at[:, :len(STARTER_KIT)].set(1),
            "equip": it["equip"].at[:, EQ_WEAPON].set(0).at[:, EQ_BODY].set(1)}


def next_key(world):
    """Return (sub_key, new_world) with the world's rng advanced.

    The stored rng is raw uint32 key data (shape [2]) so it can be vmapped and saved.
    """
    key = jax.random.wrap_key_data(world["rng"])
    key, sub = jax.random.split(key)
    return sub, {**world, "rng": jax.random.key_data(key)}


def empty_inputs():
    return {
        "move": jnp.zeros((MAX_PLAYERS,), jnp.int32),
        "attack": jnp.zeros((MAX_PLAYERS,), jnp.bool_),
        "eat": jnp.zeros((MAX_PLAYERS,), jnp.bool_),
        # RO systems (docs/ro_contract.md; one-shot per tick, -1 / 0 / False = none)
        "allot": jnp.full((MAX_PLAYERS,), -1, jnp.int32),   # stats: stat index 0..5
        "job": jnp.full((MAX_PLAYERS,), -1, jnp.int32),     # stats: first job 1..6
        "skill": jnp.zeros((MAX_PLAYERS,), jnp.int32),      # skills: skill id
        "learn": jnp.zeros((MAX_PLAYERS,), jnp.int32),      # skills: skill id
        "target": jnp.full((MAX_PLAYERS,), -1, jnp.int32),  # skills: entity id; party invite: player slot
        "use_item": jnp.full((MAX_PLAYERS,), -1, jnp.int32),
        "equip": jnp.full((MAX_PLAYERS,), -1, jnp.int32),
        "refine": jnp.full((MAX_PLAYERS,), -1, jnp.int32),
        "drop": jnp.full((MAX_PLAYERS,), -1, jnp.int32),
        "gather": jnp.zeros((MAX_PLAYERS,), jnp.bool_),
        "craft": jnp.zeros((MAX_PLAYERS,), jnp.int32),      # recipe id
        "party": jnp.zeros((MAX_PLAYERS,), jnp.int32),      # 1 create, 2 leave, 3 invite, 4 accept
        "save": jnp.zeros((MAX_PLAYERS,), jnp.bool_),       # waybell save point
        "buy": jnp.zeros((MAX_PLAYERS,), jnp.int32),        # shop: item id (0 = none)
        "sell": jnp.full((MAX_PLAYERS,), -1, jnp.int32),    # shop: inventory slot
        "sit": jnp.zeros((MAX_PLAYERS,), jnp.bool_),        # survival: toggle sitting (anim SIT; HP regen twice as often)
        "teleport": jnp.full((MAX_PLAYERS,), -1, jnp.int32),  # waybell teleport: destination town index (ro_waybell)
        "pvp": jnp.zeros((MAX_PLAYERS,), jnp.bool_),        # pvp: toggle the PK flag (ro_pvp.phase; refused in towns)
    }


def make_empty_world(key):
    """Blank world with no walls, no players, no monsters. Used by tests and as a base."""
    from game import (ro_anim, ro_craft, ro_drops, ro_items, ro_monster_modes, ro_move, ro_mvp,  # lazy: next_key
                      ro_party, ro_pvp, ro_skills, ro_stats)
    return {
        "walls": jnp.zeros((MAP, MAP), jnp.int8),
        "terrain": jnp.full((MAP, MAP), T_GRASS, jnp.int8),
        "pz": jnp.zeros((MAP, MAP), jnp.bool_),
        "dist_town": jnp.zeros((len(TOWNS), MAP, MAP), jnp.int16),
        "tick": jnp.zeros((), jnp.int32),
        "rng": jax.random.key_data(jax.random.PRNGKey(0) if key is None else key),
        "players": {
            "pos": jnp.zeros((MAX_PLAYERS, 2), jnp.int32),
            "hp": jnp.full((MAX_PLAYERS,), PLAYER_MAX_HP, jnp.int32),
            "active": jnp.zeros((MAX_PLAYERS,), jnp.bool_),
            "dead": jnp.zeros((MAX_PLAYERS,), jnp.bool_),
            "spawn": jnp.zeros((MAX_PLAYERS, 2), jnp.int32),
            "attack_cd": jnp.zeros((MAX_PLAYERS,), jnp.int32),
            "inv": jnp.zeros((MAX_PLAYERS, INV_SLOTS), jnp.int32).at[:, :len(STARTER_KIT)].set(jnp.asarray(STARTER_KIT, jnp.int32)),
            "kills": jnp.zeros((MAX_PLAYERS,), jnp.int32),
            "deaths": jnp.zeros((MAX_PLAYERS,), jnp.int32),
        },
        "monsters": {
            "pos": jnp.zeros((MAX_MONSTERS, 2), jnp.int32),
            "hp": jnp.zeros((MAX_MONSTERS,), jnp.int32),
            "kind": jnp.zeros((MAX_MONSTERS,), jnp.int32),
            "active": jnp.zeros((MAX_MONSTERS,), jnp.bool_),
            "spawn": jnp.zeros((MAX_MONSTERS, 2), jnp.int32),
            "respawn_in": jnp.zeros((MAX_MONSTERS,), jnp.int32),
            "attack_cd": jnp.zeros((MAX_MONSTERS,), jnp.int32),
            "died": jnp.zeros((MAX_MONSTERS,), jnp.bool_),
        },
        "drops": {
            "pos": jnp.zeros((MAX_DROPS, 2), jnp.int32),
            "item": jnp.zeros((MAX_DROPS,), jnp.int32),
            "active": jnp.zeros((MAX_DROPS,), jnp.bool_),
            "ttl": jnp.zeros((MAX_DROPS,), jnp.int32),
            **ro_drops.drop_fields(),  # round 3: count (a coin pile's count is its ryn), refine, cards [MAX_DROPS, 4]
        },
        # RO system blocks (docs/ro_contract.md). Nodes and MVPs are placed by bots.runner.make_world_batch.
        "stats": ro_stats.make_state(),
        "skills": ro_skills.make_state(),
        "items": starter_items(ro_items.make_state()),  # Novices start with a Knife and a Cotton Shirt worn
        "craft": ro_craft.make_state(),
        "party": ro_party.make_state(),
        "mvp": ro_mvp.make_state(),
        # Round 2: 8-direction movement with speed accumulators, animation state, monster AI modes.
        "move": ro_move.make_state(),
        "anim": ro_anim.make_state(),
        "monai": ro_monster_modes.make_state(),
        "pvp": ro_pvp.make_state(),  # ro4-pvp: PK flag, pvp_kills, pvp_deaths
    }
