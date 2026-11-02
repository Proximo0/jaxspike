# M4 gathering and crafting: what the net layer and client must expose (ro-craft team, 2026-11-02)

Engine side is `game/ro_craft.py` (+ `game/ro_craft_nodes.py`, tables in `game/data/craft.py`), bot policy
`bots/ro_craft_policy.py`. Nothing in `net/` or `client/` is changed by this team; this file lists what the lead and
the net/client teams wire.

## `snap.you` additions
- `craft`: `{"level": [6 ints], "exp": [6 ints]}` in skill order `mining, smithing, fishing, cooking, herbalism,
  woodcutting` (`game.data.craft.SKILL_NAMES`). Source: `world["craft"]["level"][slot]`, `world["craft"]["exp"][slot]`.
  Send both only when changed (they change on a successful gather or craft), else omit, like the other delta fields.
- `gathers`, `crafts`: lifetime counters (`world["craft"]["gathers"][slot]`, `["crafts"][slot]`), optional for the HUD.

## Entities: nodes become visible entities
Nodes are a fourth entity class after players, monsters and drops. Proposed global ids for `game/view.py`:
`MAX_PLAYERS + MAX_MONSTERS + MAX_DROPS + i` for node slot `i` (`N_ENTITIES` grows by `MAX_NODES = 2048`; `view_update`'s
`[V, E]` intermediates grow by 30%, so the lead should benchmark). `entity_table` rows for nodes:
- `pos`: `nodes["pos"]`
- `vis`: `nodes["placed"]` (inactive nodes stay visible, drawn depleted; see `hp` below)
- `hp`: `nodes["active"]` as 0/1 (1 = harvestable, 0 = depleted/respawning), `max_hp` 1
- `kind`: `NODE_VIEW_KIND_BASE + nodes["kind"]` = 200..207 (`game.data.craft.NODE_VIEW_KIND_BASE`, `NODE_NAMES`):
  200 copper ore, 201 iron ore, 202 coal, 203 fishing spot, 204 herb, 205 tree, 206 furnace, 207 cooking fire.
Client (`client/`): kind >= 200 draws a node sprite by `kind - 200`, greyed when `hp == 0`; nodes never block walking.
Protocol v2 `ents` entries for nodes: `{kind: 200..207, pos, hp}` (no name, no max_hp needed).

## Client messages
| t | fields | input mapping | notes |
|---|---|---|---|
| `gather` | none | `inputs["gather"][slot] = True` for the next tick, then False | the server (or the client) may repeat it while a node is adjacent and the player stands still; a node is gatherable within manhattan 1 (`GATHER_RANGE`) |
| `craft` | `recipe` int 1..N_RECIPES-1 | `inputs["craft"][slot] = recipe` for the next tick, then 0 | invalid ids are clipped to 0 by the engine; the client should grey out recipes whose skill level, inputs or station are missing (compute from `welcome` tables and `you.inv`) |
Like `attack` and `eat`, both are one-tick pulses the server clears after the tick.

## `welcome` additions (tables for tooltips and the craft panel)
- `craft_items`: `{id: name}` for ids 100..105 (raw) and 200..208 (products) from `game.data.craft.ITEM_NAMES`.
- `recipes`: list of `{id, inputs: [ids], output, skill, level, exp, station}` from the `RECIPE_*` arrays
  (`station` is a node kind 6/7 or -1 for anywhere).
- `nodes`: list of `{kind, name, skill, level, product}` from `NODE_*` for kinds 0..7.
- `craft_skills`: `SKILL_NAMES` and `LEVEL_EXP` (51 ints) so the client can draw exp bars.

## Notices
The engine returns no reason for a failed gather or craft (the roll failed, the node was taken by a lower slot, no
room, level too low). The server can derive the common ones in Python from `you` and the recipe table if the client
wants a notice; the client already has enough to grey out impossible actions.

## Item ids introduced (raw 100..199, products 200..299)
See `game/data/craft.py` `ITEM_*`. M3 (items) owns 4..99 and must not reuse 100..299. Soak invariant note: the lead's
`bench/soak_check.py` checks `inv < NUM_ITEMS`; after merge that line must become "id in the known set" (the
ro-craft soak harness `bench/soak_craft.py` already does this).
