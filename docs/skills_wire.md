# Skills wire format (M2, team ro-skills)

**Nothing in this document is implemented by the skills team.** The engine side is `game/ro_skills.py` (phase,
`make_state`, `buff_stat_mods`), `game/ro_skills_effects.py` and the tables in `game/data/skills.py` /
`game/data/elements.py`. The **net team** implements everything below: `net/protocol.py` (validation, `welcome`,
`snap.you`), `net/server.py` (message handling, input plumbing, per-tick clearing) and the host pull of the
`world["skills"]` block. The **client team** implements the UI. Contract source: `docs/m2_skills_brief.md`.

## Entity ids (the `target` convention)
Global ids, as in `game/view.py`: players `0..MAX_PLAYERS-1` (the player's slot), monsters `MAX_PLAYERS + i`,
drops `MAX_PLAYERS + MAX_MONSTERS + i` (drops are never valid skill targets). The ids in `snap.ents[*].id` are
already in this space, so the client sends the `id` of the entity it clicked. `-1` means no target. Self-target
skills (`target` type 0) ignore the value.

## Client -> server messages
```
{"t": "skill", "id": <int skill id 1..32>, "target": <int entity id, or -1>}
{"t": "learn", "id": <int skill id 1..32>}
```
Validation (`validate_client`): `id` is an int (not bool) in `1..N_SKILLS-1`; `target` is an int in
`-1..N_ENTITIES-1`; anything else is dropped (return `None`). The server does not check learned level, SP, range
or job: the JAX phase does and silently ignores an invalid request.

Mapping to engine inputs, for the sender's slot `s` only, for **exactly one tick** (like `attack`: the server sets
them when the message arrives, the next tick consumes them, then clears them back to 0 / -1 after that tick):

| message | input array (int32 `[MAX_PLAYERS]`) |
|---|---|
| `skill` | `inputs["skill"][s] = id`, `inputs["target"][s] = target` |
| `learn` | `inputs["learn"][s] = id` |

Defaults for every other slot (and for the sender once cleared): `skill = 0`, `target = -1`, `learn = 0`.
Bots build the same arrays in `bots/ro_skills_policy.py`. `skill` and `learn` are independent, so a client may
send both in the same tick. A second `skill` message before the tick runs overwrites the first (last wins).
`empty_inputs()` gains these three keys (the lead adds them; the engine does not run without them once the phase
is wired). Ground skills (Fire Wall, Arrow Shower) are targeted by clicking an entity: the effect lands on the
tile that entity stood on when the cast started. Click-on-empty-ground targeting is not in v1.

## `welcome` additions (static tables, sent once)
```
"skills": {
  "names": [str x 33],                      # index = skill id, [0] = "" / none; from SKILL_NAME
  "jobs": ["Novice", "Swordsman", "Mage", "Archer", "Acolyte", "Thief", "Merchant"],   # index = job id
  "list": [                                  # one entry per skill id 1..32 (index 0 omitted or null)
    {"id": 4, "name": "Bash", "job": 1, "max_level": 10,
     "effect": 1,            # 0 passive, 1 weapon damage, 2 magic damage, 3 heal, 4 buff, 5 none (stub)
     "target": 2,            # 0 self, 1 ally, 2 enemy, 3 ground
     "element": 0,           # index into "elements"
     "aoe": 0,               # radius in tiles (Chebyshev), 0 = single target
     "hits": 1,
     "buff": 0,              # buff id applied by effect 4, else 0
     "levels": [             # entries for level 1..max_level (level n is levels[n-1])
        {"sp": 8, "cast": 0, "cooldown": 4, "range": 1, "power": 1300, "buff_ticks": 0}
     ]}
  ]
},
"elements": ["Neutral", "Water", "Earth", "Fire", "Wind", "Poison", "Holy", "Shadow", "Ghost", "Undead"],
"buffs": ["", "Blessing", "Increase AGI", "Angelus", "Endure", "Provoke", "Hiding", "Play Dead"],
```
All numbers are straight from the `SKILL_*` arrays: `sp` = `SKILL_SP[id, lvl]`, `cast` = `SKILL_CAST[id, lvl]` (ticks
at `cast_mult` 1000, before the stats modifier), `cooldown` = `SKILL_COOLDOWN`, `range` = `SKILL_RANGE` (manhattan
tiles), `power` = `SKILL_POWER` (x1000 per hit), `buff_ticks` = `SKILL_BUFF_TICKS`. `ELEM_NAME` supplies
`elements`; the net team should also send the 10x10 `ELEM_MULT` table (percent, `[attack][defend]`) if the client
wants element tooltips. One tick is `tick_ms` (120 ms): the client converts ticks to seconds for tooltips and
the cast bar. Build the dict once at server start from the arrays (`np.asarray(...).tolist()`), not per client.

## `snap.you` additions (sent every snapshot to the owner only)
| field | type | meaning |
|---|---|---|
| `sp` | int | current SP (`skills.sp[slot]`) |
| `max_sp` | int | `derived["max_sp"][slot]` (the stats team; `50` under the fallback) |
| `buffs` | `[[buff_id, ticks_left, level], ...]` | only buffs with `ticks_left > 0` (`skills.buffs`, `skills.buff_level`), ascending buff id |
| `cooldowns` | `[[skill_id, ticks_left], ...]` | only skills with `ticks_left > 0` (`skills.cooldown`), ascending skill id |
| `learned` | `[[skill_id, level], ...]` | only skills with `level > 0` (`skills.learned`) |
| `casting` | int | skill id being cast, `0` = idle (`skills.casting`) |
| `cast_left` | int | ticks until the cast resolves (`skills.cast_left`; meaningful while `casting != 0`) |
| `skill_points` | int | `stats.skill_points[slot]` when the stats block exists; omit the field when it does not (learning is free then) |

`cast_target` is an optional extra (int, `-1` = none) so the client can draw the cast line. Total cast time for the
bar is not sent per tick: the client remembers the first `cast_left` it sees for a given `casting` (or computes
`levels[lvl-1].cast * cast_mult / 1000` with `cast_mult` from the stats `you` block). Buff durations only
decrease, so a buff icon shows `ticks_left * tick_ms / 1000` seconds. These fields are small and change often, so
they are always included in `you` (no delta suppression).

Pull plan: the per-tick host pull (`tick_fn` in `net/server.py`) adds `sp, casting, cast_left, buffs, buff_level,
cooldown, learned` rows for the viewer slots only (a few hundred bytes per viewer); build the lists on the host.

## What the client must do
1. **Skill bar:** show learned castable skills (`learned` joined with the welcome table, `effect != 0`) with a
   cooldown sweep from `cooldowns` (full length `levels[lvl-1].cooldown`), grey when `sp < levels[lvl-1].sp`,
   and bind to hotkeys. Pressing one sends `skill`; for `target` 0 send `-1`, for ally/enemy send the clicked or
   currently selected entity id (ally self-cast: own slot id).
2. **Cast bar:** while `casting != 0` draw a bar from `cast_left` (fills as it counts to 0) and the skill name;
   an instant skill never shows a bar. Moving does not cancel a cast in v1.
3. **Buff icons:** one icon per `buffs` row with a countdown, the level as a badge, names from `welcome.buffs`.
4. **SP bar** next to the HP bar from `sp / max_sp`.
5. **Learn panel:** list the player's job skills plus Novice skills, show level / max, the `skill_points` count and a
   plus button that sends `learn`; grey it when `skill_points == 0` or the level is at `max_level`. Re-read the
   state from the next snapshot (the server never acknowledges a `learn` or `skill` message).
6. **Feedback:** damage numbers come from monster hp deltas in `snap.ents` (there is no skill event stream in v1).
