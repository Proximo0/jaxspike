# Animation fields on the wire (ro-anim, protocol v2 additive)

For the client team (client-anim). Source of truth: `game/ro_anim.py` (state and rules), `game/view.py`
(`entity_table` columns, `changed_mask`), `net/protocol.py` (`entity_rows`, `ro_you`). Contract:
`docs/round2_contract.md` 1.2. All animation logic runs in the JAX tick; the client only plays clips.

## `snap.ents[*]` (players kind 0 and monsters kind 1..; drops and nodes carry none of these)

| key | meaning | notes |
|---|---|---|
| `a` | action | 0 idle, 1 walk, 2 sit, 3 pickup, 4 standby, 5 attack, 6 hurt, 7 dead, 8 cast |
| `d` | face | 0..7, RO order: 0 S, 1 SW, 2 W, 3 NW, 4 N, 5 NE, 6 E, 7 SE (mirror 5..7 from 3..1) |
| `t` | start_tick | server tick the action began; clip time = `(render_tick - t) * tick_ms` |
| `m` | dur_ms | one-shot clip length to fit the clip into; `0` = looping clip (idle, walk, standby, sit) or held (dead) |
| `s` | speed_ms | ms per straight cell step (diagonal x 1.4); interpolate positions over this, not over `tick_ms` |
| `h` | damage taken this tick | only present on the tick a hit landed; the sum of all hits that tick |

An entity is re-sent (in `ents`) whenever `a`, `d`, `t`, `s` change or it takes a hit, besides the old pos/vis/hp/
kind triggers. A new swing restarts the clip: `t` changes even when `a` stays 5.

Durations: attack `m` = amotion (half the real swing period, RO's delay = 2 x amotion); the atk frame (damage
number, target flinch) is at about 60% of `m`. Hurt `m` = 300. Cast `m` = cast ticks x `tick_ms`. An instant
skill plays attack with `m` = amotion.

Transitions the server makes (the client never infers them): a one-shot (pickup, attack, hurt, cast) ends after
`m` and becomes standby if the entity swung in the last 3 s, else idle; standby turns idle 3 s after the last
swing; walk starts on a step (or while the move block says moving) and ends one still tick after the intent stops;
hurt only interrupts idle/walk/standby (never an attack or a cast); dead holds until respawn, then idle. Facing
follows the step direction while walking; attacks and casts face their target.

## `snap.you` additions

| key | meaning |
|---|---|
| `speed_ms` | your walk speed in ms/cell (same as your entity's `s`) |
| `amotion_ms` | your current attack clip length (ASPD and Increase AGI included) |
| `after_delay` | ticks of after-cast delay left; no skill can start while > 0 (grey out the skill bar) |

`amotion_ms` and `after_delay` read 0 until the server pulls `world["anim"]` amotion and `skills.after_delay`
(lines reported to the lead and netperf2 in the ro-anim final report).

## Known gaps (v1)
- Dead entities are not visible (`vis` is `active & ~dead` for players, `active` for monsters), so `a = 7` reaches
  the client only on the tick a viewer still knows the entity; a monster's death today arrives as `gone`. Play
  the dead clip on `gone` for a monster whose last `hp` was 0.
- Misses carry no `h` (no "Miss" text yet).
