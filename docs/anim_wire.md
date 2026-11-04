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
| `h` | damage taken this tick | only present on the tick a swing resolved on this entity: the sum of all hits that tick, or **`0` = Miss** (every swing at it missed; the client shows "Miss", no flinch); **`h <= -2` = a crit of `|h|`** (yellow number) |

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

Round 3 (ro3-wire):
- **The dead stay on the wire.** A player or monster that dies stays visible for `DEAD_VISIBLE_MS` = 3000 ms with
  `a = 7`, `m = 0` (`game/ro_anim.dying`, read by `game/view.py` `vis`), then leaves the view as `gone`. The client
  plays the dead clip from `a = 7` and drops the corpse on `gone` (no second corpse); the old "play dead on gone with
  hp 0" fallback stays for a monster that vanished without ever showing `a = 7`.
- **Sit** (`a = 2`, `m = 0`, held): client message `{"t": "sit"}` toggles sitting on the next tick (the engine input
  `sit` bool, `game/survival.py`): sitting -> stand (idle), standing -> sit. Moving, attacking or dying ends it.
  HP regen is doubled while sitting (`SIT_REGEN_MULT`). A sitting entity hit does not flinch (sit is not flinchable).
- **Pickup** (`a = 3`, `m = PICKUP_MS` 300): set by `game/loot.py` on the tick a player takes a drop.
- **Miss**: `h = 0` on the tick every swing at the entity missed (anim `hit_dmg` = -1 inside the tick; the wire
  sends 0 in both JSON and binary rows, `max(hit, 0)`). A hit later in the same tick replaces the miss.

## `snap.you` additions

| key | meaning |
|---|---|
| `speed_ms` | your walk speed in ms/cell (same as your entity's `s`) |
| `amotion_ms` | your current attack clip length (ASPD and Increase AGI included) |
| `after_delay` | ticks of after-cast delay left; no skill can start while > 0 (grey out the skill bar) |
| `warped` | `true` only on the tick a warp, waybell teleport or respawn moved you 2+ cells (snap the camera; absent otherwise; the Godot EntityStore resets it to false on every other snap) |

`amotion_ms` and `after_delay` read 0 until the server pulls `world["anim"]` amotion and `skills.after_delay`
(lines reported to the lead and netperf2 in the ro-anim final report).

## Snap encoding
The Godot client sends `"format": "bin"` at login by default (`NetSession.snap_format`; `"json"` opts out). Binary
layout: `docs/proto_v2_notes.md` / `net/protocol.py` BIN_HEADER, BIN_ROW (layout 1 unchanged by round 3).

## Known gaps (after round 3)
- A miss is per target, not per attacker: when two attackers swing at one target in a tick and one lands, only the
  damage shows (no separate "Miss" for the other).
- A monster whose respawn (`MONSTER_RESPAWN_TICKS` 60) is shorter than 3 s would cut its corpse short (not the case today).
