# M5 wire: parties, MVP bosses, kafra storage, chat

Additive to `docs/brief.md` (Networking) and `docs/ro_contract.md`. Old keys keep their meaning; unknown keys are ignored by older clients. Messages keep `"v"` and `"t"` (`net/protocol.py::encode`).

## snap.you additions

`you` today is `{pos, hp, max_hp, kills, deaths, inv, town, safe}`. M5 adds:

| field | type | sent | notes |
|---|---|---|---|
| `party` | int | every snap | party id from `world["party"]["party_id"]`, -1 when none |
| `leader` | bool | every snap | true when this player leads its party; false without a party |
| `members` | list of str | on change only, else omitted | names of all active members in slot order, self included; `[]` when leaving or disbanding |
| `storage` | list of 100 int | on change only, else omitted | kafra storage item ids, `ITEM_NONE` for empty slots |

"On change" means the server compares with the last value it sent to that slot and sends the full list when it
differs. A missing key means "unchanged".

## Client messages

Validation (`validate_client`) adds: `party.op` in the four names, `target` a str or null, `chat.kind` in the three
kinds, `chat.text` a str, `storage.op` in `in|out`, `storage.slot` a real int (not bool). Wrong types get the usual
`error` "malformed message" and a close.

| t | fields | input mapping / server action | notes |
|---|---|---|---|
| `party` | `op`: `create`, `leave`, `invite`, `accept`; `target`: name or null | `inputs["party"][slot]` = 1, 2, 3, 4; `inputs["target"][slot]` = slot of the named player, -1 when unknown or null | `target` only matters for `invite`. Invite with an unknown name sends `notice` "no such player" and still sends op 3 with target -1, which the tick ignores. |
| `chat` | `kind`: `say`, `party`, `whisper`; `text`: str; `to`: name or null | no tick input; appended to the tick's chat list for `ChatRouter` | `to` required for `whisper`. Text is stripped and cut to 200 by the router. |
| `storage` | `op`: `in`, `out`; `slot`: int | only while the player stands in a town box; else `notice` "no kafra here" | `in`: `slot` is an inventory slot (0..15). `out`: `slot` is a storage slot (0..99). Runs `net.kafra.transfer`; a refused transfer changes nothing and sends no notice. |
| `save` | none | `inputs["save"][slot]` = True for one tick | `ro_kafra.phase` moves spawn to that town's temple. Outside a town: no effect. |

`party` and `save` are one-tick pulses. After the tick's inputs are built, the server resets
`inputs["party"]` and `inputs["target"]` to 0 and -1 and `inputs["save"]` to False for every slot, so an action
fires once even if the client sends it once.

## Server messages

| t | direction | fields | notes |
|---|---|---|---|
| `chat` | S->C | `kind`: `say`, `party`, `whisper`, `system`; `from`: name (absent for `system`); `text`; `tick` | one message per routed line, sent as its own JSON message, in tick order. Produced by `ChatRouter.route` (`net/chat.py`). |
| `mvp` | S->C, every client | `kind`: int index into `mvp_kinds`; `killer`: name or null (no damager); `tick` | broadcast once per new row of the MVP event ring. |

Chat delivery is per recipient: `say` reaches the view radius, `party` reaches party members, `whisper` reaches the
named player only, and the sender gets `system` "no such player" for an unknown whisper target.

## Welcome data

`welcome` gains `mvp_kinds`: a list indexed by the `ro_mvp` kind (not the entity `mkind`), `null` for non-MVP kinds. Current table:

```json
"mvp_kinds": [null, null, "Orc Lord"]
```

The list is built from `game/ro_mvp.py` at server start (length `N_MVP_KINDS`) and is display text only.

## Lines for lead-owned files

Imports (`game/state.py` or wherever `make_empty_world` lives, and `game/step.py`):

```python
from game import ro_party, ro_mvp, ro_kafra
```

`empty_inputs()` (`game/state.py`), added next to the existing keys:

```python
"party": jnp.zeros(MAX_PLAYERS, jnp.int32),
"target": jnp.full(MAX_PLAYERS, -1, jnp.int32),
"save": jnp.zeros(MAX_PLAYERS, bool),
```

`make_empty_world` (`game/state.py`):

```python
"party": ro_party.make_state(),
"mvp": ro_mvp.make_state(),
```

`game/step.py`, directly after the survival phase call, before the tick ends:

```python
world = ro_party.phase(world, inputs)
world = ro_mvp.phase(world)
world = ro_kafra.phase(world, inputs)
```

`game/combat.py`, after the player-hit scatter and before the death handling (`jnp.where(hit, PLAYER_ATTACK, 0)`
is the per-attacker damage, shape `[MAX_PLAYERS]`, zero for misses):

```python
world = ro_mvp.record_damage(world, pidx, tgt_w, jnp.where(hit, PLAYER_ATTACK, 0))
```

`net/characters.py` (lead-owned, Python):

```python
from net.kafra import new_storage, character_storage
# new_character(): return dict gains  "storage": new_storage()
# load(): after reading the npz, set  char["storage"] = character_storage(char)
# save(): the storage array is saved as an int32 key named "storage" (round trip tested in tests/test_kafra.py)
```

`net/protocol.py`: `validate_client` gains the `party`, `chat`, `storage` and `save` cases above; `welcome_message`
gains `"mvp_kinds"`; `snapshot` gains the `party`, `leader`, `members` and `storage` keys of `you`.

`net/server.py` (lead-owned):

```python
PARTY_OP = {"create": 1, "leave": 2, "invite": 3, "accept": 4}

# in the message handler, per client message:
if t == "party":
    self.pending["party"][slot] = PARTY_OP[msg["op"]]
    self.pending["target"][slot] = self._slot_of(msg.get("target"))    # -1 when None or unknown
    if msg["op"] == "invite" and self.pending["target"][slot] < 0:
        self.notice[slot] = "no such player"
elif t == "chat":
    self.chat_msgs.append({"slot": slot, "kind": msg["kind"], "text": msg["text"], "to": msg.get("to")})
elif t == "storage":
    self.storage_ops.append((slot, msg["op"], msg["slot"]))   # town check and transfer run after the tick
elif t == "save":
    self.pending["save"][slot] = True

# once per tick, after building the tick's inputs and after the tick runs:
party_np = np.asarray(self.world["party"]["party_id"])
routed = self.router.route(self.tick, self.chat_msgs, pos_np, active_np, party_np, self.names)
for s, msgs in routed.items():
    self.outbox[s].extend(msgs)
self.chat_msgs.clear()

# MVP event ring: last_event_head starts at the world's event_head when the server starts
head = int(np.asarray(self.world["mvp"]["event_head"]))
ev = np.asarray(self.world["mvp"]["event"])
for i in range(min(head - self.last_event_head, EVENT_RING)):
    kind, killer, tick = ev[(self.last_event_head + i) % EVENT_RING].tolist()
    msg = {"t": "mvp", "kind": kind, "killer": self.names.get(killer), "tick": tick}
    for c in self.clients.values():
        c.send(msg)
self.last_event_head = head

# reset the one-tick pulses after the tick
self.pending["party"][:] = 0
self.pending["target"][:] = -1
self.pending["save"][:] = False
```

Notes for the lead: `head` is an unbounded count, so the write index is `head % EVENT_RING` in `ro_mvp.phase`. `ChatRouter`
takes `pos_np` and `active_np` from the same host pull as the snapshots. Storage ops update `players.inv` and the
per-slot storage array, then resend `you.storage` and `you.inv` in the next snap.
