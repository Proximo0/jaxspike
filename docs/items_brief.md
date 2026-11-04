# M3 brief: RO items, equipment, cards, refine, ryn and drop tables (sub-lead owned, 2026-11-02)

Read `CLAUDE.md` and `docs/ro_contract.md` first. This brief is the contract for the M3 workers. Everything here is
fixed-shape JAX inside the tick: no Python `if` on traced values, no loops over players or slots, masks over fixed
shapes, randomness only through `next_key`, ties by scatter-min (lowest index wins). `jax.jit(phase)` and
`jax.vmap(phase)` must both work.

Do NOT edit `game/config.py`, `game/state.py`, `game/step.py`, `game/loot.py`, `game/survival.py`, `bots/runner.py`,
`bots/policies.py`, `net/*`. Report lines for them instead.

## Files and owners

| Worker | Owns (nothing else) |
|---|---|
| DATA | `game/data/__init__.py`, `game/data/items.py` |
| DROPS | `game/data/drops.py`, `game/ro_drops.py` |
| INV | `game/ro_items.py` |
| REFINE | `game/ro_refine.py` |
| TEST | `tests/test_ro_items.py`, `bots/ro_items_policy.py`, `bench/items_soak.py`, `docs/items_wire.md`, `docs/items_persist.md` |

## Ids and constants (pinned; `game/data/items.py` exports all of these as plain ints and `jnp` arrays)

Kinds: `KIND_NONE=0, KIND_ETC=1, KIND_CONSUMABLE=2, KIND_WEAPON=3, KIND_ARMOR=4, KIND_CARD=5`.

Weapon classes: `WC_NONE=0, WC_DAGGER=1, WC_SWORD=2, WC_BOW=3, WC_STAFF=4, WC_MACE=5, WC_AXE=6`.

Equip slots (`N_EQUIP_SLOTS = 8`): `EQ_WEAPON=0, EQ_HEAD=1, EQ_BODY=2, EQ_SHIELD=3, EQ_GARMENT=4, EQ_SHOES=5,
EQ_ACC1=6, EQ_ACC2=7`. Armor "slot type" ids (`SLOT_*`, used in the item table's `CLASS` column for armor and for
the card target): `SLOT_WEAPON=0, SLOT_HEAD=1, SLOT_BODY=2, SLOT_SHIELD=3, SLOT_GARMENT=4, SLOT_SHOES=5,
SLOT_ACCESSORY=6` (an accessory goes to `EQ_ACC1`, or `EQ_ACC2` when ACC1 is taken).

Elements: `ELEM_NEUTRAL=0, ELEM_WATER=1, ELEM_EARTH=2, ELEM_FIRE=3, ELEM_WIND=4, ELEM_POISON=5, ELEM_HOLY=6,
ELEM_SHADOW=7, ELEM_GHOST=8, ELEM_UNDEAD=9`.

Bonus ids (`N_BONUS = 14`): `BONUS_STR=0, BONUS_AGI=1, BONUS_VIT=2, BONUS_INT=3, BONUS_DEX=4, BONUS_LUK=5,
BONUS_ATK=6, BONUS_DEF=7, BONUS_MATK=8, BONUS_HIT=9, BONUS_FLEE=10, BONUS_CRIT=11, BONUS_MAX_HP=12, BONUS_MAX_SP=13`.

Other constants: `MAX_CARDS = 4` (sockets per item), `MAX_STACK = 9999`, `MAX_REFINE = 10`, `RYN_PER_COIN = 10`
(a `GOLD_COIN` in the inventory becomes 10 ryn), `MAX_RYN = 2_000_000_000`.

Item ids (`N_ITEMS = 73`). Ids 0..3 are the lead's `config.py` ids and keep their meaning. Columns are RO classic
numbers (keep it fun, not exact). Every item has: `KIND, CLASS (weapon class | armor slot type | card target slot
type | 0), ATK, DEF, MATK, ELEMENT, SLOTS (0..4 card sockets), WEIGHT, BUY, SELL, HEAL_HP, HEAL_SP, WEAPON_LEVEL
(1..4 weapons, 0 else), STACKABLE (etc, consumables, cards; MEAT is NOT stackable in v1 because survival.py still
eats it by clearing the slot id), BONUS, BONUS_VALUE (an equip bonus on the item itself, e.g. Ring STR+2; 0/0 none),
CARD_BONUS, CARD_VALUE (for cards)`.

| id | name | kind | notes |
|---|---|---|---|
| 0 | NONE | none | |
| 1 | GOLD_COIN | etc, stackable | sell 10; converted to ryn on pickup |
| 2 | MEAT | consumable, NOT stackable | heal 40 (`MEAT_HEAL`) |
| 3 | SWORD | weapon sword lvl1 | atk 25, 3 slots, weight 50, buy 100 |
| 4 | RED_POTION | consumable | heal 45 |
| 5 | ORANGE_POTION | consumable | heal 105 |
| 6 | YELLOW_POTION | consumable | heal 175 |
| 7 | WHITE_POTION | consumable | heal 325 |
| 8 | BLUE_POTION | consumable | sp 60 |
| 9 | APPLE | consumable | heal 16 |
| 10 | JELLY_BEAD | etc | sell 3 |
| 11 | CLOVER | etc | |
| 12 | FEATHER | etc | |
| 13 | FLUFF | etc | |
| 14 | STICKY_MUCUS | etc | |
| 15 | SHELL | etc | |
| 16 | ANIMAL_SKIN | etc | |
| 17 | GRUKK_TOKEN | etc | |
| 18 | TIN_LODE | etc | refine ore, weapon level 1 |
| 19 | BRONZE_LODE | etc | refine ore, weapon level 2 |
| 20 | STARSTEEL | etc | refine ore, weapon level 3 and 4 |
| 21 | MOONSTONE | etc | refine ore, armor |
| 22 | KNIFE | weapon dagger lvl1 | atk 17, 3 slots |
| 23 | MAIN_GAUCHE | weapon dagger lvl1 | atk 43, 3 slots |
| 24 | BLADE | weapon sword lvl1 | atk 53, 3 slots |
| 25 | FALCHION | weapon sword lvl1 | atk 39, 4 slots |
| 26 | SCIMITAR | weapon sword lvl3 | atk 85, 2 slots |
| 27 | BOW | weapon bow lvl1 | atk 15, 3 slots |
| 28 | COMPOSITE_BOW | weapon bow lvl1 | atk 29, 4 slots |
| 29 | ROD | weapon staff lvl1 | atk 15, matk 15, 3 slots |
| 30 | WAND | weapon staff lvl2 | atk 25, matk 25, 2 slots |
| 31 | CLUB | weapon mace lvl1 | atk 23, 3 slots |
| 32 | MACE | weapon mace lvl1 | atk 37, 3 slots |
| 33 | AXE | weapon axe lvl1 | atk 38, 3 slots |
| 34 | BATTLE_AXE | weapon axe lvl2 | atk 80, 3 slots |
| 35 | DAMASCUS | weapon dagger lvl3 | atk 118, 1 slot |
| 36 | HAT | armor head | def 2, 1 slot |
| 37 | CAP | armor head | def 3, 1 slot |
| 38 | HELM | armor head | def 6, 0 slots |
| 39 | COTTON_SHIRT | armor body | def 1, 1 slot |
| 40 | JACKET | armor body | def 2, 1 slot |
| 41 | ADVENTURER_SUIT | armor body | def 2, 1 slot |
| 42 | MANTLE | armor body | def 4, 1 slot |
| 43 | GUARD | armor shield | def 3, 1 slot |
| 44 | BUCKLER | armor shield | def 4, 1 slot |
| 45 | SHIELD | armor shield | def 6, 1 slot |
| 46 | HOOD | armor garment | def 1, 1 slot |
| 47 | MUFFLER | armor garment | def 2, 1 slot |
| 48 | SANDALS | armor shoes | def 1, 1 slot |
| 49 | SHOES | armor shoes | def 2, 1 slot |
| 50 | BOOTS | armor shoes | def 4, 1 slot |
| 51 | RING | armor accessory | STR+2, 0 slots |
| 52 | CLIP | armor accessory | 1 slot |
| 53 | NECKLACE | armor accessory | VIT+2 |
| 54 | GLOVE | armor accessory | DEX+2 |
| 55 | BROOCH | armor accessory | AGI+2 |
| 56 | ROSARY | armor accessory | LUK+2 |
| 57 | BLOBBIN_CARD | card, target body | LUK+2 |
| 58 | WRIGGLET_CARD | card, target body | VIT+1 |
| 59 | COCOONLING_CARD | card, target body | MAX_HP+70 |
| 60 | MOONBUN_CARD | card, target body | CRIT+1 |
| 61 | FIDDLEHOP_CARD | card, target body | VIT+1 |
| 62 | PILFERBUG_CARD | card, target accessory | AGI+1 |
| 63 | GRUKK_BRUTE_CARD | card, target weapon | ATK+5 |
| 64 | ANTLET_CARD | card, target weapon | ATK+20 |
| 65 | EMBERBLOB_CARD | card, target accessory | HIT+3 |
| 66 | BUZZEYE_CARD | card, target accessory | FLEE+2 |
| 67 | PEEPIT_CARD | card, target weapon | ATK+10 |
| 68 | CROAKLE_CARD | card, target body | MAX_HP+40 |
| 69 | CAPLING_CARD | card, target body | VIT+2 |
| 70 | NEEDLEBUZZ_CARD | card, target weapon | ATK+3 |
| 71 | CRAGWING_CARD | card, target shield | FLEE+10 |
| 72 | BRAMBLET_CARD | card, target head | MAX_SP+80 |

Every item id is exported as `ITEM_<NAME>`; `NAMES` is a tuple of 73 strings (lower-case display names, e.g.
"red potion"). `ITEM_TABLE` is a dict of column name -> `jnp.int32 [N_ITEMS]` (`STACKABLE` as `jnp.bool_`), and each
column is also a module-level array `KIND, CLASS, ATK, DEF, MATK, ELEMENT, SLOTS, WEIGHT, BUY, SELL, HEAL_HP, HEAL_SP,
WEAPON_LEVEL, STACKABLE, BONUS, BONUS_VALUE, CARD_BONUS, CARD_VALUE`. Also `REFINE_ORE: jnp.int32 [5]` indexed by
weapon level (0 = armor -> MOONSTONE, 1 -> TIN_LODE, 2 -> BRONZE_LODE, 3 -> STARSTEEL, 4 -> STARSTEEL) and
`REFINE_RYN: jnp.int32 [5]` = (2000, 50, 200, 5000, 20000). `table_for_client() -> list[dict]` returns the table as
plain Python for the welcome message (id, name, kind, cls, atk, def, matk, element, slots, weight, buy, sell,
heal_hp, heal_sp, weapon_level, stackable, bonus, bonus_value, card_bonus, card_value). Build the columns from plain
Python tuples / lists in one place (a row per item) so the ids above and the columns cannot drift; assert
`len(rows) == N_ITEMS` at import.

## State block `world["items"]` (`ro_items.make_state() -> dict`)

```
inv_count:  int32 [MAX_PLAYERS, INV_SLOTS]      units in the slot (0 when players.inv is ITEM_NONE)
inv_refine: int32 [MAX_PLAYERS, INV_SLOTS]      0..10
inv_cards:  int32 [MAX_PLAYERS, INV_SLOTS, 4]   card item ids per socket, 0 = empty
equip:      int32 [MAX_PLAYERS, 8]              inventory slot index of the item worn in that equip slot, -1 = none
ryn:       int32 [MAX_PLAYERS]
```
The item id of a slot stays in the lead's `world["players"]["inv"]` (`loot.py` writes new pickups there into the
first empty slot with no count; `survival.py` eats MEAT by setting the slot id to 0). The items phase keeps the
parallel arrays consistent (see `_normalize`).

## Inputs (new keys; all `int32 [MAX_PLAYERS]`, -1 = none)

`inputs["use_item"]`, `inputs["equip"]`, `inputs["refine"]`, `inputs["drop"]`: an inventory slot index. Out-of-range
values are ignored. `empty_inputs()` line for the lead: `"use_item": jnp.full((MAX_PLAYERS,), -1, jnp.int32)` and
the same for the other three. Tests and the harness build inputs as `{**empty_inputs(), **ro_items.empty_item_inputs()}`.

## `game/ro_items.py` (INV)

```
make_state() -> dict
empty_item_inputs() -> dict        the four keys above, all -1
equipped(world) -> dict            weapon_atk, weapon_elem, armor_def, armor_elem: int32 [MAX_PLAYERS]; card_bonus: int32 [MAX_PLAYERS, N_BONUS]
phase(world, inputs) -> world      normalize -> use_item -> equip -> refine (ro_refine.refine) -> drop -> normalize
```
- `_normalize(world)`: (1) slots with `inv != 0` and `inv_count == 0` get count 1 (fresh pickups); slots with
  `inv == 0` get count 0, refine 0, cards 0. (2) `GOLD_COIN` slots: `ryn += count * RYN_PER_COIN` (clamped to
  `MAX_RYN`), slot cleared. (3) Stacks merge: for STACKABLE kinds, all slots holding the same id move their counts
  into the lowest such slot (clamped to `MAX_STACK`, overflow lost) and the others are cleared. Vectorize with an
  `[MAX_PLAYERS, INV_SLOTS, INV_SLOTS]` equality table and `argmax` for the lowest matching slot; no loops.
  (4) `equip` entries that point to a slot whose id is 0, or whose kind is not weapon/armor, become -1; two equip
  entries pointing at the same slot keep the lowest equip slot only.
- `_use_item(world, inputs)`: slot valid, player active and not dead, `inv_count > 0`. Consumable: `hp = min(hp +
  HEAL_HP, PLAYER_MAX_HP)` (max_hp comes from `ro_stats.derived` later; say so in a comment), `HEAL_SP` ignored for
  now (no sp array yet; report the line), count -1, slot cleared at 0. Card: `ro_refine.compound_card` (see below),
  count -1 only where that function reports success. Any other kind: no-op.
- `_equip(world, inputs)`: slot valid, alive, kind weapon or armor. If the slot is already in `equip` anywhere,
  unequip it (toggle). Else target = `EQ_WEAPON` for weapons, `CLASS` for armor slot types 1..5, accessories go to
  `EQ_ACC1` if it is -1 else `EQ_ACC2` (replacing what is there). Equipping replaces the previous occupant.
- `_drop(world, inputs)`: slot valid, alive, count > 0. Place one unit as a ground drop at the player's tile in a
  free drop slot (`world["drops"]`: pos, item, active, ttl = DROP_TTL), using the same rank-of-free-slot scatter as
  `loot._spawn_drops` (lowest free slot to the lowest player index; players beyond the free count drop nothing and
  keep the item). Stackables lose one unit; a weapon/armor is removed whole (its refine and cards are lost with it,
  the drop array has no room for them; report as a known limit) and any `equip` pointer to it becomes -1.
- Helpers other modules may call: `item_of(world) -> int32 [MAX_PLAYERS, INV_SLOTS]` is just `players["inv"]`;
  `clear_slots(world, mask [MAX_PLAYERS, INV_SLOTS]) -> world` zeroes id, count, refine, cards and fixes `equip`.
- `equipped`: `weapon_atk = ATK[id] + REFINE_ATK[weapon_level] * refine` where `REFINE_ATK = (0, 2, 3, 5, 7)`;
  `weapon_elem = ELEMENT[id]`; `armor_def = sum over EQ_HEAD..EQ_ACC2 of DEF[id] + refine`; `armor_elem =
  ELEMENT[body item]`; all 0 when the equip slot is -1. `card_bonus[p, b]` = sum of `BONUS_VALUE` of worn items with
  `BONUS == b` plus `CARD_VALUE` of every card in their sockets with `CARD_BONUS == b` (sockets beyond `SLOTS[id]`
  and card id 0 contribute nothing). Build with one `.at[pidx, bonus_id].add(value)` scatter over the flattened
  `[MAX_PLAYERS, 8 * (1 + MAX_CARDS)]` list; no loops.

## `game/ro_refine.py` (REFINE)

```
REFINE_CHANCE: jnp.int32 [5, 11]   per-mille success for refining FROM level r (column r) an item of weapon level
                                   0 (armor) .. 4; 1000 up to the safe limit: armor +4, lvl1 +7, lvl2 +6, lvl3 +5, lvl4 +4
refine(world, inputs, key) -> world
compound_card(world, slot [MAX_PLAYERS], ok [MAX_PLAYERS]) -> (world, done [MAX_PLAYERS] bool)
```
Chances above safe (RO pre-renewal, per-mille, by target level): armor +5..+10 = 600, 400, 400, 200, 200, 90;
lvl1 +8, +9, +10 = 600, 400, 190; lvl2 +7..+10 = 600, 400, 200, 190; lvl3 +6..+10 = 600, 500, 200, 200, 190;
lvl4 +5..+10 = 600, 400, 400, 200, 200, 90. Column r holds the chance of going from r to r+1; column 10 is 0.
- `refine`: slot valid, alive, kind weapon or armor, `inv_refine < MAX_REFINE`, the player has at least one unit of
  `REFINE_ORE[weapon_level]` in some slot and `ryn >= REFINE_RYN[weapon_level]`. Pay the ryn, consume one ore
  unit from the lowest slot holding it (a slot that reaches 0 is cleared). Roll `jax.random.uniform(key,
  (MAX_PLAYERS,))`; `u * 1000 < chance` succeeds: `inv_refine += 1`. Failure destroys the item: id 0 in
  `players["inv"]`, count 0, refine 0, cards 0, and any `equip` entry pointing at the slot becomes -1.
- `compound_card`: for each player, `slot` holds a card item (kind card) with count > 0 and `ok` is True.
  Target equip slot from `CLASS[card]` (`SLOT_WEAPON..SLOT_SHOES` map to `EQ_WEAPON..EQ_SHOES`; `SLOT_ACCESSORY`
  picks `EQ_ACC1` if it holds an item with a free socket, else `EQ_ACC2`). The worn item at that equip slot must
  exist and have a free socket (`inv_cards[p, s, k] == 0` for some `k < SLOTS[id]`; lowest k). Write the card id
  there; `done` is True exactly for those players. The caller decrements the card count.

## `game/data/drops.py` + `game/ro_drops.py` (DROPS)

`drops.py`: `DROPS_PER_KIND = 6`, `N_DROP_KINDS = 2` (rat = `MONSTER_KIND_RAT`, orc = `MONSTER_KIND_ORC`; the lead's
monster table grows later, `roll_drops` clips kind to `N_DROP_KINDS - 1`). `DROP_ITEM: int32 [N_DROP_KINDS,
DROPS_PER_KIND]` item ids (0 = unused entry), `DROP_CHANCE: int32 [..]` per-mille, `DROP_COUNT: int32 [..]` units
(1 unless noted), `RYN_MIN, RYN_MAX: int32 [N_DROP_KINDS]` ryn (whole numbers; the roll is uniform inclusive).
Rat: JELLY_BEAD 700, STICKY_MUCUS 400, APPLE 200, RED_POTION 100, KNIFE 10, BLOBBIN_CARD 1; ryn 10..30.
Orc: GRUKK_TOKEN 550, ANIMAL_SKIN 300, ORANGE_POTION 100, TIN_LODE 100, AXE 20, GRUKK_BRUTE_CARD 1; ryn 30..80.
`table_for_client() -> list[dict]` per kind (item, chance, count, ryn range).

`ro_drops.py`:
```
roll_drops(world, key) -> (items int32 [MAX_MONSTERS, DROPS_PER_KIND], counts same, ryn int32 [MAX_MONSTERS])
place_drops(world, items, counts, ryn) -> world
```
- `roll_drops` rolls every entry independently for every monster slot (`u * 1000 < chance`), returns item 0 and
  count 0 where the roll failed or the entry is unused, and a ryn amount for every monster (the caller masks by
  `died`). Uses only `key`; the caller obtains it with `next_key`. Fixed shapes, no per-entity map tables.
- `place_drops` creates ground drops for monsters with `world["monsters"]["died"]` True: one drop per rolled entry
  plus one `GOLD_COIN` drop per monster with ryn > 0 (`count = ryn // RYN_PER_COIN`, at least 1). Flatten to
  `MAX_MONSTERS * (DROPS_PER_KIND + 1)` candidates, rank them with a cumsum, and assign free drop slots as
  `loot._spawn_drops` does (rank beyond the free count is discarded). Writes `pos, item, active, ttl` of
  `world["drops"]`. Until the lead adds `drops["count"]`, counts above 1 are lost on the ground (one unit each);
  say so in the docstring and report the line `"count": jnp.ones((MAX_DROPS,), jnp.int32)` for `state.py`.
- Report for the lead: in `loot._spawn_drops`, replace the `roll_item` call with
  `items, counts, ryn = ro_drops.roll_drops(world, sub)` and `world = ro_drops.place_drops(world, items, counts, ryn)`.

## Tests, bot policy, harness, docs (TEST)

- `tests/test_ro_items.py`: under 30 s on CPU; build worlds as `w = make_empty_world(key); w = {**w, "items":
  ro_items.make_state()}`, inputs as `{**empty_inputs(), **ro_items.empty_item_inputs()}`. Cover: table ids 0..3
  unchanged and `N_ITEMS == 73`; normalize (fresh pickup gets count 1, coin becomes ryn, stacks merge into the
  lowest slot, equip pointer to an emptied slot clears); use_item heals and caps at `PLAYER_MAX_HP`, decrements and
  clears, dead player can't; equip toggles, accessories fill ACC1 then ACC2, replacing; `equipped` sums weapon atk
  with refine, armor def, item bonus and card bonus by id, zeros when nothing worn; refine pays ryn and ore, safe
  levels always succeed (chance 1000), a forced failure (seed search or a chance-0 level) destroys the item and
  clears `equip`; compound_card fills the lowest free socket and refuses when full or wrong target; drop places a
  ground item at the player's tile with ttl `DROP_TTL`, one unit for stackables, whole for weapons, nothing when
  the drop array is full; `roll_drops` chances within 3 sigma over 20k rolls per entry and `place_drops` ranks
  slots like loot; one `jax.jit(jax.vmap(phase))` test over a 2-world batch equal to per-world `jax.jit(phase)`.
- `bots/ro_items_policy.py`: `item_inputs(world, key) -> dict` with the four keys: a living bot equips the
  highest-ATK weapon in its inventory when it is better than what it wears (one equip per tick), uses the first
  potion/APPLE slot when `hp < HUNT_RETREAT_HP`, refines its worn weapon when it holds the right ore and ryn and
  the weapon is below its safe limit, and compounds a held card when a worn item has a socket. Pure JAX.
- `bench/items_soak.py`: `--batch 4 --chunks 5 --ticks 100 --seed 0`. Builds `make_world_batch` worlds, inserts the
  items block, and runs a harness tick: `movement -> monsters -> combat -> drops via ro_drops (roll + place, i.e.
  loot._spawn_drops replaced; then loot._pickup and loot._decay) -> survival -> ro_items.phase -> tick + 1`, with
  inputs `{**bot_inputs, **item_inputs}`. Checks the base `soak_check.check_invariants` minus the `NUM_ITEMS`
  inventory check, plus: `inv_count >= 1` iff `inv != 0`, `inv_count <= MAX_STACK`, refine in 0..10, cards 0 or a
  card id with socket index `< SLOTS`, `equip` in -1..INV_SLOTS-1 and pointing at a weapon/armor of the right
  slot type, no slot equipped twice, `ryn >= 0`, player hp in range. Prints per batch: kills, deaths, items
  picked per bot per 1000 ticks by kind (etc / consumable / weapon / armor / card), ryn per bot, equips, refines
  (attempts, successes, breaks), cards compounded. Ends with `soak: clean` or the violation list; exit 1 on any.
- `docs/items_wire.md`: `snap.you` additions (`ryn`, `inv` as `[{item, count, refine, cards}]` per slot,
  `equip` as 8 slot indices), client messages `use_item {slot}`, `equip {slot}`, `refine {slot}`, `drop {slot}` and
  their input mapping (one-shot: applied for one tick, then back to -1), and `welcome.items` = `table_for_client()`
  plus `welcome.drops`. `docs/items_persist.md`: the per-character fields `net/characters.py` must save (`inv_count`
  int32 [INV_SLOTS], `inv_refine`, `inv_cards` [INV_SLOTS, 4], `equip` [8], `ryn` int) with their validation.

## Rules for every worker
- Only edit your files. Don't commit or push. Never run more than 2 engine processes at once; never leave processes
  running. Python: `C:\Users\crp27\jaxspike\venv\Scripts\python.exe` (absolute). Run from the worktree root
  `C:\Users\crp27\jaxspike\.claude\worktrees\ro-items-5fe81d`.
- Pure JAX in `game/` and `bots/`; tables are plain tuples turned into `jnp` arrays at import. Fixed shapes. No
  `[N, MAP, MAP]` tables. Masks over `players["active"] & ~players["dead"]`.
- Final message: files changed, the checks run with their output lines, and any line the lead must add elsewhere.
