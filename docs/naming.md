# Naming bible: the world of Aurelune

Team ro3-names, 2026-11-02. The user's direction: *"I'm a little concerned it's too like Ragnarok. Let's keep the
maps and names of towns and money and the names of things like kafra different."* The **mechanics stay** (jobs,
ASPD, zones with portals, cards, refine, parties, MVPs; `docs/ro_fidelity.md` keeps citing rAthena and irowiki as
the design source). The **proper nouns become our own**. Generic job names (Swordsman, Mage, Archer, Acolyte,
Thief, Merchant, Novice), generic skill names (Bash, Heal, Provoke...) and generic items (Red Potion, Knife, Cotton
Shirt, Apple, Shell, Feather...) stay.

This file is the single source of names. `tools/rename_names.py` applies it (the lead runs it at integration, when
nobody else is editing; `--dry-run` first) and `tests/test_naming.py` fails on any old name left behind. Status:
**pending** (not applied yet).

## 1. The world: Aurelune

Aurelune is a small, sunny continent where the two moons never set at the same time, so there is always a little
light in the sky. It is a bright anime world: cobbled squares with fountains, candy-coloured roofs, chibi heroes,
cute-or-creepy monsters, and a guild of bell-ringing clerks who can send you anywhere for a few coins. Nobody is
grim here; even the undead in the caves are mostly embarrassed about it.

### 1.1 Towns (index = `world_layout.TOWNS` index, roles unchanged)

| # | old | new | id prefix | one line of character |
|---|---|---|---|---|
| 0 | Prontera | **Solmera** | `sol` | The sunlit capital: a great fountain square ringed by shop awnings, a castle that is mostly balconies, and the first Waybell tower. |
| 1 | Geffen | **Lunavell** | `lun` | The mage city under the second moon: one tall crooked tower, violet rooftops, students who set their own hats on fire. |
| 2 | Payon | **Pinehollow** | `pin` | The archer village in the pine hills: wooden walkways, paper lanterns, a bow range that doubles as the main street. |
| 3 | Alberta | **Tidewick** | `tid` | The merchant port: salt-white docks, cranes, crates stacked like a board game, every fourth building is a counting house. |
| 4 | Morroc | **Kharsand** | `kha` | The desert thief town: red sandstone, striped awnings, a bazaar where half the stalls are a front for the Guild. |

Lowercase zone names follow the town: `solmera`, `lunavell`, `pinehollow`, `tidewick`, `kharsand`
(old `prontera`, `geffen`, `payon`, `alberta`, `morocc`).

### 1.2 Zones (same id pattern: 3-letter prefix + kind + number; kinds `fld` field, `dun` dungeon, `sew` sewer)

| old id | new id | what it is |
|---|---|---|
| `prt_sewb1` | `sol_sew01` | Solmera Culvert (old Prontera Culvert): the capital's sewer dungeon, home of the Gildscarab |
| `pay_dun00` | `pin_dun00` | Pinehollow Cave (old Payon Cave): the undead cave under the pine hills, home of Scorchstripe |
| `gef_fild00` | `lun_fld00` | Lunavell field |
| `prt_fild01` | `sol_fld01` | Solmera field, north (the beginner field) |
| `pay_fild01` | `pin_fld01` | Pinehollow field |
| `prt_fild05` | `sol_fld05` | Solmera field, west |
| `prt_fild08` | `sol_fld08` | Solmera field, centre (level 1-6) |
| `prt_fild06` | `sol_fld06` | Solmera field, east |
| `moc_fild01` | `kha_fld01` | Kharsand field (the desert) |
| `prt_fild02` | `sol_fld02` | Solmera field, south |
| `alb_fild01` | `tid_fld01` | Tidewick field (the coast) |

Dungeon display names come out of the town rename: "Prontera Culvert" -> "Solmera Culvert", "Payon Cave" ->
"Pinehollow Cave".

### 1.3 Currency: the Ryn

Old: zeny. New: **Ryn** (plural ryn), symbol **Ʀ** (U+01A6; ASCII fallback `R`). Struck in Solmera from
moon-silver; the coin has a crescent on one side and the fountain on the other. UI strings: "Ryn 1,234",
`Ʀ1,234` where a symbol fits. Identifier forms: `ryn`, `Ryn`, `RYN` (`MAX_RYN`, `ryn_label`, `items_ryn`,
`REFINE_RYN`, `RYN_PER_COIN`). The wire key `you.ryn` and the save field `items_ryn` rename with it (see section 6).

### 1.4 The Waybell (old: kafra)

The **Waybell** is the guild that runs the bell towers in every town square: ring the bell (talk to the clerk) to
**save** your return point, use the **storage** vault under the tower, or pay ryn to be **sent** to another
town's bell (teleport). The clerks are **Bellkeepers**: cheerful uniformed staff in blue-and-gold with a small
hand bell, one per town, the same face everywhere (it is the same guild). Identifier forms: `waybell`, `Waybell`,
`WAYBELL` (`T_WAYBELL`, `WAYBELL_TELEPORT`, `waybell_tiles`, `ro_waybell`, `net/waybell.py`). UI: "Waybell
storage (click = take out)", "Bellkeeper" as the NPC name, minimap marker "waybell".

### 1.5 Monsters (same 20 rows, same ids, same numbers and roles as `game/data/monsters.py`)

| id | old | new | role / look (anime chibi, bright) |
|---|---|---|---|
| 0 | Poring | **Blobbin** | the bouncy pink jelly starter: round, two dot eyes, hops, picks up loot (looter) |
| 1 | Orc Warrior | **Grukk Brute** | the orc-like warrior: green, tusks, a club bigger than its head; aggressive (legacy orc slot) |
| 2 | Fabre | **Wrigglet** | green caterpillar with a leaf on its head |
| 3 | Lunatic | **Moonbun** | white round bunny with red eyes, hops in the moonlight |
| 4 | Drops | **Emberblob** | the orange jelly (Fire): a Blobbin that sat too close to a campfire; looter |
| 5 | Poporing | **Vexblob** | the green jelly (Poison): sticky and faintly annoyed; looter |
| 6 | Willow | **Bramblet** | a walking stump with a bird's nest for a hat |
| 7 | Rocker | **Fiddlehop** | grasshopper with a fiddle: plays before it attacks |
| 8 | Spore | **Capling** | mushroom cap with legs and a shy face |
| 9 | Thief Bug | **Pilferbug** | small fast beetle that steals drops (assist, looter) |
| 10 | Zombie | **Rotling** | a shuffling green undead that is a bit sorry about it (cast sensor) |
| 11 | Wolf | **Snarlfang** | blue-grey wolf; calls the pack (assist) |
| 12 | Skeleton | **Clackbone** | a skeleton with a loose jaw that rattles |
| 13 | Peco Peco | **Gallopeck** | the big yellow riding bird: long legs, loud |
| 14 | Hode | **Sandgulper** | desert worm that swallows sand and spits pebbles |
| 15 | Soldier Skeleton | **Clackbone Guard** | a Clackbone with a dented helmet and a spear (assist, cast sensor) |
| 16 | Bigfoot | **Grumblebear** | big brown bear, grumpy, slow swipes (large) |
| 17 | Mummy | **Tombwrap** | bandaged undead, trailing a loose strip (cast sensor) |
| 18 | Golden Thief Bug | **Gildscarab** | MVP: a golden beetle the size of a cart, king of the Pilferbugs; lives in the Solmera Culvert |
| 19 | Eddga | **Scorchstripe** | MVP: a huge orange tiger with ember-coloured stripes; lives in the Pinehollow Cave |

Abbreviation: GTB -> **Gildscarab** (no abbreviation).

Monsters that exist only as cards or legacy art slots (`client/assets/slots.json` `monster.*` keys,
`client/assets/build_sheets.py`):

| old | new | note |
|---|---|---|
| Pupa | **Cocoonling** | Wrigglet's cocoon (card: Max HP) |
| Andre | **Antlet** | small ant (card: ATK) |
| Chonchon | **Buzzeye** | flying eyeball-fly (card: FLEE) |
| Picky | **Peepit** | yellow chick (card: ATK) |
| Roda Frog | **Croakle** | big frog (card: Max HP) |
| Hornet | **Needlebuzz** | wasp (card: ATK) |
| Condor | **Cragwing** | desert vulture (card: FLEE) |
| Baphomet Jr | **Imphorn** | legacy sheet slot only |

Legacy slots `monster.rat` and `monster.orc` (the pre-RO prototype kinds) are generic words and stay.

### 1.6 Items (RO-specific names only; generic names stay)

| old | new | note |
|---|---|---|
| Jellopy | **Jelly Bead** | the Blobbin drop (`ITEM_JELLY_BEAD`) |
| Fly Wing | **Flit Feather** | random teleport consumable |
| Butterfly Wing | **Homing Feather** | teleport to the save point |
| Phracon | **Tin Lode** | level 1 weapon refine ore |
| Emveretarcon | **Bronze Lode** | level 2 weapon refine ore |
| Oridecon | **Starsteel** | level 3-4 weapon refine ore |
| Elunium | **Moonstone** | armor refine ore |
| Orcish Voucher | **Grukk Token** | Grukk Brute drop |
| `<monster> Card` | `<new monster> Card` | every card follows its monster (Blobbin Card, Gildscarab Card...) |
| Wolf Claw | **Snarlfang Claw** | follows its monster (the only etc drop that names one) |

Kept as generic: Red/Orange/Yellow/White/Blue Potion, Apple, Meat, Clover, Feather, Fluff, Sticky Mucus, Shell,
Animal Skin, Trunk, Decayed Nail, Skel Bone, Rotten Bandage, Tiger Skin, Grasshopper Leg, Mushroom Spore, Bill of
Birds, Earthworm Peeling, Bear Foot, every weapon and armor (Knife, Cutter, Main Gauche, Blade, Falchion, Scimitar,
Damascus, Bow, Composite Bow, Rod, Wand, Club, Mace, Axe, Battle Axe, Hat, Cap, Helm, Cotton Shirt, Jacket,
Adventurer Suit, Mantle, Guard, Buckler, Shield, Hood, Muffler, Sandals, Shoes, Boots, Ring, Clip, Necklace, Glove,
Brooch, Rosary), the craft items (ores, bars, fish, herbs, planks).

## 2. Rules for future names

1. **No RO proper nouns**: no Ragnarok Online town, map id, monster, item, NPC, MVP, currency or service name,
   and no transliteration of one (not "Prontero", not "Porring"). "RO", "rAthena" and "irowiki" stay allowed in
   prose and comments that explain the *design source* (numbers, formulas, mechanics).
2. **No other trademarked game names**: nothing from Final Fantasy, Dragon Quest, Pokemon, Zelda, Maplestory,
   Tibia, WoW, Fallout, Minecraft, or any other game. Generic English words are fine (Wolf, Skeleton, Red Potion),
   but prefer an invented word for anything players will say out loud (monsters, towns, bosses).
3. **Web-check every candidate** before it lands here: search `"<Name>" game` and `"<Name>" game monster` (or
   `town`, `item`, `boss`). If the exact word is another game's proper noun, pick another. Record the check
   date in the commit or the hand-back. Every name in this file was checked on 2026-11-02; four first choices
   were replaced (Everlune: a PS4 game; Stumpkin: Dragon Quest; Stingwing: Fallout; Dustwing: Catch a Monster).
4. **Bright anime register**: short, bouncy, easy to say, a smile in it. Monsters are cute-or-creepy compound
   words (Blobbin, Fiddlehop, Tombwrap); towns are two soft syllables with a place suffix (-mera, -vell, -hollow,
   -wick, -sand); nothing edgy, nothing Latin-grim.
5. **One name, every form**: a new name must work as a Title ("Grukk Brute"), a lowercase row name
   ("grukk brute"), a snake id (`grukk_brute`, sprite folder and `monster.*` slot key) and a constant
   (`GRUKK_BRUTE`, `ITEM_GRUKK_BRUTE_CARD`). Add all four forms to `tools/rename_names.py` when a name changes.
6. **No new name may contain an old name** as a word (so the rename script cannot cascade), and no two names
   may share a snake id.
7. **Zone ids** keep the pattern `<3-letter town prefix>_<kind><2 digits>`: prefixes `sol lun pin tid kha`,
   kinds `fld dun sew`. A new town gets its own 3-letter prefix (unique among the five).
8. **Family naming**: variants of a monster reuse the base word (Blobbin / Emberblob / Vexblob; Clackbone /
   Clackbone Guard). A card is always `<monster> Card`.

## 3. Where the names live (every form the script maps)

- `game/world_layout.py`: `TOWN_NAMES`, `ZONES` ids, `DUNGEONS` display names, `FIELD_BANDS` monster names,
  `kafra_tiles`/`KAFRA_*` -> `waybell_tiles`/`WAYBELL_*`.
- `game/data/monsters.py`: `_ROWS` names, `_MODE_ROWS` keys, `KIND_BY_NAME` asserts.
- `game/data/items.py`: row names and `ITEM_*` constants (the assert `ITEM_<NAME.upper()>` == row index holds
  because both forms rename together), `_ORE_WORDS`.
- `game/data/drops.py`, `game/ro_shop.py`, `game/ro_mvp.py`, `game/ro_kafra.py` -> `game/ro_waybell.py`,
  `net/kafra.py` -> `net/waybell.py`, `net/protocol.py` (`kafra_teleport`, `zeny` keys), `net/char_fields.py`.
- Client: `client/assets/slots.json` keys (`monster.<id>`), `client/assets/anim/<id>/` folders and their
  `act.json` + `*.png.import` files, `client/assets/anim/index.json`, `client/assets/build_sheets.py`,
  `client/game/art/asset_slots.gd`, `client/tools/gen_placeholder_anim.py`, `client/game/ui/hud.gd` ("Zeny"),
  `client/game/ui/inventory_panel.gd` ("Kafra storage"), `client/game/ui/minimap.gd` (`KAFRA_COLOR`, marker kind),
  `client/game/input/cursor_state.gd` (`KAFRA_TERRAIN`), zone banner strings in tests.
- Docs and tests everywhere else (`docs/world_wire.md`, `docs/items_wire.md`, `tests/test_zones.py`...).

## 4. What the script leaves alone (and the test skips)

- `docs/ro_fidelity.md`, `docs/round2_contract.md`, `docs/round3_contract.md`, `docs/ro_contract.md`: they describe
  RO itself as the source (rAthena / irowiki citations, "towns (Prontera, Geffen...)"). Historical contracts.
- Any line holding a URL (`http://`, `https://`), a `rAthena` / `irowiki` / `iRO` citation, or the marker
  `NAMING_KEEP`.
- The word "RO" and "Ragnarok" in prose that names the design source (`CLAUDE.md`, `docs/production-handover.md`).
- `venv/`, `client/.godot/`, `characters/`, `.git/`, `.claude/` (other sessions' worktrees), `__pycache__/`.
- This file, `tools/rename_names.py` and `tests/test_naming.py` (they hold the old names on purpose).

## 5. Ambiguities the lead decides by hand (see the dry run)

- **Drops** the monster vs "drops" the loot block (`world["drops"]`, `MAX_DROPS`, `DROPS_PER_KIND`,
  "Drops the path..."). The script maps Title-case `Drops` only when it is not followed by a space and a lowercase
  word, `ITEM_DROPS_CARD` / `"drops card"` explicitly, and the lowercase id `drops` only inside the monster-id
  lists of the five client art files. Never `MAX_DROPS`, `_DROPS`, `world["drops"]`.
- **Spore** vs "mushroom spore" (kept), **Skeleton** vs the humanoid-rig "skeleton" in `client/docs/*` (kept),
  the CSS class `dash-skeleton` (kept).
- Sprite folders under `client/assets/anim/` and the `*.png.import` files inside them: the script renames the
  folders and rewrites `source_file=` paths; Godot re-imports them on the next editor open (the `.godot/imported`
  cache names are hashed from the old path and simply go stale).

## 6. Compatibility notes for the lead

- Player saves in `characters/` (git-ignored) store `items_zeny`; after the rename the loader reads `items_ryn`.
  Either delete the dev saves or add a one-line alias in `net/char_fields.py` (`it.get("zeny", it.get("ryn"))`).
- The wire keys `you.zeny` and `welcome.kafra_teleport` rename with the code on both sides (server and client are
  in one repo), so old clients cannot talk to a renamed server. Fine for development; note it in the protocol doc.
- Module renames: `game/ro_kafra.py` -> `game/ro_waybell.py`, `net/kafra.py` -> `net/waybell.py`,
  `tests/test_ro_kafra.py` -> `tests/test_ro_waybell.py`, `tests/test_kafra.py` -> `tests/test_waybell.py`
  (imports follow because `ro_kafra` / `net.kafra` are mapped as identifiers). Delete `__pycache__` after.
