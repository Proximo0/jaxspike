# RO fidelity audit (team ro-fidelity, round 2, 2026-11-02)

Target: classic **pre-renewal** Ragnarok Online. Every number below was read from rAthena's pre-renewal database
(GitHub `rathena/rathena`, branch `master`, folder `db/pre-re/`) and cross-checked with iRO wiki classic pages
where the two disagree. Data fixes are in the files this team owns; engine formulas we do not own are listed as
exact lines (section 3) for their owners. Tests: `tests/test_ro_fidelity.py` (63 checks green, 3 `xfail` that flip
to passes when the engine lines in section 3 land).

Sources (all fetched 2026-11-02):
- `job_exp.yml` (base/job exp), `job_basepoints.yml` (BaseHp/BaseSp per level), `job_stats.yml` (HpFactor,
  SpIncrease, MaxWeight), `job_aspd.yml` (base amotion per weapon), `attr_fix.yml` (element table),
  `mob_db.yml` (monsters, drops), `item_db_equip.yml` / `item_db_usable.yml` / `item_db_etc.yml` (items, cards),
  `skill_db.yml` (SP, cast, after-cast delay, cooldown, range, durations).
- iRO wiki classic (`https://irowiki.org/classic/<Skill>`): skill damage %, buff effects, Heal formula, HP/SP/ASPD
  formulas. Conflicts noted inline.
- rAthena source conventions: `pc_gets_status_point` = (old level + 15) / 5, `PC_STATUS_POINT_COST` =
  1 + (value + 9) / 10, `min_skill_delay_limit` = 100 ms.

## 1. Checked, with before -> after

### 1.1 Stats, exp, job change (`game/data/stats.py`)
| Item | Before | After (RO) | Source |
|---|---|---|---|
| Base exp table (1..98) | log-log curve, 9 / 300 at L10 / 3 M at L98 | exact table: 9, 16, 25, 36, 77 ... 320 at L10 ... 99,999,998 at L98 | job_exp.yml |
| Novice job exp (1..9) | curve 10..450 | 10, 18, 28, 40, 91, 151, 205, 268, 340 | job_exp.yml |
| First job exp (1..49) | curve 60..60,000 | 30, 43, 58 ... 509,596 | job_exp.yml |
| Status points / level | 3 + (L-1)/5 | unchanged: already RO (48 start + 1225 = **1273** at 99) | rAthena pc.cpp |
| Stat raise cost | 2 + (v-1)/10 | unchanged: already RO | rAthena pc.hpp |
| Base HP | job-neutral 150..3500 curve x job factor | **exact per job**: 35 + 5L + sum round(A*i), A = 0 / .7 / .3 / .5 / .4 / .5 / .4 (Nov/Swd/Mag/Arc/Aco/Thf/Mer). New `JOB_BASE_HP_TABLE[job][L]` (Novice L1 = 40, Swordsman L99 = 3999). Interim `BASE_HP_TABLE` = exact Archer curve, `JOB_HP_FACTOR` = (1000, 1250, 730, 1000, 857, 1000, 857) | job_basepoints.yml (formula checked against all 99 levels x 7 jobs) |
| Base SP | 10..650 curve x factor | exact 10 + SP_JOB*L, SP_JOB = 1/2/6/2/5/2/3 (Merchant 3). New `JOB_BASE_SP_TABLE`. Interim `BASE_SP_TABLE` = 10 + 2L, factors (850, 1000, 2781, 1000, 2335, 1000, 1445) | job_basepoints.yml |
| ASPD base | one per job 146..160 | `JOB_WEAPON_ASPD[job][weapon class]` = 200 - amotion/10 (e.g. Novice fist 150 / knife 135, Swordsman sword 145, Mage rod 130, Archer bow 130, Acolyte mace 140, Thief dagger 150, Merchant axe 130; 0 = cannot use); interim `JOB_ASPD_BASE` = starter weapon (135, 145, 130, 130, 140, 150, 130) | job_aspd.yml |
| SP regen cadence | 16 ticks (1.9 s) | 8 s standing (67 ticks at 120 ms, computed from TICK_MS) | iRO wiki "SP Recovery" |
| Max job level | 10 / 50 | unchanged (RO) | job_exp.yml |
| Job change | Novice at job 10 -> any first job | unchanged (RO also needs Basic Skill 9, i.e. the 9 points spent; not enforced: see 4) | iRO wiki |

### 1.2 Skills (`game/data/skills.py`)
Times are now written in RO ms and converted with `TICK_MS`. New columns: `SKILL_AFTER_DELAY_MS` (ms, the hook
ro_skills already reads) and `SKILL_DELAY_ASPD` (1 = RO delay is the caster's amotion). `SKILL_COOLDOWN` is now RO's
real reuse cooldown (only Provoke 1 s, Endure 10 s); the old per-skill 4..16 tick cooldowns were invented.
Every skill blocks the next for at least 100 ms (rAthena `min_skill_delay_limit`).

| Skill | Fixed (before -> after) |
|---|---|
| Bash | cooldown 4 ticks -> 0, ACD = amotion (ASPD); SP 8/15, 130%..400% already right |
| Magnum Break | AoE 3x3 -> **5x5** (aoe 2), cooldown 16 ticks -> ACD 2000 ms; 120%..300% fire right |
| Provoke | cooldown 8 ticks -> 1 s; SP 4..13 right |
| Endure | duration flat 9.6 s -> 7 + 3*lv s (10..37 s); cooldown -> 10 s |
| Fire / Cold / Lightning Bolt | ACD -> 800 + 200*lv ms (1.0..2.8 s); cast 0.7 s*lv, SP 12..30 right |
| Napalm Beat | SP 9,9,9,10.. -> **9,9,9,12,12,12,15,15,15,18**; damage 70%..160% -> **80%..170%**; ACD 1.0..0.5 s |
| Soul Strike | cast 0.5..2.4 s -> **0.5 s flat**; ACD 1200,1000,1400,...,1800 ms |
| Fire Wall | cast 2.16 s flat -> 2.0..0.65 s (2150 - 150*lv) |
| Double Strafe | cooldown -> ACD amotion; 2 hits x 100%..190% right |
| Arrow Shower | 75%..120% -> **80%..125%**; 3x3 -> **5x5**; ACD 1 s |
| Heal | cast 0.96 s -> **none**; ACD 1 s; element Holy; SP 13..40 right |
| Blessing | cast 0.96 s -> **none**; durations 60..240 s right |
| Increase AGI | ACD 1 s; cast 1 s, SP 18..45, 60..240 s right |
| Angelus | ACD 3.5 s; cast 0.5 s, 30 s*lv right |
| Cure | cast 0.5 s -> none; ACD 1 s |
| Hiding | duration 60..330 s -> **30 s*lv** |
| Steal | ACD 1 s |
| Play Dead | 100000 ticks -> 600 s |
| Vending, First Aid | invented cooldowns removed |
| **Appended** ids 33..41 | Frost Diver (SP 25..16, cast 0.8 s, ACD 1.5 s, 110%..200% water), Stone Curse (SP 25..16, cast 1 s, range 2), Improve Concentration (SP 25..70, 60..240 s), Teleport (lv 2, SP 10/9), Warp Portal (lv 4, SP 35/32/29/26, cast 1 s), Detoxify (SP 10), Improve Dodge, Overcharge (passives), Item Appraisal (SP 10). Status effects / teleports have no engine hook yet (EFFECT_NONE: they cost SP and play the cast). |

Unchanged and already RO: Bash SP, Provoke SP, bolts SP/cast/hits, Soul Strike SP/hits, Heal/Blessing/Inc AGI/
Angelus SP, Double Strafe, Mammonite (SP 5, 150%..600%), Envenom (SP 12), ranges (9 for spells and bows, 1 melee).

### 1.3 Elements (`game/data/elements.py`)
Against attr_fix.yml level 1: Earth vs Poison 125 -> **100**; Holy vs Undead 125 -> **150**; Ghost vs Shadow 100
-> **75**; Undead vs Shadow 100 -> **0**, Undead vs Undead 100 -> **0**; Poison vs Undead 50 -> 0 (RO -25 heals;
clamped). Everything else matched.

### 1.4 Monsters (`game/data/monsters.py`, `_ROWS` only; modes belong to ro-monsters)
Every one of the 20 rows was wrong (invented HP/ATK/exp scaled for a Tibia-like curve). Now rAthena pre-re:

| Monster | Lv | HP | ATK | DEF | MDEF | Base/Job exp | Element | Size | Race | Walk ms | rAthena Ai |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Poring | 1 | 50 | 7-10 | 0 | 5 | 2/1 | Water 1 | M | Plant | 400 | 02 passive looter |
| Orc Warrior | 24 | 1400 | 104-126 | 10 | 5 | 408/160 | Earth 1 | M | Demihuman | 200 | 04 aggressive |
| Fabre | 2 | 63 | 8-11 | 0 | 0 | 3/2 | Earth 1 | S | Insect | 400 | 01 passive |
| Lunatic | 3 | 60 | 9-12 | 0 | 20 | 6/2 | Neutral 3 | S | Brute | 200 | 01 |
| Drops | 3 | 55 | 10-13 | 0 | 0 | 4/3 | Fire 1 | M | Plant | 400 | 02 |
| Poporing | 14 | 344 | 59-72 | 0 | 10 | 81/44 | Poison 1 | M | Plant | 300 | 02 |
| Willow | 4 | 95 | 9-12 | 5 | 15 | 5/4 | Earth 1 | M | Plant | 200 | 01 |
| Rocker | 9 | 198 | 24-29 | 5 | 10 | 20/16 | Earth 1 | M | Insect | 200 | 01 |
| Spore | 16 | 510 | 24-48 | 0 | 5 | 66/108 | Water 1 | M | Plant | 200 | 01 |
| Thief Bug | 6 | 126 | 18-24 | 5 | 0 | 17/5 | **Neutral 3** (was Shadow) | S | Insect | 150 | 07 aggr. looter assist |
| Zombie | 15 | 534 | 67-79 | 0 | 10 | 50/33 | Undead 1 | M | Undead | 400 | 04 |
| Wolf | 25 | 919 | 37-46 | 0 | 0 | 329/199 | Earth 1 | M | Brute | 200 | 03 passive assist |
| Skeleton | 10 | 234 | 39-47 | 10 | 10 | 18/14 | Undead 1 | M | Undead | 200 | 17 passive, cast sensor |
| Peco Peco | 19 | 531 | 50-64 | 0 | 0 | 159/72 | Fire 1 | L | Brute | 200 | 03 passive assist |
| Hode | 26 | 2282 | 146-177 | 0 | 30 | 550/300 | Earth 2 | M | Brute | 200 | 01 |
| Soldier Skeleton | 29 | 2334 | 221-245 | 10 | 15 | 372/226 | Undead 1 | M | Undead | 200 | 04 |
| Bigfoot | 25 | 1619 | 198-220 | 10 | 0 | 310/188 | Earth 1 | L | Brute | 300 | 17 |
| Mummy | 37 | 5176 | 305-360 | 0 | 10 | 800/602 | Undead 2 | M | Undead | 300 | 04 |
| Golden Thief Bug | 64 | 126000 | 870-1145 | 60 | 45 | 14300/7150 | Fire 2 | L | Insect | 100 | 07, MVP |
| Eddga | 65 | 152000 | 1215-1565 | 15 | 15 | 25025/12870 | Fire 1 | L | Brute | 300 | 21, MVP |

Aggro flag changed to rAthena's Ai: Wolf, Skeleton, Hode, Bigfoot are **passive** in pre-re (were aggressive).
Sizes: Orc Warrior L -> M, Drops S -> M. MDEF, race, AGI/DEX/LUK (for HIT/FLEE) and element levels are not columns
yet (section 4); the values are in this table and in the mob_db extract below for whoever adds them.

Monster STR/AGI/VIT/INT/DEX/LUK (rAthena): Poring 1/1/1/0/6/30, Orc Warrior 1/24/48/25/34/10, Fabre 1/2/4/0/7/5,
Lunatic 1/3/3/10/8/60, Drops 1/3/3/0/12/15, Poporing 1/14/14/0/19/15, Willow 1/4/8/30/9/10, Rocker 1/9/18/10/14/15,
Spore 1/12/12/5/19/8, Thief Bug 1/6/6/0/11/0, Zombie 1/8/7/0/15/0, Wolf 1/20/28/15/32/20, Skeleton 1/5/10/0/12/0,
Peco Peco 1/13/13/25/27/9, Hode 1/26/42/5/49/40, Soldier Skeleton 1/15/22/5/40/15, Bigfoot 1/25/55/15/20/25,
Mummy 28/19/32/0/63/20, GTB 65/75/35/45/85/150, Eddga 78/70/85/66/90/85. Attack delay / motion ms: Poring
1872/672, Orc Warrior 1864/864, Wolf 1054/504, Zombie 2612/912, Skeleton 2228/528, Peco 1564/864, GTB 768/768,
Eddga 872/1344 (others in mob_db.yml).

### 1.5 Items (`game/data/items.py`)
Weights in our table are rAthena/10. Fixed (buy price / weight / slots / def / weapon level): Main Gauche
1000 -> 2400 z, Blade 1500 -> 2900, Falchion slots 4 -> 3, Scimitar 5000 -> 17000 and WL 3 -> 2, Bow 200 -> 1000,
Composite Bow slots 4 -> 3 and 1500 -> 2500, Mace 3000 -> 1600, Axe 3000 -> 500, Battle Axe 7000 -> 5400 and WL 2 -> 1,
Damascus 8000 -> 49000, Club 15 -> 120, Hat 80 -> 1000, Cap def 3 -> 4 and 200 -> 12000, Helm 1500 -> 44000,
Cotton Shirt weight 100 -> 10, Adventurer's Suit def 2 -> 3, Mantle 1000 -> 10000, Guard 200 -> 500, Buckler
1000 -> 14000, Shield 2400 -> 56000, Hood 100 -> 1000, Muffler 500 -> 5000, Sandals 200 -> 400, Shoes 800 -> 3500,
Boots 3500 -> 18000, accessories 1000 -> 30000 (Rosary 15000), Clip gets MaxSP +10; sell = buy / 2 throughout.
Potions heal the middle of RO's random range (Red 45-65 -> 55, Orange 105-145 -> 125, Yellow 175-235 -> 205,
White 325-405 -> 365, Blue SP 40-60 -> 50, Apple 16-22 -> 19). Etc loot buy prices (Sticky Mucus 140 -> 70,
Orcish Voucher 34 -> 142, Oridecon/Elunium 2000 -> 1100, the ro-items2 rows: Tree Root 12, Decayed Nail 82, Wolf
Claw 58, Skel Bone 232, Rotten Bandage 350, Tiger's Skin 548). Cards: NPC sell **10 z** (rAthena Buy 20) and slot /
main bonus from the card script: Fabre, Lunatic, Drops -> weapon; Thief Bug, Picky -> armor; Chonchon -> shoes;
Condor -> garment; Spore, Poporing -> accessory; Orc Warrior, Bigfoot -> shield; Zombie, Eddga -> shoes; Pupa
MaxHP 70 -> 700; Roda Frog 40 -> 400; Rocker ATK +5; Wolf ATK +15; Hode FLEE +5; cards whose effect needs a missing
hook (race/element resist, magic immunity, endure, skill grant) carry no bonus rather than an invented one.
**Appended** (ids 90..92): Cutter (dagger, ATK 30, 3 slots, 1250 z), Fly Wing (60 z), Butterfly Wing (300 z).
Every other requested staple already existed (Knife, Main Gauche, Sword, Falchion, Blade, Bow, Rod, Mace, Axe,
Cotton Shirt, Guard, Buckler, Hood, Sandals, Clip, Red/Orange/Yellow/White Potion).

## 2. Corrected drop rows for team ro-items2 (`game/data/drops.py`)
rAthena rates are per 10000; RO cards are **0.01 % (1 / 10000)**, which per-mille cannot express (ro_drops floors
it to 1 per mille = 10x RO). Recommend `DROP_CHANCE` per 10000 (engine roll `U[0, 9999] < chance`). Classic
monsters drop **no zeny** and **no potions** (ours are a deliberate income source; keep or drop as the lead likes).
Rows below use items we have, in rAthena per-10000; `(+)` = item to add as an etc row to be exact.

| Monster | Drops (item rate/10000) |
|---|---|
| Poring | Jellopy 7000, Apple 1000, Empty Bottle(+) 1500, Sticky Mucus 400, Apple 150, Knife 100, Unripe Apple(+) 20, Poring Card 1 |
| Orc Warrior | Orcish Voucher 9000, Iron(+) 210, Axe 100, Rough Oridecon(+) 40, Battle Axe 10, Orcish Axe(+) 5, Cigar(+) 3, Orc Warrior Card 1 |
| Fabre | Fluff 6500, Clover 1000, Green Herb(+) 700, Feather 500, Club 200, Club[slotted] 80, Azure Jewel(+) 5, Fabre Card 1 |
| Lunatic | Clover 6500, Apple 2000, Carrot(+) 1100, Feather 1000, Red Herb(+) 600, Rainbow Carrot(+) 20, Pierrot Nose(+) 4, Lunatic Card 1 |
| Drops | Jellopy 7500, Empty Bottle(+) 1700, Apple 1100, Apple 800, Sticky Mucus 500, Rod 80, Orange Juice(+) 20, Drops Card 1 |
| Poporing | Sticky Mucus 5500, Garlet(+) 1500, Green Herb(+) 500, Apple 250, Grape(+) 200, Apple 5, Main Gauche 5, Poporing Card 1 |
| Willow | Tree Root (trunk) 9000, Resin(+) 1500, Sweet Potato(+) 700, Wooden Block(+) 100, Willow Card 1 |
| Rocker | Grasshopper's Leg(+) 9000, Javelin(+) 80, Guitar/Leaflet/Doll/Hinalle(+) 10 each, Green Feeler(+) 4, Rocker Card 1 |
| Spore | Mushroom Spore(+) 9000, Red Herb(+) 800, Strawberry(+) 600, Blue Herb(+) 50, Hat 40, Spore Doll(+) 10, Poison Spore(+) 5, Spore Card 1 |
| Thief Bug | Worm Peelings(+) 2500, Jellopy 2000, Red Herb(+) 350, Iron Ore(+) 250, Jacket (Leather Jacket) 120, Thief Bug Card 1 |
| Zombie | Decayed Nail 9000, Sticky Mucus 1000, White Jewel(+) 70, Horrendous Mouth(+) 50, Cardinal Jewel(+) 5, Zombie Card 1 |
| Wolf | Wolf Claw 9000, Animal Skin 5500, Monster's Feed(+) 1050, Meat 650, Strawberry(+) 600, Emveretarcon 20, Mantle 10, Wolf Card 1 |
| Skeleton | Jellopy 3000, Red Herb(+) 850, Skel Bone 800, Phracon 90, Mace 80, Skul Ring(+) 30, Skeleton Card 1 |
| Peco Peco | Bill of Birds(+) 9000, Orange(+) 1000, Red Herb(+) 900, Yellow Herb(+) 200, Wand 100, Sandals 20, Peco Peco Card 1 |
| Hode | Earthworm Peeling(+) 9000, Sticky Mucus 3000, Yellow Live(+) 120, Rough Elunium(+) 80, Town Sword(+) 10, Hode Card 1 |
| Soldier Skeleton | Skel Bone 5500, Red Herb(+) 700, Knife 150, Rough Oridecon(+) 60, Stiletto(+) 50, Dagger(+) 12, Mementos(+) 10, Soldier Skeleton Card 1 |
| Bigfoot | Bear's Foot(+) 9000, Animal Skin 5000, Sweet Potato(+) 1500, Honey(+) 450, Stuffed Doll(+) 80, Rough Oridecon(+) 43, Poo Poo Hat(+) 5, Bigfoot Card 1 |
| Mummy | Rotten Bandage 9000, Yellow Herb(+) 850, Mementos(+) 550, Panacea(+) 250, Rough Oridecon(+) 100, Silver Ring(+) 10, Glove 1, Mummy Card 1 |
| Golden Thief Bug | Elunium 2000, Oridecon 1500, Gold(+) 1000, Golden Bell(+) 500, Emperium(+) 300, Golden Gear(+) 250, Golden Mace(+) 150, GTB Card 1; MVP: Gold Ring(+) 2000, Ora Ora(+) 1000 |
| Eddga | Honey(+) 10000, Elunium 2300, Katar of Raging Blaze(+) 500, Smoking Pipe(+) 250, Tiger Footskin(+) 250, Fire Brand(+) 150, Krieg(+) 100, Eddga Card 1; MVP: Tiger's Skin 5000, Flame Heart(+) 3000, Tiger Footskin(+) 1000 |

## 3. Engine lines for the owners (not ours; exact replacements)
- **E1 ASPD and attack delay** [LANDED round 3, ro3-engine: `ro_stats.derived` reads the worn weapon class, exports `aspd`; fresh Novice fist 151 = 10 ticks, Thief knife 151 = 10, Thief fist 161 = 8] (`game/ro_stats.py:118-120`, lead): RO pre-re is
  `aspd = 200 - (200 - base) * (1000 - (4*agi + dex)) / 1000` with base = `JOB_WEAPON_ASPD_J[job, weapon_class]`
  (needs the equipped weapon class from ro_items), cap 190, **no 150 floor**; attack delay = 2 x amotion =
  `20 * (200 - aspd)` ms, so `aspd_ticks = max(1, (20 * (200 - aspd) + TICK_MS // 2) // TICK_MS)`.
  Today: `aspd = clip(JOB_ASPD_BASE_J[job] + (ag*4 + dx)//10, 150, 190)` (stat bonus 2x RO, floor 150) and
  `aps = 1500 + (aspd - 150) * 45` (1.5 swings/s at ASPD 150 instead of 1.0; 3.3/s at 190 instead of 5.0).
- **E2 max HP / SP** [LANDED round 3: per-job tables; Novice L1 40 HP / 11 SP] (`game/ro_stats.py:108-109`): `max_hp = JOB_BASE_HP_TABLE_J[job, lvl] * (100 + vi) // 100`,
  `max_sp = JOB_BASE_SP_TABLE_J[job, lvl] * (100 + it) // 100` (and `_fresh_max_sp` at line 64 the same way).
- **E3 HIT / FLEE / MATK / MDEF** [LANDED round 3: hit = lvl + DEX, flee = lvl + AGI, matk = mean(min, max), mdef = INT + VIT // 2; combat uses monster AGI/DEX columns appended to `game/data/monsters.py`; ranged status ATK still open] (`game/ro_stats.py:111-115`): `hit = lvl + dx`, `flee = lvl + ag`,
  `matk_min = it + (it // 7) ** 2`, `matk_max = it + (it // 5) ** 2`, soft `mdef = it + vi // 2`. Then
  `game/combat.py:36-43`: player hit% = `clip(80 + hit - (mon_level + mon_agi), 5, 95)` and monster hit% =
  `clip(80 + (mon_level + mon_dex) - flee, 5, 95)` (needs monster AGI/DEX columns, values in 1.4). Ranged weapons
  (bows) use `dex + (dex // 10) ** 2 + str // 5 + luk // 5` for status ATK.
- **E4 DEF** [LANDED round 3: hard DEF percent then flat VIT (player `def` = VIT, monster `VIT` column); crits ignore both] (`game/combat.py:124-128` and `:98-101`): RO applies hard DEF (armor, percent) then subtracts soft
  (VIT) DEF flat: `dmg = raw * (100 - armor_def) // 100 - vit_def`; today VIT DEF is added into the percent.
  Monster DEF in our table is hard DEF (percent) and is right as used; monster VIT DEF = mob VIT (1.4).
- **E5 after-cast delay for ASPD-based skills** [OPEN: `game/ro_skills.py` is not owned this round; line reported to the lead] (`game/ro_skills.py:174-176`, ro-anim): use
  `where(S.SKILL_DELAY_ASPD[rid] == 1, amo[:mp], S.SKILL_AFTER_DELAY_MS[rid, lvl])`; the table holds 500 ms for those
  rows until then.
- **E6 AoE radius** [LANDED round 3: `R_MAX = 2`] (`game/ro_skills_effects.py:27` `R_MAX = 1`): Magnum Break and Arrow Shower are 5x5 in RO
  (`SKILL_AOE = 2`); R_MAX 2 (25 offsets) makes them hit 5x5.
- **E7 buff and passive effects** [PART round 3: Increase AGI sets `move.speed_ms` 120 in `ro_skills_effects.resolve`; the `aspd_pct` removal and the passives need `game/ro_skills.py`, lines reported to the lead] (`game/ro_skills.py:61-72`): Increase AGI gives AGI +2+lv and **+25 % move speed,
  not ASPD %** (`aspd_pct = agi` is not RO; ASPD rises through AGI only); Increase HP Recovery is
  `5*lv + max_hp*lv/500` HP every 10 s (not +10 %/lv regen); add Improve Dodge `flee_flat = 3*lv`, Double Attack
  `5*lv %` chance of 2 hits with daggers, Vulture's Eye bow range `+lv`, Sword Mastery only with daggers/1h swords,
  Owl's Eye right. Improve Concentration: AGI/DEX +(2+lv) %.
- **E8 Heal** [LANDED round 3: `heal_amount(base_level, int, lv)`; Heal on Undead still open] (`game/ro_skills_effects.py:48-50`): RO heal = `(base_level + INT) // 8 * (4 + 8*lv)`; today
  `(10 + matk) * power // 1000`. Heal on Undead deals holy damage of half the heal.
- **E9 death penalty** [LANDED round 3: base and job exp, Novices exempt, no level floor] (`game/ro_stats.py:209-211`): RO loses 1 % of base **and job** exp to next level, Novices
  exempt (not base level < 10).
- **E10 HP regen** [LANDED round 3 as `ro_stats.hp_regen(world, sitting)` (60 ticks standing / 30 sitting); `game/survival.py` (ro3-wire) wires it] (survival): RO every 6 s standing (3 s sitting): `1 + max_hp // 200 + vit // 5`.
- **E11 persistence** [LANDED round 3: `_skills` zero-pads short arrays] (`net/char_fields.py:170`): `skills_learned` is now length 42 (was 33): pad an older,
  shorter saved array with zeros instead of rejecting it, or old characters lose their skill levels at login.

- **E12 spawn HP (blocks bench/soak_check today)** (`game/config.py:19`, lead): `PLAYER_MAX_HP = 150` is above
  a level-1 Novice's RO max HP (40), so `soak_check` flags "player hp out of range" at tick 0. Minimal fix:
  `PLAYER_MAX_HP = 40` (the interim Novice factor now gives exactly 40 at level 1) and `HUNT_RETREAT_HP = 16`
  (`game/config.py:75`, keeps 40 %). Proper fix: spawn and respawn hp from `ro_stats.derived(world)["max_hp"]` in
  `game/state.py:105`, `bots/runner.py:56`, `net/characters.py:32`, and the bot rest checks that compare with
  PLAYER_MAX_HP (`bots/policies.py:68,108`, `bots/ro_craft_policy.py:59`, `bots/ro_items_policy.py:66`,
  `bots/ro_nav_policy.py:126`, `bots/ro_skills_policy.py:91`).

## 4. Remaining deliberate simplifications
- Element levels 2-4 use the level-1 row (Lunatic, Thief Bug Neutral 3; Hode Earth 2; Mummy Undead 2; GTB Fire 2).
  Negative element entries (heal) are clamped to 0.
- Size, race and MDEF have no damage modifiers yet (weapon size penalty, race cards, Demon Bane).
- Item slots on shop armor kept at 1 (RO shop versions have 0; slotted copies drop from monsters) so cards stay
  usable with the current item set. Rod / Wand MATK is a flat column; RO gives MATK +15 %.
- Potions heal a fixed midpoint instead of `rand(lo, hi)`.
- Bolts, Soul Strike and Fire Wall resolve as one hit of the summed damage; Fire Wall's 1x3 wall and hit count,
  Frost Diver's freeze, Stone Curse, Steal chance, Teleport/Warp and Fly/Butterfly Wing effects are not modeled.
- Job change does not check Basic Skill 9; skill points carry over (RO forces the 9 Novice points into Basic Skill).
- Cast is not interrupted by damage (RO interrupts unless Phen/Endure); contract keeps "no cancel" in v1.

## 5. Corrected feel targets
- **Swing rate** = 50 / (200 - ASPD) swings/s: a fresh Novice with a knife (ASPD 135) swings **0.77/s**, bare
  hands 1.0/s; early first jobs 150-160 = **1.0-1.25/s**; a 99 AGI Thief ~ 185-190 = 3.3-5/s (cap 190).
- **Walk** 150 ms/cell players; monsters 100 (GTB) to 400 ms (Poring, Fabre, Drops, Zombie).
- **Early fights** at RO numbers: Novice L1 has **40 HP / 11 SP**; Poring (50 HP, hits 7-10) dies in 3 knife hits
  (~4 s) and needs 4-6 hits to kill a novice; Orc Warrior (104-126 dmg) one-shots any character below ~level 20:
  zones must band by these levels (world team).
- **Leveling** at 1x rates: base 10 needs 881 exp total (about 440 Porings or 150 Lunatics), job 10 needs 1151 job
  exp. Classic is slow by design; most servers run rates. If the lead wants a faster curve, add an `EXP_RATE`
  multiplier in `game/config.py` (applied in ro_stats kill payout) rather than bending the tables.
- **Casting**: Fire Bolt lv 10 = 7 s cast at 0 DEX, 2.8 s ACD; DEX 150 removes cast time (cast x (1 - DEX/150)).
