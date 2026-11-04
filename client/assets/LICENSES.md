# Asset licenses

All placeholder art is CC0 1.0 (public domain) by Kenney (www.kenney.nl). Only the files used are copied; each
pack dir holds its own `License.txt`. Rebuild the composed sheets with
`python client/assets/build_sheets.py --src <dir with the unzipped packs>`.

| Pack dir | Version / date | URL | License | Files used |
|---|---|---|---|---|
| `client/assets/kenney/tiny-dungeon` | 1.0 | https://kenney.nl/assets/tiny-dungeon | CC0 | `tilemap_packed.png` |
| `client/assets/kenney/tiny-town` | 1.0 | https://kenney.nl/assets/tiny-town | CC0 | `tilemap_packed.png` |
| `client/assets/kenney/roguelike-rpg-pack` | 2020 | https://kenney.nl/assets/roguelike-rpg-pack | CC0 | `roguelikeSheet_transparent.png` (water tile) |
| `client/assets/kenney/ui-pack-rpg-expansion` | 2020 | https://kenney.nl/assets/ui-pack-rpg-expansion | CC0 | panel, button, bar, cursor and icon PNGs listed in `build_sheets.py` |

Derived files in `client/assets/sheets/` (terrain, jobs, monsters) are composed from those packs and are CC0 too.

## Audio (team client-feel, 2026-11-02)

| Dir | Source | License | Files |
|---|---|---|---|
| `client/assets/audio` | generated procedurally for jaxspike by `client/assets/audio/gen_sounds.py` (standard library, fixed seed) | CC0 1.0 (public domain dedication) | `walk.wav`, `attack.wav`, `hit.wav`, `level_up.wav`, `pickup.wav`, `warp.wav` |

Placeholders only: replace with CC0 packs (for example Kenney's "RPG Audio" / "Interface Sounds") and record them here.

## Animated sheets (round 2, `client/assets/anim/`)

Every sheet in `client/assets/anim/<kind>/` (7 jobs, 20 monsters, 2 NPCs `npc_shop` and `npc_waybell`, `act.json` + one PNG per action) is generated
procedurally by `client/tools/gen_placeholder_anim.py` (stdlib only, no source images, nothing photo-derived). They
are project-original and dedicated CC0 1.0. Rebuild: `python client/tools/gen_placeholder_anim.py`. The review
contact sheet lands in `client/assets/anim/_review/` (`.gdignore`, not exported).

CC0 sources evaluated 2026-11-02 and **not used** (none is chibi + 8 directions + idle/walk/attack/hurt/dead):

| Source | License | Why not used |
|---|---|---|
| [Chibi Base](https://opengameart.org/node/119681) (thecilekli) | CC0 | chibi, but one base body with a 4-frame walk, no 8 directions or combat clips |
| [8 Directional Sprite Sets](https://opengameart.org/content/8-directional-sprite-sets) (Technopeasant) | CC0 | 8 directions, walk only, not chibi/anime |
| [8d run/walk & idle](https://opengameart.org/content/8d-runwalk-idle-sprite-sheet) (adhochero) | CC0 | 8x8 px frames, far too small |
| [8 Direction Walk Cycle](https://ordinary-bumblebee.itch.io/8-direction-walk-cycle) (Ordinary Bumblebee, itch.io) | CC0 | 32x48 walk + stand only, not chibi; best fallback base if a hand-drawn body is ever wanted |
| [Hand-Drawn Square Characters 8 Directions](https://rgsdev.itch.io/hand-drawn-square-characters-animated-8-directions-top-down-free-cc0) (itch.io) | CC0 | square doodle style, off the RO anime direction |
| [Adventurer and Slime](https://opengameart.org/node/153748), [Slime 4](https://opengameart.org/content/slime-4) | CC0 | side-view / 3-4 directions only |

No CC-BY-SA or GPL sheets were downloaded. Hand-drawn RO-style art remains the art team's open item.

## Pre-rendered sheets (round 3, team ro3-art, 2026-11-03)

The 7 job and 20 monster sheets in `client/assets/anim/<kind>/` are now rendered from CC0 3D models with the
Blender pipeline in `client/tools/art/` (`render_kind.py`, manifest `kinds.json`, source list `licenses.json`):
cel-shaded 3-step ramp, dark outline, orthographic camera at 35 degrees, 5 directions drawn and 3 mirrored. The
source models stay on `D:/jaxspike-art/src` (not in git); only the rendered PNGs and `act.json` are committed. The
renders are original work derived from CC0 sources and are dedicated CC0 1.0 as well. No RO or other game assets.

| Source | Author | URL | License | Used for |
|---|---|---|---|---|
| KayKit Character Pack: Adventurers 1.0 | Kay Lousberg (www.kaylousberg.com) | https://github.com/KayKit-Game-Assets/KayKit-Character-Pack-Adventures-1.0 | CC0 1.0 | jobs: Knight (swordsman), Mage (mage, acolyte), Rogue (novice, thief), Rogue_Hooded (archer), Barbarian (merchant) |
| KayKit Character Pack: Skeletons 1.0 | Kay Lousberg | https://github.com/KayKit-Game-Assets/KayKit-Character-Pack-Skeletons-1.0 | CC0 1.0 | monsters 12 skeleton (Minion), 15 soldier_skeleton (Warrior), 17 mummy (Mage, tinted) |
| Ultimate Monsters pack (Oct 2022) | Quaternius (quaternius.com) | https://quaternius.com/packs/ultimatemonsters.html | CC0 1.0 (License.txt in the pack) | the other 17 monsters: Blob/PinkBlob (blobbin, emberblob), GreenBlob (vexblob), GreenSpikyBlob (wrigglet), Alien (fiddlehop), Mushnub (capling), Dog (snarlfang), Cat (scorchstripe); Big/Orc (grukk_brute), Bunny (moonbun), Cactoro (bramblet), Tribal (rotling), Birb (gallopeck), Fish (sandgulper), Yeti (grumblebear); Flying/Armabee (pilferbug), Armabee_Evolved (gildscarab) |

Effect sprites `client/assets/anim/fx/` (cast_circle, hit_spark, level_up, heal + `fx.json`) are generated
procedurally by `client/tools/art/gen_effects.py` (Pillow, fixed seeds, no source images): project-original, CC0.

Evaluated and not used this round: Universal LPC Spritesheet Generator (CC-BY-SA 3.0 / GPL 3.0 mixed, 4 directions,
pixel art, no diagonals); Mixamo (needs an Adobe account); VRoid sample models (need a VRoid Hub account).

## Terrain tileset and map props (round 4, team ro4-tiles, `client/assets/tiles/`)

| File | Source | License | Contents |
|---|---|---|---|
| `client/assets/tiles/terrain.png` | generated procedurally by `client/tools/art/tiles/gen_tiles.py` (Pillow, fixed seeds, no source images, no models) | CC0 1.0, project-original | 13 terrain columns x 16 rows of 64 px cells: grass (8 variants), road / temple / water / wall (16-tile 4-neighbour blob sets), floor (4), tree base, rock (3 boulder variants), warp, waybell, station, cave (4), shop |
| `client/assets/tiles/props.png` | same script | CC0 1.0, project-original | trees (3), Waybell bell post, market stalls (4 awning colours), temple fountain (4x4 cells), furnace and cooking fire (for crafting nodes 406/407) |
| `client/assets/tiles/tiles.json` | same script | CC0 1.0 | atlas index read by `client/game/world/map_renderer.gd` |

Every tile is drawn procedurally in the character sheets' cel look (light from the upper left, three flat tones,
white highlight spot, near-black ink outline). No third-party model, texture or game asset is used; nothing is
ripped from RO or any other game. Rebuild: `python client/tools/art/tiles/gen_tiles.py`.

## Card art (round 4, team ro4-cards, `client/assets/cards/`)

| File | Source | License | Contents |
|---|---|---|---|
| `client/assets/cards/<monster>.png` (27 cards, 192x256 RGBA) | composed by `client/tools/cards/compose_cards.py` (Pillow, no fonts, no source images besides the portraits below) | CC0 1.0, project-original | a framed card per card item: gold frame (rose-gold with gems on the two MVP cards), a sunburst window tinted from the monster's own colours, the portrait, a plate with the socket-type gem and rarity stars |
| portraits inside the 20 monster cards | the round-3 chibi renders in `client/assets/anim/<monster>/idle.png` (SW frame) or the art pipeline's `render_kind.py --portrait` close-ups, both made from the CC0 models listed in the round-3 section above | CC0 1.0 | the monster portrait |
| the 7 card-only cards (cocoonling, antlet, buzzeye, peepit, croakle, needlebuzz, cragwing) | same script, a drawn silhouette and question mark | CC0 1.0, project-original | "not found in Aurelune" mystery art |

No text is baked into the cards (the client draws names), so no font licence applies. Nothing is taken from RO or
any other game. Rebuild: `python client/tools/cards/compose_cards.py` (from `client/`).
