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

Evaluated and not used this round: Universal LPC Spritesheet Generator (CC-BY-SA 3.0 / GPL 3.0 mixed, 4 directions,
pixel art, no diagonals); Mixamo (needs an Adobe account); VRoid sample models (need a VRoid Hub account).
