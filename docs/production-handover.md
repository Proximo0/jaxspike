# Production hand-over: from the Tibia prototype to a Ragnarok-style MMO (JAX core + Godot client)

Written 2026-11-02 by the outgoing lead for the incoming production lead. Read this first, then `docs/handoff.md` (engine state, rules, lessons in detail), then `docs/brief.md` (module contracts).

## 1. The decision (user, 2026-11-02)
- The prototype is done and the user is moving the game to **full production**.
- **Reference game changes from Tibia to Ragnarok Online** (classic RO: jobs/classes with job change, STR/AGI/VIT/INT/DEX/LUK stats, skill trees with skill points, ASPD/cast times, cards slotted into gear, refining, MVP bosses, zeny, warps between maps, parties/guilds, kafra storage, free-to-play feel). Keep what the user already said about scope: fun, not over-complicated; RuneScape-like gathering/crafting is still wanted; "great GUI, great 2D graphics"; public server, hundreds of players.
- **Architecture: push everything you can into the JAX core; run the rest in Godot.** The user's own words. The JAX tick stays the authoritative rules engine and the bot soak. Godot (4.x, GDScript) is the client: rendering, GUI (RO-style panels, inventory, equipment, skill bar, chat), input, audio, and the editor for maps. The Python layer between them stays as thin as possible (WebSocket, login, persistence, click-to-walk A*, anything with irregular shape such as chat and quests).

## 2. What exists (local git repo, branch `main`, no remote, head abd12ec)
- `game/`: pure-JAX rules engine — fixed-shape world dict, phases movement → monster AI → combat → loot → survival, jit + vmap over worlds, `lax.scan` over ticks. 512x512 map with 5 walled towns, temples, gated roads, lakes, forests, all in map fractions (`game/world_layout.py`); `MAP = 2048` is a one-constant change.
- `bots/`: stateless bot policy that leaves town, hunts, retreats, rests; batch runner.
- `net/`: asyncio WebSocket server (one world, 120 ms tick that never awaits a client, bots in every slot without a human, periodic saves in a thread), JSON protocol v1 with viewport snapshots, name-only persistent characters (atomic npz), bounded A* for click-to-walk, threaded client. `play.py` is a tkinter network client (the user has played the single-process version and called the world solid; the network version was never seen on screen).
- `bench/`: invariant soak (`soak_check.py`), step timing, load bench (`server_load.py`: 200 clients, 0 overruns).
- 147 tests. GPU soaks (WSL, 80% cap) clean up to 256 worlds at MAP 512.
- Numbers today: 3.3 kills / 0.45 deaths per bot per 1000 ticks, MONSTER_DAMAGE (3, 9).

## 3. Concepts proven for "keep it in JAX" (reuse for RO)
- **Distance fields instead of pathfinding**: one `[MAP, MAP]` int16 field per destination (gates, warps, kafra, mines); "go to X" = step to the smallest neighbour. Computed once per world by relaxation in a `while_loop`.
- **Tables indexed by id** for items, monster kinds, recipes, skills, cards: `MONSTER_HP[kind]`, `SKILL_COST[skill]`, `CARD_EFFECT[card]`.
- **Per-entity counters** for every timer: attack cooldown, respawn, cast time, buff duration, skill cooldowns.
- **Stateless policies**: modes derived from the world each tick (bots, monster AI); hysteresis via position/HP, not stored state.
- **Conflicts by scatter-min**: lowest slot wins a tile, a drop, an ore node.
- **Masks over fixed shapes**: inactive slots ignored by `active`; never Python loops over entities; one host sync per tick.
- **The hybrid exception**: a human's click-to-walk is server-side A* feeding one move per tick, so the engine never needs a general pathfinder.
- For RO specifically, the natural JAX homes are: stats and derived values (ATK/MATK/HIT/FLEE/ASPD as vectorised formulas), skill effects as masked array ops with cast/cooldown counters, buffs as `[MAX_PLAYERS, N_BUFFS]` duration arrays, cards as per-slot effect ids summed into stat modifiers, element tables `[ELEM, ELEM]` damage multipliers, exp/job-exp arrays with level from a lookup, drops from per-kind tables, MVP as a monster kind with a spawn timer. Warps between maps: either one big map with warp tiles (teleport = position set) or a `map_id` per entity with per-map arrays; one big map keeps a single scan.

## 4. Lessons from this build (full list in `docs/handoff.md`; the ones that cost the most)
1. Invariant soak finds integration bugs unit tests miss; run it after every change, biggest batch the GPU cap allows.
2. Avoid per-entity tables under vmap (memory cliffs); use argsort/cumsum/scatter-min.
3. Spawns from one permutation; dead players are non-physical; ties deadlock (the gate jam: first-index tie-breaks made two bots wait on each other's tile — exclude tiles an entity should never want, and print where entities ARE after a soak, not just kill counts).
4. Never gather `field[home]` per player (268 MB per world); index `field[home, x, y]`.
5. Cap GPU memory on every run (`XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`); the machine nearly crashed uncapped. 24 cores, 8 GB shared GPU; back off agents when load climbs.
6. Balance from the soak with one group of numbers at a time; tests encode numbers, so read constants from config.
7. Worker briefs with file ownership and a shared contract written by the lead first; subagent-driven development with per-task review caught real bugs (save ordering race, lost taps, terrain-bearing snapshots overwritten, out-of-map click disconnecting a player).
8. The lead cannot see the window: say so; the user checks visuals.
9. Terrain as JSON per step dominates bandwidth (38–49 KB/client/s): **protocol v2 — send the map once at login (zlib), only entity deltas per tick — before any public client**. Godot makes this natural (the map lives in the client).
10. Writing-plans/brainstorming skills: one question at a time, spec → plan → subagent execution worked well; keep rulings in a ledger that survives compaction.

## 5. Lessons from the "Programming language efficiency and preferences" thread (session local_766a1fd2)
- Claude is most efficient in Python, then TypeScript; typed languages shorten the loop because the compiler is the reviewer.
- **GDScript is the language Claude hallucinates in most** (small corpus, Godot 3→4 API churn): keep the Godot docs open, keep the runtime error output in the loop, and use the headless render checks from the starter kit on every change. Typed GDScript (`var x: int`) helps. C# in Godot has no web export (as of mid-2026), so GDScript was chosen.
- Four-way comparison outcome for this project: JAX core beats every engine on the server half (thousands of bots per GPU, deterministic replays, 200 live clients with 0 overruns) and loses on the client half; **the recommendation that survived the full circle was JAX core + Python server + Godot GDScript client**. Phaser/TypeScript only if a browser link becomes a goal (the user explicitly does not need a web client). Godot's Control nodes fit MMO panels; Godot exports desktop/mobile/web from one project; consoles via W4 Games (paid, NDA devkits) — Phaser effectively cannot.
- Unity rejected (editor-centric, licensing); Unreal worst fit for a text agent.
- Godot precedent: hits are premium Steam games by 1–2 person teams; no shipped Godot game runs a persistent hundreds-of-players world — which this design sidesteps because the world lives in JAX and Godot only draws it. Account/payment layer would be built on the Python side.

## 6. The starter-kit pipeline (use the `anthropic-skills:game-project-kickoff` skill; it is the source of truth)
- Kit repo: https://github.com/Proximo0/game-starter-kit — project template, scaffolder `tools/new_project.sh <dest> "<Name>"`, CI, session hook, and docs: `docs/playbook.md`, `docs/lessons_learned.md`, `docs/lessons/skate_park.md`, `docs/lessons/sandblood.md`, `docs/production_pipeline.md`, `docs/agent_brief.md`, `docs/new_project_checklist.md`, `docs/performance.md`, `docs/online.md`, `docs/android.md`, `docs/agent_dashboard.md`. Build guides: Skate Park https://claude.ai/artifact/WtmrkVRtizP8oJBRwhxdWW, Sandblood https://claude.ai/artifact/LBNp22258n18mRFtk2ke4i.
- Steps the skill prescribes: read the playbook; agree vision + 3–6 measurable feel targets with the user in one round and write them to memory; scaffold; **debug layer before content** (live-tunable Tuning class + panel, per-tick debug frame with snapshot/restore, flight recorder with replay, overlay, render toggles, a lab measuring the feel targets, then a play-bot); plan in contracts (shared API, data formats, input keys, file ownership) and fan out (`anthropic-skills:agent-fanout-playbook`); memory + build guide + ops dashboard; use the user's PC for GPU/Houdini/self-hosted runner; ship a playable build each milestone; harvest lessons into the kit.
- Rules from the kit that prevented past mistakes: reproduce before fixing; check lab warnings against the user's design wishes; only the lead runs git, staging by name; gate on summary lines not exit codes; time real frames p50/p95/max after warm-up; ~8 local workers on a 4-core box, more via cloud agents and the Actions matrix.
- **Conflict to resolve with the user on day one:** the kit wants a public GitHub repo (free Actions minutes) and uses cloud agents; this project so far has run local-only ("GitHub minutes unavailable", no cloud sessions). Ask which applies now before scaffolding.

## 7. Suggested first moves for the production lead
1. Kickoff skill: vision round with the user (RO-style class list for v1, number of maps, feel targets such as tick latency, frames, time-to-first-kill, ASPD feel).
2. Decide repo shape: Godot project scaffolded from the kit, with the existing `jaxspike` engine as the server package (same repo or two). The server contract that Godot must speak is in `docs/brief.md` "Networking"; design protocol v2 (map once at login + entity deltas) as part of the Godot client spec.
3. Debug layer in Godot first (per the kit), then a thin client that logs in and draws the viewport, before any RO content.
4. Then RO systems in JAX in milestone order: stats/jobs/exp → skills/buffs/elements → items/cards/refine → gathering/crafting → MVPs/parties.
5. Keep: Sonnet workers with file ownership, SDD per-task review, soak after every engine change, GPU cap 80%, the lead says what it has not seen.
