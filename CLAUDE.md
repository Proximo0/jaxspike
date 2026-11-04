# Jaxspike: a Ragnarok-Online-style MMO with a JAX rules engine and a Godot client

One repo, two halves. The **engine and server** at the root: a pure-JAX tick (`game/`, fixed-shape world dict, jit +
vmap over worlds, `lax.scan` over ticks), bots (`bots/`), an asyncio WebSocket server (`net/`), benches (`bench/`)
and pytest tests (`tests/`). The **client** in `client/`: a Godot 4.7.2 typed-GDScript project started from the
game starter kit (its own `client/CLAUDE.md` holds the Godot rules and commands). Vision, scope, feel targets and
milestones: `docs/production-handover.md` and project memory. Module contracts: `docs/brief.md`.

## The one rule that orders everything
**Anything that can be made to work in JAX goes into the JAX tick first, for speed** (the user's rule). Stats,
skills, buffs, items, cards, crafting, parties, MVPs, viewport culling and delta building are fixed-shape array ops
inside the tick. Python keeps only irregular-shaped work (sockets, login, chat text, a human's click-to-walk A*).
Godot only draws, takes input and shows panels. Every contract brief restates this.

## Commands (engine)
- Tests: `venv\Scripts\python.exe -m pytest tests -q` (Windows venv, CPU JAX). Run after every engine change.
- Soak: `venv\Scripts\python.exe bench/soak_check.py` (invariants over bot-played worlds); the biggest batch the cap
  allows after changes to movement, respawn or occupancy.
- GPU runs: only in WSL with `~/jaxgpu/bin/python` and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6` (60% cap, never above: the user runs
  their own things beside Claude's).
- Server: `venv\Scripts\python.exe -m net.server` (add `--cheats` for playtest cheats: F5 panel in the client);
  load bench: `bench/server_load.py --embedded --clients 200`.

## Commands (client, from `client/`)
- `tools/check_all.sh` (unit, sims, lint, bot and lab when they exist), ends `CHECK_ALL PASS`.
- On this Windows PC run the tools from Git Bash with
  `GODOT=/c/Users/crp27/Downloads/Godot_4.7.2/Godot_v4.7.2-stable_win64_console.exe PATH="$PWD/tools/winshim:$PATH"`
  (the shim supplies the missing `flock`). CI and cloud sessions run the same scripts on Linux.

## Load limits (the PC has 16 GB RAM; three Claude crashes on 2026-11-02 came from running out of it)
- **Claude's work may use at most 60% of the machine** (the user's rule: they run their own things beside it):
  RAM, GPU memory (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`) and cores (`JOBS=14` at most for client tools).
- At most **60 agents in all**: at most 6 team sessions at once, each with at most 10 workers
  (`.claude/settings.json` caps subagents per session at 10). More teams wait in a queue; the lead starts the next
  when one reports.
- One heavy process per team at a time (pytest, soak, bench, Godot, Blender, test server). Iterate on the test
  files you changed; the full suite, `check_all`, soaks and benches are the lead's job at integration, run one at
  a time on a quiet box. GPU soaks only in WSL under the 60% cap. Soak batches of at most 4 worlds on the PC.
- `.claude/hooks/guard_heavy.py` runs before every Bash call and blocks a heavy command when Claude's heavy
  processes (python, Godot, Blender) would pass 60% of RAM with it, when under 25% of RAM (4 GB) is free for the
  user's own programs, when 6+ python, 2+ Godot or 2+ Blender processes exist, or when a soak asks for more than 4
  worlds. A block is not a retry prompt: wait for your own process, or do less.
- After a crash, the user runs `tools\kill_orphans.ps1` (add `-KeepServers` to spare a playtest server); the lead
  never leaves a server or soak running past its use.
- Shared-checkout rounds: teams re-read a shared file right before each edit and never run git.

## Rules
- Pure JAX in `game/` and `bots/`: no Python `if` on traced values, no loops over entities, masks over fixed shapes,
  randomness only via `next_key`, constants only in `game/config.py`, coordinates only in `game/world_layout.py`.
- The lead writes contracts first (`docs/<step>_brief.md`), workers own disjoint files, only the lead runs git
  (stage by name, gate on summary lines `passed=N failed=0`, `CHECK_ALL PASS`).
- Workers are Sonnet by default; Haiku for data and docs; Opus only for the hardest core logic.
- Reproduce before fixing (test, soak, bot, replay, screenshot pixels). Add a row to `client/docs/lessons_learned.md`
  whenever something goes wrong.
- The lead cannot see the game window: say so and ask the user to check visuals.
- No secrets in git (`config/*credentials*` is ignored). Player saves in `characters/` are ignored.
