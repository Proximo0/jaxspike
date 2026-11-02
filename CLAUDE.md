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
  `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` (80% cap, never above).
- Server: `venv\Scripts\python.exe -m net.server`; load bench: `bench/server_load.py --embedded --clients 200`.

## Commands (client, from `client/`)
- `tools/check_all.sh` (unit, sims, lint, bot and lab when they exist), ends `CHECK_ALL PASS`.
- On this Windows PC run the tools from Git Bash with
  `GODOT=/c/Users/crp27/Downloads/Godot_4.7.2/Godot_v4.7.2-stable_win64_console.exe PATH="$PWD/tools/winshim:$PATH"`
  (the shim supplies the missing `flock`). CI and cloud sessions run the same scripts on Linux.

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
