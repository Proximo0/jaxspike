# World build scale: measured at MAP 512, estimated at 2048

Bench: `bench/world_scale.py` (CPU, JAX, seed 0, batch 1 and 4). MAP is a config constant, so 2048 numbers are
**estimates by scaling**, not runs. `map_gen.distance_fields_ro` did not exist when this was measured; rerun
the bench once it lands (the script times it automatically). Expect it to cost about the same as
`distance_fields`, plus extra iterations for the warp hops (dungeon tiles are reached through the exit warp).

## Measured (MAP 512, 5 towns, CPU)
| item | first call (with compile) | second call |
|---|---|---|
| `make_terrain` (jit) | 1.0 s | 0.01 s |
| `distance_fields` (jit, one terrain) | 11.5 s | 11.2 s |
| `make_world_batch(0, 1)` | 21.0 s per world | 13.9 s per world |
| `make_world_batch(0, 4)` | 63.6 s (15.9 s per world) | 57.4 s (14.3 s per world) |

- Relaxation iterations: max finite distance 815, so **816 iterations** (about 13.7 ms per iteration over
  5 x 512 x 512 int32).
- The fields are 80 % of a world build (11.2 of about 14 s); terrain and spawn are noise. Compile adds 1-7 s once.
- `lax.map` over worlds is linear: batch 4 costs 4x batch 1.

| array per world | MB |
|---|---|
| `dist_town` int16 [5, 512, 512] | 2.50 |
| `terrain` int8 | 0.25 |
| `walls` int8 | 0.25 |
| `pz` bool | 0.25 |
| whole world pytree (players, monsters, the above) | 3.46 |
| jax live buffers after build | 6.0 MB at batch 1, 17.3 MB at batch 4 (3.46 MB per world held plus the keys and inputs/outputs still alive) |

Players and monsters are tiny next to the maps: `dist_town` is 72 % of the world.

## Estimate at MAP 2048
Area x16, relaxation iterations x4 (path lengths scale with the side), so the relaxation is about x64 and
everything else x16.

| item | 512 (measured) | 2048 (estimate) |
|---|---|---|
| `distance_fields` CPU | 11.2 s | about 720 s (11.2 x 64) |
| `make_world_batch` per world CPU | 14 s | about 12 min |
| iterations | 816 | about 3300 |
| `dist_town` | 2.5 MB | 42 MB (5 x 2048 x 2048 x 2 B) |
| terrain + walls + pz | 0.75 MB | 12 MB (4 + 4 + 4) |
| world pytree | 3.5 MB | about 54 MB plus a few MB of entities |
| transient inside `distance_fields` (about 6 live int32 [5, MAP, MAP] copies: d, nd, padded, four shifted views fused) | about 50 MB | about 500 MB, once, because `lax.map` runs worlds one at a time |

GPU time is not measured (CPU only here). The CPU relaxation is memory-bound, and a GPU does a 5 x 2048 x
2048 stencil pass in well under a millisecond, so 3300 passes should be seconds, not minutes; but verify in WSL.

## GPU batch ceiling with the 6.5 GB cap (80 % of 8 GB)
Reference points from `docs/handoff.md`: at MAP 512 the verified ceiling is **256 worlds** (512 fails on a
1.27 GiB `reduce_min` over `dist_town`). 6.5 GB / 256 = about 25 MB per world, i.e. about 7x the 3.5 MB of
stored state: the rest is batched `MAP x MAP` temporaries (`monsters._chase` claim table,
`combat._respawn_players` occupancy, the spawn reduce).

| MAP | storage only: 6.5 GB / MB per world | with today's per-world MAP x MAP temporaries (about 7x storage, scaled from the 256 ceiling) |
|---|---|---|
| 512 | about 1,900 worlds | 256 (verified) |
| 2048 | 6,500 / 54 = about 120 worlds | about 16 (256 / 16 area; 6,500 / (7 x 54) = 17) |

So the temporaries, not the stored fields, decide the ceiling. Moving the claim table and occupancy to the
sort-based approach and spawning with `lax.map` takes 2048 from about 16 toward about 100 worlds; storage then
bounds it, and `dist_town` (42 of 54 MB) is the item to shrink (plan items 3 and 4 below).
A live server needs one world, which is 54 MB: no problem at any plan item.

## Plan for 2048, ranked
1. **Compute the fields per town with `lax.map` over the town axis, inside the per-world `lax.map`.** Peak
   temporaries drop from NT x MAP x MAP int32 (about 500 MB estimated) to 1 x MAP x MAP int32 (about 100 MB),
   a 5x cut, and the result is identical (towns are independent). Cost: five sequential loops instead of one,
   which is free on CPU (memory-bound, same total work) and slightly slower on GPU for lack of parallel width.
2. Keep int16 storage but relax in int32 chunks (cast tiles of rows in and out): saves the int32 copy of the whole
   field, about 2x on the temporaries; smaller than (1) and stacks with it. Note int16 holds 32767, enough for
   the 3300 iterations here.
3. Cache fields per terrain seed on disk (`numpy.savez`, 42 MB per world, a few MB compressed). Removes the
   estimated 720 s CPU build (about 12 min) per world entirely for repeated seeds. The fields depend only on the
   terrain, so the cache key is the seed (plus a layout version hash).
4. Coarse 4 x 4-cell field for far steering plus the fine field only in a band around towns. 16x smaller storage
   (42 MB to about 2.6 MB per world) and 4x fewer iterations, but it changes steering and spawn rules (spawn bands
   read the fine field), so it needs bot and spawn re-tuning.

**First: item 1.** It is a few lines in `make_world_batch`/`distance_fields` (map over `gate_aprons` rows), it
cuts the transient that will not fit next to a 42 MB field on the cap by 5x (500 MB to 100 MB), keeps results
bit-identical so the existing tests prove it, and it is a prerequisite for the others rather than a
rival. Item 3 saves more wall time (12 min per world) and should come second, because it only helps repeated seeds.

## Caveats and two more items (sub-orchestrator, after review)
- The numbers above were measured while four workers shared the 24 cores; `docs/handoff.md` reports about
  1 s per world for `distance_fields` on an idle CPU. Treat 11 s as an upper bound and rerun
  `bench/world_scale.py` on an idle machine before quoting it. The ratios (fields = 80 % of a build,
  iterations = max distance + 1) do not depend on load.
- `distance_fields_ro` (warp-aware) now exists: it adds one `[NT, N_WARPS]` gather and scatter-min per
  iteration, which is noise next to the stencil; dungeon tiles converge within the same iteration budget
  because they are a short walk from an exit warp.
- **Int32 claim keys overflow at 2048.** `movement.move_players` and `ro_warps.apply_warps` build
  `tile * MAX_PLAYERS + slot` with `tile < MAP * MAP`: 4,194,304 x 512 = 2.1e9 > 2^31 - 1. At 2048 these
  keys must become int64 (`jax_enable_x64`) or be split (sort by tile, then by slot within equal tiles, or use
  `tile * MAX_PLAYERS` in uint32 where 2048^2 x 512 = 2^31 fits exactly below 2^32). Lead-owned fix in
  `movement.py`; `ro_warps.py` follows the same change. This is a blocker before any 2048 soak.
5. **Creation-time BFS outside the tick.** World creation is not the tick: a multi-source numpy BFS
   (`collections.deque`, 5 towns x 4M tiles) runs in about 20-40 s per world on one core and needs no
   3300-iteration relaxation. It keeps the stored field identical (`tests/test_map.py` already compares the
   JAX field to this BFS). Worth it only if GPU relaxation turns out slower than expected; the JAX path stays
   the default per the "JAX first" rule.
