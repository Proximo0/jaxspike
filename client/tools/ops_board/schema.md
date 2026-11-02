# Ops board dataset schema

Each dataset is a JSON array of row objects, uploaded as a file and named by `datasets/<id>` in the
board's store. **Core** datasets come from `refresh.py` for every repo. **Optional** ones come from an
optional collector, or from a PC or perf-runner session. The page hides a card whose dataset is
missing or empty. Times are ISO 8601 UTC (`2026-10-22T18:25:48Z`).

The board reads an empty file (`[]`) as a failed source. So an empty result is written as one marker
row, `[{"_empty": "<why>"}]`. The page and its calculations skip marker rows.

## Core (refresh.py)

### project: one row, from the config's `project` object
| field | type | meaning |
| --- | --- | --- |
| name, board_title | text | project name; page title |
| owner, pc_label | text | person the "Needs" card names; the PC session's label |
| repo, repo_url, main_branch | text | GitHub slug and URL (derived from the clone if left out), main branch |
| checks_workflow, perf_workflow | text | Actions workflow names for tests and for real-GPU perf (perf optional) |
| tabs | text | `id:Label;id:Label`. Ids: overview, work, quality, performance, balance, assets, goals. A tab not listed is hidden. |
| deadline_utc, deadline_label | text | countdown tile; leave out to hide it |
| match_word | text | what one round is called ("fight", "match", "race") |
| fps_target, freeze_target_ms, hitch_ms | number | frame budget line, match-start freeze target, hitch threshold |
| perf_avg_measure | text | the `perf.measure` value that is the average frame |
| perf_scenario, perf_scenario_short, perf_machine, soak_label | text | perf card wording |
| asset_budget_label, assets_label | text | asset card wording |
| balance_matrix_path | text | matrix CSV path; leave out to hide the matrix card |

### workers: copied from `workers_src.json`
| field | type | meaning |
| --- | --- | --- |
| area | text (key) | what the worker builds |
| where | text | e.g. Shared cloud box, Cloud thread, Owner's PC |
| model | text | model name, or "Not recorded" |
| status | text | Running, Finished or Stopped (paused, work kept) |
| branch | text | branch or checkout |
| session | text | optional: the thread's cloud session id (`session_…`), read by the cost routine |

### needs_owner: copied from `needs_owner_src.json`
| field | type | meaning |
| --- | --- | --- |
| item | text (key) | what is needed |
| kind | text | Blocking or Optional |
| detail | text | why |

### commits: main over `commit_days`
| field | type | meaning |
| --- | --- | --- |
| sha | text (key) | short hash |
| time, day, hour | text | commit time; its UTC day and hour |
| message | text | first line |

### branches: every pushed branch plus local-only ones
| field | type | meaning |
| --- | --- | --- |
| branch | text (key) | name |
| where | text | Pushed or Local only |
| ahead, behind | number | commits relative to main |
| last_commit, last_message | text | newest commit on the branch |

### runs: last 100 Actions runs
| field | type | meaning |
| --- | --- | --- |
| run_id | text (key) | Actions run id |
| workflow, branch, sha, event | text | |
| status, result | text | status; conclusion (success, failure, cancelled) or the status while running |
| started | text | run start |
| duration_s | number or null | start to finish, null while running |

### checks: newest finished `checks_workflow` run on main
| field | type | meaning |
| --- | --- | --- |
| role | text (key) | unit, sims, lint or bots (from `check_jobs`) |
| check, count_what | text | display label; what `count` counts |
| result | text | success if every matching job passed |
| count | number | test functions, sim files, scripts, or job count |
| sha, run_id, run_at | text | the run |

### assets_glb: asset files on main
| field | type | meaning |
| --- | --- | --- |
| folder | text (key) | first folder under `assets.dir` |
| files | number | files with `assets.ext` |

### goals: the goals table in `goals.md` (under `goals.heading`)
| field | type | meaning |
| --- | --- | --- |
| target | text (key) | |
| measured, status, source | text | as written; rows in `goals.live` are rewritten from workflow runs |
| status_class | text | Met, Missed, Unmeasured, No target, Weak or Other |
| n | number | row order |

## Optional (collectors that no-op when their source is missing)

### balance: `balance.studies` in the config
| field | type | meaning |
| --- | --- | --- |
| study, study_title, study_note | text | one card per study |
| matchup | text | row label |
| before, after | number 0-1 | win rates |
| before_wins, after_wins, fights | number | counts (markdown studies) |
| band_lo, band_hi, ref, ref_label | number/text | shaded band and dashed line |
| read_from | text | the file the rows came from |

A markdown source needs a table whose rows start with `vs `. The first number column is "before".
The column whose header says "kept" is "after" (else the last column). "of N" in the first header gives
the fight count.

### extra_files: any CSV/TSV/JSON in the repo, one dataset each
`{dataset, path, format?, where: "main"|"local", empty_if_missing?}`. Rows are copied as they are. Use
this for `balance_matrix` (string row and column labels plus a numeric rate) and bot fight stats.

## Optional (PC or perf-runner session uploads)
| dataset | fields |
| --- | --- |
| perf | tier, measure, ms |
| soak | fight, seed, game_seconds, fps, p95_frame_ms, worst_frame_ms, static_mem_mb, video_mem_mb, nodes, objects |
| scripts | script, kind (frame or physics), ms |
| build_parts | part, ms, nodes, fix |
| match_start | fight, build_ms, first_frame_ms |
| hitch_frames | fight, frame, ms, cause (key: fight,frame) |
| runner_perf | run, tier, old_counter_ms, physics_ms, effects_ms, draw_calls, triangles, video_mem_mb |
| runner_jobs | started (key), result, failed_step, cause |
| models | model, group, triangles, size_kb, budget |
| jobs | job (key), status, note |

## Optional (cost routine upload)
Written by `thread_costs.py` from the cloud session records of the project's threads (the daily cost routine; see
docs/agent_dashboard.md). Thread ids come from the `session` field of `workers_src.json` rows.

### threads: one row per thread session
| field | type | meaning |
| --- | --- | --- |
| id | text (key) | session id |
| title, role, model, effort | text | thread title; lead or worker; last served model; effort level |
| status | text | running, blocked, review, done, failed |
| created, updated | text | session start; last activity |
| minutes | number | created to updated |
| cost_usd, cost_per_hour | number | session total; per wall-clock hour |
| input_tokens, cache_read_tokens, cache_write_tokens, output_tokens | number | session totals |
| context_used | number | context tokens at the last turn |

### threads_total: one row
| field | type | meaning |
| --- | --- | --- |
| threads, running | number | rows; rows running |
| cost_usd, minutes, tokens, output_tokens | number | sums |
| cache_read_share | number | cache reads / all tokens (0-1) |
