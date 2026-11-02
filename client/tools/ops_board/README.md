# Ops board: refresh and setup

A Dashboard artifact that shows a game project's work, quality, performance, balance, assets and goals.
The page (`index.html`) holds no project names or numbers. Everything comes from datasets, including a
small `project` dataset with the names, workflow names, tab labels and targets. `refresh.py` rebuilds
the cloud-side datasets. It needs Python 3 (standard library only), git and the `gh` CLI (REST calls
only, no GraphQL).

Files: `refresh.py`, `index.html`, `schema.md` (every dataset's fields), `datasets.json` (dataset
titles and descriptions), `config.example.json`, `workers_src.example.json`,
`needs_owner_src.example.json`.

## Refresh (each time)

1. Edit `workers_src.json` and `needs_owner_src.json` (next to the config) if workers or asks changed.
2. Run:
   ```
   python3 refresh.py --repo <clone> --config <board>.config.json
   ```
   It prints `CHANGED <id> <file>` for each file whose content changed, then
   `REFRESH changed=N problems=M main=<sha> at=<time>`. When a collector fails, its last good file is
   kept and the failure is printed as a `PROBLEM` line. The run also writes `changed.json`.
3. Upload each changed file with the Artifact tool, one call per JSON file:
   `{action: "publish", url: <board>, asset: true, file_path: <out>/<id>.json}`. Each call returns a
   new `/_blob/<id>` url.
4. Build the store update:
   ```
   python3 refresh.py --config <board>.config.json --updates commits=/_blob/aaa runs=/_blob/bbb ...
   ```
   This writes `<out>/updates/<id>.json` (`source` with the new url, plus `updated: {at, by}`) and
   prints the batch `writes`.
5. List the board's `datasets` collection (ArtifactData `list`) to get each document's `version`. Put
   that version in place of `"<version>"` and send all of them as one ArtifactData `batch`. Only the
   changed datasets' `source` and `updated` fields change. Titles and descriptions stay as they are.
6. Read `data/users/me` doc `check` (as the type's SKILL.md says). If there is no document, nobody has
   the board open. It checks itself the next time it is opened.

The page itself only changes when `index.html` changes. To update it, `set` `files/index.html` to
`{text: <whole file>}` with its version.

## Setup for a new project (once)

1. Copy `tools/ops_board/` into the repo, or keep it beside the lead's scratchpad.
2. Write `<project>.config.json` from `config.example.json`:
   - `project`: name, board_title, owner, checks_workflow, plus the optional fields in `schema.md`.
     Leave a field out to hide what it drives: no `perf_workflow` hides the perf runner tile, no
     `deadline_utc` hides the countdown, no `balance_matrix_path` hides the matrix card, and `tabs`
     lists only the tabs you want.
   - `repo_slug` is optional (it is read from `git remote get-url origin`). `main_branch` defaults to
     `main`.
   - The defaults fit the starter-kit layout: `tests/unit/test_*.gd` with `func test_`,
     `tests/sim/sim_suite.gd` `SIM_FILES`, `.gd` lint, and the Actions `checks` workflow with jobs
     `unit`, `sims`, `lint` and `bot-*`. To change the defaults, override `tests` and `check_jobs`.
   - Optional collectors: `goals` (a markdown goals table), `balance.studies`, and `extra_files`
     (any CSV or JSON in the repo, such as a balance matrix or bot stats). Each one no-ops when its
     source is missing. With `empty_if_missing`, it writes `[]` instead, so the card shows its
     placeholder.
3. Copy `workers_src.example.json` and `needs_owner_src.example.json` to `workers_src.json` and
   `needs_owner_src.json`.
4. Create the board. Run Artifact `quickstart` with intent `other`, then publish from the Dashboard
   type's `type_url` with a `title`. Write `dash/meta` `{title}`.
5. Run refresh, upload every file in `<out>/`, then run
   `python3 refresh.py --config <cfg> --updates id=/_blob/... ... --create`. That writes whole new
   dataset documents, with titles and descriptions from `datasets.json`. Send them, plus
   `files/index.html` `{text: <index.html>}`, in one ArtifactData `batch` (no `if_version` on new
   documents).
6. Optional perf datasets (`perf`, `soak`, `scripts`, `build_parts`, `match_start`, `hitch_frames`,
   `runner_perf`, `runner_jobs`, `models`, `jobs`) come from a PC session or the GPU runner. Upload
   them with the fields in `schema.md`, or copy them from another board's assets
   (`{asset: true, from_url, asset_ids}`). Then add a dataset document for each. Until they exist,
   the Performance and Assets cards stay hidden.
7. Read `data/users/me` doc `check` after the first write. Fix what it reports, at most twice.
