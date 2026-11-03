# Ops board

Board: **https://claude.ai/artifact/CYTg8hXicgNUYbM55yrLvo** (a claude.ai Dashboard artifact, private until shared
from its Share menu). It shows `main`, the `ro/*` branches and, since round 2, one card per team in
`docs/round2_contract.md` section 2 (round 2 has no branches), from primary sources only: git, file times, pytest
collection, the GitHub Actions API and CI job logs. No session reports its own numbers (see `docs/lessons_learned.md`,
"Dashboards").

## Refresh

From the repo root, on the lead's PC (needs the venv Python, `git` and a logged-in `gh`):

```
venv\Scripts\python.exe tools/ops/refresh_ops.py
```

It fetches origin, collects everything below in about 20 s, writes `tools/ops/ops_data.json` (all datasets in one
file) and `tools/ops/out/<dataset>.json` (one file per dataset, the upload format), and prints `PROBLEM` lines
followed by one summary line:

```
OPS_REFRESH teams=12 teams_active=10 teams_stale=2 tests_collected=74 branches=17 green=0 red=1 stale=11 main_ci=failure engine_tests=490 client_tests=133 worktrees=10 problems=0 at=2026-11-02T22:57:58Z
```

It takes about 80 s now: the team view runs `pytest --collect-only` on each team's test files (`--no-collect` skips it).

Options: `--baseline <commit>` (default `6451f3e`, the round-2 contract commit the team view diffs against),
`--no-collect`, `--repo <path>` (default `.`), `--out <dir>` (default `tools/ops`), `--pytest-log <file>` (default
`tools/ops/logs/pytest_latest.log`; to feed it, run
`venv\Scripts\python.exe -m pytest tests -q 2>&1 | tee tools/ops/logs/pytest_latest.log`), `--no-fetch`,
`--stale-hours` (default 2).

**Republish step** (the session that owns the board, after every refresh): upload each changed
`tools/ops/out/<id>.json` as an asset of the board (Artifact tool, `{action: "publish", url: <board>, asset: true,
file_path: tools/ops/out/<id>.json}`, one call per JSON file), then one ArtifactData `batch` that `update`s each
`datasets/<id>` document's `source.url` to the new `/_blob/<id>` url and `updated: {at, by}` to the refresh time,
pinned with each document's current `version` (ArtifactData `list` on `datasets`). The page itself
(`tools/ops/index.html`) only needs rewriting when it changes: `set` `files/index.html` to `{text: <whole file>}`.
A refresh without the republish step changes nothing on the board.

## What the board shows

**Header.** When the snapshot was taken and how old it is, and the `origin/main` sha. A warning appears when the
snapshot is more than 2 h old, and a red line when the last refresh had collector problems.

**KPI row** (dataset `kpi`, a calc over `branches`, `ci_runs` and `tests`):
- *Branches green / red*: branches whose latest `checks` workflow run concluded success / failure. A branch that
  was never pushed has no run and counts in neither.
- *Branches stale*: branches whose last commit is older than 2 h.
- *Main CI*: conclusion of the latest `checks` run on `main` (or its status while running).
- *Engine tests*: tests passed, parsed from the `engine-pytest` job log of the latest finished `checks` run on
  `main` (the `N passed` pytest summary line).
- *Client tests*: tests passed, parsed from the `client-unit` job log of the same run (`TESTS passed=N failed=M`).

**Teams (round 2)** (dataset `teams`, one row per team row of the contract table, refreshed with the contract):
the owned paths are the backticked tokens containing `/` in the table's "owns" cell. Files changed = tracked files
differing from the baseline commit (`git diff --numstat`) plus untracked files, matched against those paths; a file
owned by two teams counts for both and shows as "shared". Last activity = newest modification time among those files
and the team's tests; **stale** = nothing touched for 2 h, or nothing yet. Tests = new or changed Python test files in
the team's paths (or named like the team's area, `TEAM_TEST_KEYS`) passed to `pytest --collect-only -q`: the count is
tests collected, not passed, and a collection error shows in red. Godot test files are listed but not collected.
KPI tiles *Teams active* (files changed > 0) and *Teams stale* sit beside the branch tiles.

**Branch cards** (datasets `branches`, `ci_runs`, `ci_jobs`), one per `main`, `ro/*` and `claude/ro-*` branch, local
or on origin. Left stripe: green when the latest run passed, red when it failed, amber when the branch is stale
with no run. Lines: where the branch is (`pushed`, `local only`, or `unpushed commits` when the local head differs
from origin), age of the last commit, head sha with subject and author, ahead/behind `origin/main`, then the latest
CI run with one pill per job and its conclusion.

**Latest CI failures** (calc `failures` over `ci_jobs`): every job whose conclusion is not success in the latest run
of each branch and in the last 30 runs that failed, newest first, with the job name and run id.

**Test counts** (dataset `tests`): one row per source (CI engine, CI client, local pytest log when present) with
passed, failed, when it was measured and the exact command or file it was parsed from.

**Worktrees** (dataset `worktrees`): `git worktree list` on the lead's PC, with each worktree's branch and head.

## Rules the data follows

- One field name per fact across all datasets (`updated`, `stale`, `last_commit`, `head_sha`, `run_id`, `job`,
  `conclusion`).
- Every row carries `updated` (ISO UTC, when it was collected) and `stale` (true when the fact's own time, such as
  the last commit or the run start, is older than `--stale-hours`). The page adds its own age check on the
  snapshot time.
- A collector that fails writes a `PROBLEM` line and leaves its dataset empty rather than inventing rows; the
  count shows in the summary line and on the board.
- Check values against the primary source before trusting a card: `git log -1 <branch>`,
  `gh run list --workflow checks --branch <branch> --limit 1`, `gh run view <id> --json jobs`,
  `gh run view <id> --log --job <job id>`.
