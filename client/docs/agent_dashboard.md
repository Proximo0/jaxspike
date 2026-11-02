# Agent dashboard

A live-ish page for the person running the project: which agents are working, on which model, how many tokens
each has used, the **token pull per minute** across all agents, cache hit rate, commits and CI runs. Set it up on
day one, next to CI, and refresh it on every worker hand-back.

Read `docs/lessons/ops_board.md` first: a dashboard fails on missing input far more often than on rendering.

## Pieces
- `tools/agent_stats.py` (standard library only) reads the Claude Code transcripts of the lead session and its
  subagents (`~/.claude/projects/<project>/<session>.jsonl`, `<session>/subagents/agent-*.jsonl` and `.meta.json`)
  and writes three row files:
  - `agents.json`: one row per agent (role, model, status running/idle/done, minutes, turns, tool calls,
    input/cache/output tokens).
  - `tokens_per_minute.json`: one row per minute and model, zeros included. Token pull = input + cache reads +
    cache writes + output.
  - `project.json` (with `--repo .`): commits from `git log` and CI runs from `gh run list`.
- `tools/agent_dashboard/index.html`: the dashboard page (headline tiles, token pull per minute stacked by model,
  tokens by model, tokens by agent, an agents table, CI runs). It is written for the claude.ai **Dashboard**
  Artifact type and reads three datasets named `agents`, `tokens` and `project`.

## Set up (lead, once)
1. `python3 tools/agent_stats.py --repo . --out out/agent_stats` (prints `AGENT_STATS agents=.. running=.. tokens=..`).
   With no `--session` it picks the newest session under `~/.claude/projects/`.
2. Create a dashboard from the Dashboard Artifact type (Artifact `quickstart`, intent `other`), titled
   "<Project> agent tracker".
3. Upload the three JSON files as assets of that dashboard, then write in one batch: `dash/meta` `{title}`,
   `datasets/agents`, `datasets/tokens`, `datasets/project` (each `source: {kind: "file", url: <asset url>, name}`,
   `updated: {at, by}`, `schedule`, `staleAfterHours: 2`), and `files/index.html` `{text: <tools/agent_dashboard/index.html>}`.
4. Save the dashboard link in project memory and the build guide.

## Refresh (lead, on every hand-back and push)
Run step 1 again, upload the three files, and `update` each dataset's `source.url` and `updated`. Old assets can be
deleted once nothing points at them. A routine can do this on a schedule if the session stays up.

## Thread cost (project threads on their own machines)
`agent_stats.py` sees only the lead's local transcripts. Project threads each run in their own cloud session, and
each session record (`get_session`) carries `external_metadata.usage` (`cost_usd`, input, output, cache read and
write tokens), `created_at`/`updated_at`, the served model and status. `list_sessions` carries no usage and does not
list project threads, so the thread ids come from the board's `workers` rows (`session` field, written by the lead
when it starts a thread). `tools/ops_board/thread_costs.py` turns the records into the `threads` and
`threads_total` datasets (schema in `tools/ops_board/schema.md`).

Daily cost routine (a cloud routine on a cheap model, created by a session that is not device-linked):
1. Read the board's `workers` dataset (ArtifactData `get datasets/workers`, then Artifact `read` its source path)
   and collect each row's `session`, plus the coordinator's own session id.
2. Call `get_session` for each id and write each result, one JSON object per line, to `sessions.jsonl`.
3. `python3 tools/ops_board/thread_costs.py sessions.jsonl --out out/` (prints `THREAD_COSTS threads=.. cost_usd=..`).
4. Upload `out/threads.json` and `out/threads_total.json` as board assets and update `datasets/threads` and
   `datasets/threads_total` (`source.url`, `updated`) in one ArtifactData batch, as in the ops board refresh.
A failed `get_session` keeps the thread's previous row out of the file rather than failing the run; the summary
line's `threads=` count shows it.
The board's Work tab shows a Thread cost card (`threads-card`, table on `threads` and totals from `threads_total`). It stays hidden until the `threads` dataset exists.
