#!/usr/bin/env python3
"""Agent tracking stats for the lead session and its subagents, for the agent dashboard.

Reads Claude Code transcripts (the lead's <session>.jsonl and subagents/agent-*.jsonl plus their .meta.json) and
writes two JSON files of row objects:
  <out>/agents.json             one row per agent: role, model, status, tokens, turns, tool calls, times
  <out>/tokens_per_minute.json  one row per minute and model: input, cache, output and total tokens
  <out>/project.json            with --repo: one row per commit (time, subject) plus the latest CI runs via `gh`

Token pull = every token the model processed: input + cache reads + cache writes + output.
Usage: python3 tools/agent_stats.py [--session DIR_OR_JSONL] [--out DIR] [--minutes N]
With no --session it picks the most recently modified session under ~/.claude/projects/.
Standard library only.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timedelta, timezone

IDLE_AFTER_S = 300  # a running agent with no activity for this long shows as idle


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def find_session(arg):
    if arg:
        if os.path.isdir(arg):
            return arg.rstrip("/") + ".jsonl", arg
        return arg, arg[: -len(".jsonl")]
    cands = glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl"))
    if not cands:
        sys.exit("agent_stats: no session transcripts under ~/.claude/projects")
    main = max(cands, key=os.path.getmtime)
    return main, main[: -len(".jsonl")]


def read_usage(path):
    """Yields (timestamp, model, usage dict) for each assistant message, counting each API request once."""
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") != "assistant":
                continue
            m = d.get("message") or {}
            u = m.get("usage")
            if not u or "timestamp" not in d:
                continue
            key = d.get("requestId") or m.get("id") or d.get("uuid")
            if key in seen:  # one API response can be split over several transcript lines
                continue
            seen.add(key)
            yield parse_ts(d["timestamp"]), m.get("model", "unknown"), u


def tool_calls(path):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"tool_use"' in line and '"type":"assistant"' in line.replace(" ", ""):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                for c in (d.get("message") or {}).get("content") or []:
                    if isinstance(c, dict) and c.get("type") == "tool_use":
                        n += 1
    return n


def first_ts(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "timestamp" in d:
                return parse_ts(d["timestamp"])
    return None


def short_model(m):
    for k in ("fable", "opus", "sonnet", "haiku"):
        if k in m:
            return k
    return m


def collect(main_path, sub_dir, now):
    agents, minute = [], {}
    sources = [("lead", main_path, {"description": "Lead (orchestrator)", "model": "", "finished": False})]
    for p in sorted(glob.glob(os.path.join(sub_dir, "subagents", "agent-*.jsonl"))):
        meta_path = p[: -len(".jsonl")] + ".meta.json"
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        sources.append((os.path.basename(p)[len("agent-"): -len(".jsonl")], p, meta))
    for agent_id, path, meta in sources:
        if not os.path.exists(path):
            continue
        tot = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
        turns, last, models = 0, None, {}
        for ts, model, u in read_usage(path):
            row = {
                "input": u.get("input_tokens", 0) or 0,
                "cache_read": u.get("cache_read_input_tokens", 0) or 0,
                "cache_write": u.get("cache_creation_input_tokens", 0) or 0,
                "output": u.get("output_tokens", 0) or 0,
            }
            for k in tot:
                tot[k] += row[k]
            turns += 1
            last = ts if last is None or ts > last else last
            sm = short_model(model)
            models[sm] = models.get(sm, 0) + 1
            mk = (ts.replace(second=0, microsecond=0), sm)
            b = minute.setdefault(mk, {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "requests": 0})
            for k in row:
                b[k] += row[k]
            b["requests"] += 1
        started = first_ts(path)
        if meta.get("finished"):
            status = "done"
        elif last and (now - last).total_seconds() > IDLE_AFTER_S:
            status = "idle"
        else:
            status = "running"
        model = max(models, key=models.get) if models else short_model(meta.get("model") or "unknown")
        agents.append({
            "agent": meta.get("description") or agent_id,
            "id": agent_id,
            "role": "lead" if agent_id == "lead" else "worker",
            "model": model,
            "status": status,
            "started": started.isoformat() if started else "",
            "last_active": last.isoformat() if last else "",
            "minutes": round(((last or now) - started).total_seconds() / 60.0, 1) if started else 0,
            "turns": turns,
            "tool_calls": tool_calls(path),
            "input_tokens": tot["input"],
            "cache_read_tokens": tot["cache_read"],
            "cache_write_tokens": tot["cache_write"],
            "output_tokens": tot["output"],
            "total_tokens": sum(tot.values()),
        })
    return agents, minute


def project_rows(repo):
    """Commits and CI runs as one row list with a `kind` column ("commit" or "ci")."""
    import subprocess
    rows = []
    try:
        log = subprocess.run(["git", "-C", repo, "log", "--format=%cI%x09%h%x09%s", "-n", "500"],
                             capture_output=True, text=True, check=True).stdout
        for line in log.splitlines():
            when, sha, subject = line.split("\t", 2)
            rows.append({"kind": "commit", "time": when, "id": sha, "title": subject, "status": "", "conclusion": ""})
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    try:
        out = subprocess.run(["gh", "run", "list", "-L", "50", "--json",
                              "databaseId,createdAt,displayTitle,status,conclusion"],
                             cwd=repo, capture_output=True, text=True, check=True, timeout=60).stdout
        for r in json.loads(out):
            rows.append({"kind": "ci", "time": r["createdAt"], "id": str(r["databaseId"]), "title": r["displayTitle"],
                         "status": r["status"], "conclusion": r.get("conclusion") or ""})
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session")
    ap.add_argument("--out", default="out/agent_stats")
    ap.add_argument("--minutes", type=int, default=0, help="only the last N minutes of the timeline (0 = all)")
    ap.add_argument("--repo", help="a git checkout: also write project.json (commits and CI runs)")
    a = ap.parse_args()
    main_path, sub_dir = find_session(a.session)
    now = datetime.now(timezone.utc)
    agents, minute = collect(main_path, sub_dir, now)
    rows = []
    if minute:
        start = min(k[0] for k in minute)
        if a.minutes:
            start = max(start, now.replace(second=0, microsecond=0) - timedelta(minutes=a.minutes))
        end = now.replace(second=0, microsecond=0)
        models = sorted({k[1] for k in minute})
        t = start
        while t <= end:  # every minute and model, zeros included, so charts show quiet stretches
            for m in models:
                b = minute.get((t, m), {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "requests": 0})
                rows.append({
                    "minute": t.strftime("%Y-%m-%dT%H:%M:00Z"), "model": m, "requests": b["requests"],
                    "input_tokens": b["input"], "cache_read_tokens": b["cache_read"],
                    "cache_write_tokens": b["cache_write"], "output_tokens": b["output"],
                    "total_tokens": b["input"] + b["cache_read"] + b["cache_write"] + b["output"],
                })
            t += timedelta(minutes=1)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "agents.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": now.isoformat(), "rows": agents}, f, indent=1)
    with open(os.path.join(a.out, "tokens_per_minute.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": now.isoformat(), "rows": rows}, f, indent=1)
    if a.repo:
        with open(os.path.join(a.out, "project.json"), "w", encoding="utf-8") as f:
            json.dump({"generated": now.isoformat(), "rows": project_rows(a.repo)}, f, indent=1)
    total = sum(r["total_tokens"] for r in agents)
    running = sum(1 for r in agents if r["status"] == "running")
    print("AGENT_STATS agents=%d running=%d tokens=%d minutes=%d out=%s"
          % (len(agents), running, total, len({r["minute"] for r in rows}), a.out))


if __name__ == "__main__":
    main()
