#!/usr/bin/env python3
"""Per-thread cost and wall-clock rows for the agent dashboard, from cloud session records.

Usage: python3 tools/thread_costs.py <sessions.jsonl | -> [--out DIR]

Project threads run on their own machines, so agent_stats.py (which reads local transcripts) cannot see them.
Their totals live in each session's record instead: the session that refreshes the dashboard (the coordinator, or a
routine) calls get_session for every thread (thread cmsg/cse id X -> session_X) and writes one record per line,
either the tool's {"ccr": {...}} wrapper or the bare session object. This turns them into:
  - threads.json: one row per thread, the agents.json row shape (id, role, model, status, minutes, input/cache/output
    tokens) plus title, cost_usd, created/updated, context_used and cost_per_hour.
  - threads_total.json: one row with the totals.
Wall clock is created_at -> updated_at (last activity), so an idle thread stops accruing. Prints one summary line,
"THREAD_COSTS threads=n cost_usd=c tokens=t".
"""
import datetime
import json
import os
import sys

STATUS = {"WORKING": "running", "BLOCKED": "blocked", "REVIEW_READY": "review", "COMPLETED": "done", "FAILED": "failed"}


def _ts(s: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_records(text: str) -> list:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out.append(rec.get("ccr", rec))
    return out


def row(rec: dict) -> dict:
    meta = rec.get("external_metadata") or {}
    use = meta.get("usage") or {}
    created = _ts(rec["created_at"])
    updated = _ts(rec.get("updated_at") or rec["created_at"])
    minutes = round((updated - created).total_seconds() / 60.0, 1)
    cost = round(float(use.get("cost_usd", 0.0) or 0.0), 2)
    tokens = {k: int(use.get(k, 0) or 0) for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens",
                                                   "output_tokens")}
    bucket = (rec.get("status_bucket") or "").replace("SESSION_STATUS_BUCKET_", "")
    model = meta.get("last_served_model") or (rec.get("session_context") or {}).get("model") or "unknown"
    return dict({
        "id": rec["id"],
        "title": rec.get("title", ""),
        "role": "worker" if rec.get("parent_session_id") else "lead",
        "model": model.split("[")[0],
        "effort": (rec.get("session_context") or {}).get("effort_level", ""),
        "status": STATUS.get(bucket, bucket.lower() or "unknown"),
        "created": rec["created_at"],
        "updated": rec.get("updated_at", ""),
        "minutes": minutes,
        "cost_usd": cost,
        "cost_per_hour": round(cost / (minutes / 60.0), 2) if minutes > 0 else 0.0,
        "context_used": int((meta.get("context_usage") or {}).get("used_tokens", 0) or 0),
    }, **tokens)


def main() -> int:
    args = sys.argv[1:]
    out_dir = "."
    if "--out" in args:
        i = args.index("--out")
        out_dir = args[i + 1]
        del args[i:i + 2]
    if len(args) != 1:
        print(__doc__.strip().split("\n")[2], file=sys.stderr)
        return 2
    text = sys.stdin.read() if args[0] == "-" else open(args[0], encoding="utf-8").read()
    rows = sorted((row(r) for r in parse_records(text)), key=lambda r: -r["cost_usd"])
    total_tokens = sum(r["input_tokens"] + r["cache_read_tokens"] + r["cache_write_tokens"] + r["output_tokens"]
                       for r in rows)
    total = {
        "threads": len(rows),
        "running": sum(1 for r in rows if r["status"] == "running"),
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 2),
        "minutes": round(sum(r["minutes"] for r in rows), 1),
        "tokens": total_tokens,
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "cache_read_share": round(sum(r["cache_read_tokens"] for r in rows) / total_tokens, 3) if total_tokens else 0.0,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "threads.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    with open(os.path.join(out_dir, "threads_total.json"), "w", encoding="utf-8") as fh:
        json.dump([total], fh, indent=1)
    print("THREAD_COSTS threads=%d cost_usd=%.2f tokens=%d" % (total["threads"], total["cost_usd"], total_tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
