#!/usr/bin/env python3
"""Ops board refresh for jaxspike: collect branch, CI, test and worktree facts from primary sources only.

Usage:
  python tools/ops/refresh_ops.py [--repo .] [--out tools/ops] [--pytest-log tools/ops/logs/pytest_latest.log]
                                  [--no-fetch] [--stale-hours 2]

Sources (nothing is self-reported by a session):
  git         fetch origin, branches main + ro/* + claude/ro-*, ahead/behind origin/main, worktree list
  gh          latest `checks` workflow run per branch and its jobs; recent failed runs and their jobs
  gh logs     test counts from the engine-pytest and client-unit job logs of the latest finished main run
  local log   test counts from the latest local pytest log, when the file exists

Writes <out>/ops_data.json (every dataset under one key, plus `generated`) and one <out>/out/<id>.json per
dataset for upload to the Dashboard artifact. Every row carries `updated` (collection time, ISO UTC) and
`stale` (true when the fact's own time is older than --stale-hours). Adapted from client/tools/ops_board/refresh.py
(standard library only, read-only git, gh CLI). Prints PROBLEM lines, then one OPS_REFRESH summary line.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

BRANCH_RE = re.compile(r"^(main|ro/.+|claude/ro-.+)$")
CHECKS_WORKFLOW = "checks"
TEST_JOBS = {"engine": "engine-pytest", "client": "client-unit"}
RECENT_RUNS = 30
problems = []


def now_utc():
	return datetime.now(timezone.utc)


def iso(dt):
	return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(s):
	return datetime.fromisoformat(s.replace("Z", "+00:00"))


def run(cmd, cwd=None, check=True):
	r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
	if check and r.returncode != 0:
		raise RuntimeError("%s: %s" % (" ".join(cmd[:3]), (r.stderr or r.stdout).strip()[:300]))
	return r.stdout


def git(repo, *args):
	return run(["git", "-C", repo] + list(args))


def gh_json(args):
	return json.loads(run(["gh"] + args))


# ---------- collectors ----------

def c_branches(repo, now, stale_after):
	"""One row per branch named main, ro/* or claude/ro-* (local or on origin). Facts from git only."""
	fmt = "%(refname:short)\t%(objectname:short)\t%(committerdate:iso-strict)\t%(authorname)\t%(subject)"
	rows = {}
	remote = {}
	for line in git(repo, "for-each-ref", "--format=" + fmt, "refs/heads", "refs/remotes/origin").splitlines():
		parts = line.split("\t", 4)
		if len(parts) < 5:
			continue
		ref, sha, when, author, subject = parts
		is_remote = ref.startswith("origin/")
		name = ref[len("origin/"):] if is_remote else ref
		if not BRANCH_RE.match(name):
			continue
		rec = {"sha": sha, "when": when, "author": author, "subject": subject}
		if is_remote:
			remote[name] = rec
		else:
			rows[name] = rec
	local_names = set(rows)
	for name, rec in remote.items():
		rows.setdefault(name, rec)
	out = []
	for name, rec in rows.items():
		r = remote.get(name)
		if r is None:
			where = "local only"
		elif r["sha"] == rec["sha"]:
			where = "pushed"
		else:
			where = "unpushed commits"
		ref = name if name in local_names else "origin/" + name
		lr = git(repo, "rev-list", "--left-right", "--count", "origin/main...%s" % ref).split()
		behind, ahead = int(lr[0]), int(lr[1])
		t = parse_time(rec["when"])
		out.append({"branch": name, "head_sha": rec["sha"], "last_commit": iso(t), "author": rec["author"],
			"subject": rec["subject"][:120], "ahead": ahead, "behind": behind, "where": where,
			"updated": iso(now), "stale": (now - t) > stale_after})
	out.sort(key=lambda r: (r["branch"] != "main", r["branch"]))
	return out


def latest_run_for(branch):
	runs = gh_json(["run", "list", "--workflow", CHECKS_WORKFLOW, "--branch", branch, "--limit", "1", "--json",
		"databaseId,conclusion,status,createdAt,headSha,url,headBranch"])
	return runs[0] if runs else None


def jobs_of(run_id):
	return gh_json(["run", "view", str(run_id), "--json", "jobs"])["jobs"]


def c_ci(branches, now, stale_after):
	"""Latest checks run per branch (gh run list) with per-job conclusions (gh run view --json jobs)."""
	runs_rows, jobs_rows = [], []
	seen = set()

	def add_run(rr, branch):
		if rr is None or rr["databaseId"] in seen:
			return
		seen.add(rr["databaseId"])
		started = parse_time(rr["createdAt"])
		result = rr["conclusion"] or rr["status"]
		runs_rows.append({"run_id": str(rr["databaseId"]), "branch": branch, "head_sha": rr["headSha"][:7],
			"result": result, "started": iso(started), "url": rr["url"], "updated": iso(now),
			"stale": (now - started) > stale_after})
		try:
			for j in jobs_of(rr["databaseId"]):
				js = parse_time(j.get("startedAt") or rr["createdAt"]) if j.get("startedAt") else started
				jobs_rows.append({"run_id": str(rr["databaseId"]), "job": j["name"], "job_id": str(j["databaseId"]),
					"branch": branch, "conclusion": j["conclusion"] or j["status"], "started": iso(js),
					"url": j.get("url", ""), "updated": iso(now), "stale": (now - js) > stale_after})
		except Exception as e:
			problems.append("jobs of run %s: %s" % (rr["databaseId"], e))

	for b in branches:
		try:
			add_run(latest_run_for(b["branch"]), b["branch"])
		except Exception as e:
			problems.append("runs of %s: %s" % (b["branch"], e))
	try:
		recent = gh_json(["run", "list", "--workflow", CHECKS_WORKFLOW, "--limit", str(RECENT_RUNS), "--json",
			"databaseId,conclusion,status,createdAt,headSha,url,headBranch"])
		for rr in recent:
			if rr["conclusion"] == "failure" and BRANCH_RE.match(rr["headBranch"] or ""):
				add_run(rr, rr["headBranch"])
	except Exception as e:
		problems.append("recent runs: %s" % e)
	runs_rows.sort(key=lambda r: r["started"], reverse=True)
	jobs_rows.sort(key=lambda r: (r["started"], r["job"]), reverse=True)
	return runs_rows, jobs_rows


def parse_test_summary(text):
	"""(passed, failed) from a pytest summary or the client's TESTS line, else None."""
	m = re.search(r"TESTS passed=(\d+) failed=(\d+)", text)
	if m:
		return int(m.group(1)), int(m.group(2))
	tail = text[-4000:]
	m = re.findall(r"(\d+) (passed|failed)", tail)
	if m:
		counts = {k: int(v) for v, k in m}
		return counts.get("passed", 0), counts.get("failed", 0)
	return None


def c_tests(runs_rows, jobs_rows, pytest_log, now, stale_after):
	"""Test counts: CI job logs of the latest finished main run, and the local pytest log when present."""
	rows = []
	main_run = next((r for r in runs_rows if r["branch"] == "main" and r["result"] in ("success", "failure")), None)
	if main_run:
		for suite, job_name in TEST_JOBS.items():
			job = next((j for j in jobs_rows if j["run_id"] == main_run["run_id"] and j["job"] == job_name), None)
			if job is None:
				problems.append("no job %s in run %s" % (job_name, main_run["run_id"]))
				continue
			try:
				log = run(["gh", "run", "view", main_run["run_id"], "--log", "--job", job["job_id"]])
			except Exception as e:
				problems.append("log of %s: %s" % (job_name, e))
				continue
			got = parse_test_summary(log)
			if got is None:
				problems.append("no test summary in %s log of run %s" % (job_name, main_run["run_id"]))
				continue
			t = parse_time(job["started"])
			rows.append({"suite": suite + " (CI main)", "passed": got[0], "failed": got[1],
				"source": "gh run view %s --log --job %s" % (main_run["run_id"], job["job_id"]),
				"measured": iso(t), "updated": iso(now), "stale": (now - t) > stale_after})
	if pytest_log and os.path.exists(pytest_log):
		text = open(pytest_log, encoding="utf-8", errors="replace").read()
		got = parse_test_summary(text)
		t = datetime.fromtimestamp(os.path.getmtime(pytest_log), timezone.utc)
		if got is None:
			problems.append("no pytest summary in %s" % pytest_log)
		else:
			rows.append({"suite": "engine (local log)", "passed": got[0], "failed": got[1],
				"source": pytest_log.replace("\\", "/"), "measured": iso(t), "updated": iso(now),
				"stale": (now - t) > stale_after})
	return rows


def c_worktrees(repo, now, stale_after):
	rows, cur = [], {}
	for line in git(repo, "worktree", "list", "--porcelain").splitlines() + [""]:
		if not line:
			if cur:
				rows.append(cur)
			cur = {}
			continue
		k, _, v = line.partition(" ")
		if k == "worktree":
			cur["path"] = v.replace("\\", "/")
		elif k == "HEAD":
			cur["head_sha"] = v[:7]
		elif k == "branch":
			cur["branch"] = v.replace("refs/heads/", "")
	out = []
	for w in rows:
		when = git(repo, "log", "-1", "--format=%cI", w.get("head_sha", "HEAD")).strip()
		t = parse_time(when) if when else now
		out.append({"path": w["path"], "branch": w.get("branch", "(detached)"), "head_sha": w.get("head_sha", ""),
			"last_commit": iso(t), "updated": iso(now), "stale": (now - t) > stale_after})
	return out


# ---------- main ----------

def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--repo", default=".")
	ap.add_argument("--out", default=None, help="output dir (default: the script's own folder)")
	ap.add_argument("--pytest-log", default=None, help="latest local pytest log (default: <out>/logs/pytest_latest.log)")
	ap.add_argument("--no-fetch", action="store_true")
	ap.add_argument("--stale-hours", type=float, default=2.0)
	args = ap.parse_args()
	repo = os.path.abspath(args.repo)
	out = os.path.abspath(args.out or os.path.dirname(os.path.abspath(__file__)))
	pytest_log = args.pytest_log or os.path.join(out, "logs", "pytest_latest.log")
	now = now_utc()
	stale_after = timedelta(hours=args.stale_hours)

	if not args.no_fetch:
		try:
			git(repo, "fetch", "origin", "--prune")
		except Exception as e:
			problems.append("fetch: %s" % e)
	data = {}
	try:
		data["branches"] = c_branches(repo, now, stale_after)
	except Exception as e:
		problems.append("branches: %s" % e)
		data["branches"] = []
	data["ci_runs"], data["ci_jobs"] = c_ci(data["branches"], now, stale_after)
	data["tests"] = c_tests(data["ci_runs"], data["ci_jobs"], pytest_log, now, stale_after)
	try:
		data["worktrees"] = c_worktrees(repo, now, stale_after)
	except Exception as e:
		problems.append("worktrees: %s" % e)
		data["worktrees"] = []
	remote = run(["git", "-C", repo, "remote", "get-url", "origin"], check=False).strip()
	main_sha = run(["git", "-C", repo, "rev-parse", "--short", "origin/main"], check=False).strip()
	data["meta"] = [{"generated": iso(now), "repo": remote, "main_sha": main_sha, "stale_hours": args.stale_hours,
		"problems": len(problems), "problem_text": "; ".join(problems)[:500], "updated": iso(now), "stale": False}]

	os.makedirs(os.path.join(out, "out"), exist_ok=True)
	for ds_id, rows in data.items():
		with open(os.path.join(out, "out", ds_id + ".json"), "w", encoding="utf-8") as f:
			json.dump(rows if rows else [{"_empty": "no rows"}], f, indent=1)
			f.write("\n")
	with open(os.path.join(out, "ops_data.json"), "w", encoding="utf-8") as f:
		json.dump(dict(generated=iso(now), **data), f, indent=1)
		f.write("\n")

	latest_by_branch = {}
	for r in data["ci_runs"]:
		latest_by_branch.setdefault(r["branch"], r)
	green = sum(1 for b in data["branches"] if latest_by_branch.get(b["branch"], {}).get("result") == "success")
	red = sum(1 for b in data["branches"] if latest_by_branch.get(b["branch"], {}).get("result") == "failure")
	stale = sum(1 for b in data["branches"] if b["stale"])
	main_ci = latest_by_branch.get("main", {}).get("result", "none")
	tests = {r["suite"]: r for r in data["tests"]}
	eng = tests.get("engine (CI main)", {}).get("passed", "?")
	cli = tests.get("client (CI main)", {}).get("passed", "?")
	for p in problems:
		print("PROBLEM", p)
	print("OPS_REFRESH branches=%d green=%d red=%d stale=%d main_ci=%s engine_tests=%s client_tests=%s worktrees=%d problems=%d at=%s"
		% (len(data["branches"]), green, red, stale, main_ci, eng, cli, len(data["worktrees"]), len(problems), iso(now)))
	return 0


if __name__ == "__main__":
	sys.exit(main())
