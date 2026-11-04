#!/usr/bin/env python3
"""Ops board refresh for jaxspike: collect branch, CI, test and worktree facts from primary sources only.

Usage:
  python tools/ops/refresh_ops.py [--repo .] [--out tools/ops] [--pytest-log tools/ops/logs/pytest_latest.log]
                                  [--no-fetch] [--stale-hours 2] [--baseline dbe5049] [--no-collect]

Sources (nothing is self-reported by a session):
  git         fetch origin, branches main + ro/* + claude/ro-*, ahead/behind origin/main, worktree list
  gh          latest `checks` workflow run per branch and its jobs; recent failed runs and their jobs
  gh logs     test counts from the engine-pytest and client-unit job logs of the latest finished main run
  local log   test counts from the latest local pytest log, when the file exists
  contract    team rows (docs/round3_contract.md section 2): owned paths per team
  git diff    files changed (tracked diff against --baseline plus untracked), line counts; file mtimes for last activity
  pytest      `--collect-only -q` of each team's new or changed Python test files (shared checkout, no branches)

Writes <out>/ops_data.json (every dataset under one key, plus `generated`) and one <out>/out/<id>.json per
dataset for upload to the Dashboard artifact. Every row carries `updated` (collection time, ISO UTC) and
`stale` (true when the fact's own time is older than --stale-hours). Adapted from client/tools/ops_board/refresh.py
(standard library only, read-only git, gh CLI). Prints PROBLEM lines, then one OPS_REFRESH summary line.
"""
import argparse
import fnmatch
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
	manual = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manual_tests.json")
	if os.path.exists(manual):  # facts the lead measured elsewhere (GPU runs in WSL); shown with their source, never refreshed
		for m in json.load(open(manual, encoding="utf-8")):
			m.update({"updated": iso(now), "stale": False})
			rows.append(m)
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


# ---------- round 2: per-team view (shared checkout, no branches) ----------

CONTRACT = "docs/round3_contract.md"
IGNORED = ("tools/ops/out/", "tools/ops/logs/", "tools/ops/ops_data.json", "__pycache__/", ".pytest_cache/")
EXTRA_TEAMS = {"ro3-names": ["tools/rename_world.py", "docs/names.md", "game/data/names.py", "game/data/towns.py"]}
TEAM_ROW_RE = re.compile(r"^\|\s*\*\*([\w-]+)\*\*\s*\|(.*?)\|(.*)\|\s*$")
# Tests the contract names only as "tests" are attributed to a team by file-name keyword.
TEAM_TEST_KEYS = {"ro3-engine": ("stats", "combat", "skills"), "ro3-balance": ("spawn", "soak", "balance"),
	"ro3-perf": ("anim", "view", "shop", "monster"), "ro3-wire": ("protocol", "server", "client"),
	"ro3-exp": ("exp", "mvp", "party"), "ro3-items3": ("item", "drop", "shop"),
	"ro3-world3": ("world", "warp", "map_gen", "craft"), "ro3-client": ("hud", "input", "ui"),
	"ro3-anim3": ("anim",), "ro3-bots3": ("bot", "policy"), "ro3-names": ("name",), "ro3-netperf": ("protocol", "server", "net_"),
	"ro-move": ("move", "pathing", "direction"), "ro-anim": ("anim",), "client-anim": ("anim",),
	"ro-fidelity": ("fidelity",), "ro-monsters": ("monster",), "ro-world2": ("world", "warp", "map_gen", "terrain"),
	"ro-items2": ("item", "drop", "shop", "kafra"), "netperf2": ("protocol", "server", "net_", "view"),
	"bots2": ("bot", "policy", "soak"), "art-anim": ("asset", "slot"), "client-feel": ("hud", "minimap", "input", "audio")}


def parse_teams(repo):
	"""[(team, [owned path patterns])] from the contract's section 2 table. Only backticked tokens holding a `/` count."""
	path = os.path.join(repo, CONTRACT)
	if not os.path.exists(path):
		problems.append("contract %s missing" % CONTRACT)
		return []
	rows, in_sec = [], False
	for line in open(path, encoding="utf-8", errors="replace").read().splitlines():
		if line.startswith("## "):
			in_sec = line.startswith("## 2.")
		m = TEAM_ROW_RE.match(line) if in_sec else None
		if m:
			pats = [t for t in re.findall(r"`([^`]+)`", m.group(2)) if "/" in t and " " not in t]
			rows.append((m.group(1), pats))
	# Teams started after the contract was written (not in its table); their paths are what the lead briefed, so shared.
	for name, pats in EXTRA_TEAMS.items():
		if name not in [r[0] for r in rows]:
			rows.append((name, pats))
	return rows


def path_matches(f, pat):
	if pat.endswith("/**") or (pat.endswith("/*") and "*" not in pat[:-2]):
		return f.startswith(pat.rsplit("/", 1)[0] + "/")
	if "*" in pat:
		return fnmatch.fnmatch(f, pat)
	return f == pat


def changed_files(repo, baseline):
	"""{path: (added, removed, is_new)} for tracked changes against the baseline plus untracked files; read-only git."""
	files = {}
	for line in git(repo, "diff", "--numstat", baseline).splitlines():
		a, r, f = line.split("\t", 2)
		files[f] = (int(a) if a.isdigit() else 0, int(r) if r.isdigit() else 0, False)
	for f in git(repo, "ls-files", "--others", "--exclude-standard").splitlines():
		files[f] = (0, 0, True)
	return {f: v for f, v in files.items() if not any(i in f for i in IGNORED)}


def is_test_file(f):
	b = os.path.basename(f)
	return (f.endswith(".gd") and "/tests/" in "/" + f) or (b.startswith("test_") and b.endswith(".py"))


def collect_tests(repo, files):
	"""(count or None, note): `pytest --collect-only -q` over Python test files; Godot tests cannot be collected here."""
	py = [f for f in files if f.endswith(".py")]
	if not py:
		return None, "godot tests only" if files else "no test files"
	try:
		r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"] + py,
			cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240,
			env=dict(os.environ, JAX_PLATFORMS="cpu"))
	except Exception as e:
		return None, "collect did not run: %s" % str(e)[:80]
	m = re.search(r"(\d+) tests? collected", r.stdout)
	n = int(m.group(1)) if m else len([l for l in r.stdout.splitlines() if "::" in l]) or None
	if r.returncode != 0 or n is None:
		tail = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()][-1:] or ["collect failed"]
		return n, "collect error: " + tail[0][:100]
	return n, "ok"


def c_teams(repo, baseline, now, stale_after, do_collect):
	"""One row per team in the contract: owned paths, changed files since the baseline commit, newest mtime, tests."""
	try:
		changed = changed_files(repo, baseline)
	except Exception as e:
		problems.append("team diff against %s: %s" % (baseline, e))
		return []
	teams = parse_teams(repo)
	owners = {}
	for name, pats in teams:
		for f in changed:
			if any(path_matches(f, p) for p in pats):
				owners.setdefault(f, []).append(name)
	rows = []
	for name, pats in teams:
		mine = sorted(f for f, o in owners.items() if name in o)
		shared = [f for f in mine if len(owners[f]) > 1]
		tests = set(f for f in mine if is_test_file(f))
		keys = TEAM_TEST_KEYS.get(name, ())
		for f in changed:
			if f not in owners and is_test_file(f) and any(k in os.path.basename(f) for k in keys):
				tests.add(f)
		tests = sorted(tests)
		mt = [(os.path.getmtime(os.path.join(repo, f)), f) for f in sorted(set(mine) | set(tests))
			if os.path.exists(os.path.join(repo, f))]
		last = datetime.fromtimestamp(max(mt)[0], timezone.utc) if mt else None
		n_col, note = collect_tests(repo, tests) if do_collect else (None, "skipped (--no-collect)")
		rows.append({"team": name, "owns": ", ".join(pats)[:300], "files_changed": len(mine),
			"files_new": sum(1 for f in mine if changed[f][2]), "files_shared": len(shared),
			"lines_added": sum(changed[f][0] for f in mine), "lines_removed": sum(changed[f][1] for f in mine),
			"newest_file": max(mt)[1] if mt else "", "last_activity": iso(last) if last else "",
			"test_files": len(tests), "test_names": ", ".join(os.path.basename(t) for t in tests)[:300],
			"tests_collected": n_col, "collect_note": note, "baseline": baseline,
			"updated": iso(now), "stale": last is None or (now - last) > stale_after})
	return rows


# ---------- main ----------

def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--repo", default=".")
	ap.add_argument("--out", default=None, help="output dir (default: the script's own folder)")
	ap.add_argument("--pytest-log", default=None, help="latest local pytest log (default: <out>/logs/pytest_latest.log)")
	ap.add_argument("--no-fetch", action="store_true")
	ap.add_argument("--stale-hours", type=float, default=2.0)
	ap.add_argument("--baseline", default="dbe5049", help="round-3 contract commit the team view diffs against")
	ap.add_argument("--no-collect", action="store_true", help="skip the per-team pytest --collect-only")
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
	try:
		data["teams"] = c_teams(repo, args.baseline, now, stale_after, not args.no_collect)
	except Exception as e:
		problems.append("teams: %s" % e)
		data["teams"] = []
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
	teams = data["teams"]
	for p in problems:
		print("PROBLEM", p)
	print("OPS_REFRESH teams=%d teams_active=%d teams_stale=%d tests_collected=%d branches=%d green=%d red=%d stale=%d main_ci=%s engine_tests=%s client_tests=%s worktrees=%d problems=%d at=%s"
		% (len(teams), sum(1 for t in teams if t["files_changed"]), sum(1 for t in teams if t["stale"]),
		sum(t["tests_collected"] or 0 for t in teams), len(data["branches"]), green, red, stale, main_ci, eng, cli, len(data["worktrees"]), len(problems), iso(now)))
	return 0


if __name__ == "__main__":
	sys.exit(main())
