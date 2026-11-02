#!/usr/bin/env python3
"""Ops board refresh: rebuild the board's cloud-side datasets for any starter-kit repo.

Usage:
  python3 refresh.py --repo <repo path> --config <config.json>
  python3 refresh.py --config <config.json> --updates ID=/_blob/URL [ID=/_blob/URL ...] [--create]

Standard library, read-only git, and `gh api` REST calls (no GraphQL). Every collector writes one JSON
file of row objects into the config's out_dir. Core collectors (project, workers, needs_owner, commits,
branches, runs, checks, assets_glb, goals) run for every repo; optional ones (balance, extra_files) run
only when the config names them and no-op when their source is missing. Quiet: prints problems and
changed files, then one summary line. Schema: schema.md. Setup: README.md.
"""
import argparse
import base64
import csv
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timedelta, timezone

DEFAULTS = {
	"main_branch": "main",
	"out_dir": ".",
	"commit_days": 2,
	"updated_by": "Claude (refresh.py)",
	"sources": {"workers": "workers_src.json", "needs_owner": "needs_owner_src.json"},
	"tests": {
		"unit_file_regex": r"^tests/unit/test_.*\.gd$",
		"unit_func_regex": r"^func test_",
		"sim_suite": "tests/sim/sim_suite.gd",
		"sim_list_regex": r"const SIM_FILES[^=]*=\s*\[(.*?)\]",
		"sim_item_regex": r'"sim_\w+"',
		"lint_ext": ".gd",
	},
	"check_jobs": [
		{"role": "unit", "check": "Unit tests", "jobs": ["unit"], "count": "unit", "count_what": "test functions"},
		{"role": "sims", "check": "Regression sims", "job_prefix": "sims", "count": "sims", "count_what": "sim files in the suite"},
		{"role": "lint", "check": "Lint", "jobs": ["lint"], "count": "lint", "count_what": "scripts linted"},
		{"role": "bots", "check": "Bot matches", "job_prefix": "bot", "count": "jobs", "count_what": "bot shards"},
	],
	"assets": {"dir": "assets", "ext": ".glb"},
	"goals": None,
	"balance": None,
	"extra_files": [],
}

problems = []


# ---------- small helpers ----------

def now_utc():
	return datetime.now(timezone.utc)


def iso(dt):
	return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(s):
	return datetime.fromisoformat(s.replace("Z", "+00:00"))


def gh(path, paginate=False):
	"""One REST call; with paginate, walks &page=N (some proxies refuse gh's Link-header pages)."""
	if paginate:
		out = []
		for page in range(1, 21):
			chunk = gh("%s%spage=%d" % (path, "&" if "?" in path else "?", page))
			out.extend(chunk)
			if len(chunk) < 100:
				break
		return out
	r = subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github+json", path], capture_output=True, text=True)
	if r.returncode != 0:
		raise RuntimeError("gh api %s: %s" % (path, (r.stderr or r.stdout).strip()[:200]))
	return json.loads(r.stdout)


class Repo:
	"""Read-only view of the local clone plus main's tree (local objects, else the API tarball)."""

	def __init__(self, path, slug, main_branch):
		self.path = os.path.abspath(path)
		self.slug = slug
		self.main_branch = main_branch
		self._tree = None
		self.main_sha = None

	def git(self, *args):
		r = subprocess.run(["git", "-C", self.path] + list(args), capture_output=True, text=True)
		if r.returncode != 0:
			raise RuntimeError("git %s: %s" % (" ".join(args), r.stderr.strip()[:200]))
		return r.stdout

	def has(self, sha):
		return subprocess.run(["git", "-C", self.path, "cat-file", "-e", sha + "^{commit}"], capture_output=True).returncode == 0

	def tree(self):
		"""(paths, read(path) -> text) for main at main_sha."""
		if self._tree is None:
			sha = self.main_sha
			if self.has(sha):
				paths = [p for p in self.git("ls-tree", "-r", "--name-only", sha).split("\n") if p]
				self._tree = (paths, lambda p: self.git("show", "%s:%s" % (sha, p)))
			else:
				r = subprocess.run(["gh", "api", "repos/%s/tarball/%s" % (self.slug, sha)], capture_output=True)
				if r.returncode != 0:
					raise RuntimeError("tarball: %s" % r.stderr.decode()[:200])
				tf = tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:gz")
				files = {m.name.split("/", 1)[1]: m for m in tf.getmembers() if m.isfile() and "/" in m.name}
				self._tree = (list(files), lambda p: tf.extractfile(files[p]).read().decode("utf-8", "replace"))
		return self._tree

	def read_main_or_local(self, rel):
		"""(text, where) from main if the file is there, else the local working tree, else (None, None)."""
		paths, read = self.tree()
		if rel in paths:
			return read(rel), "main"
		local = os.path.join(self.path, rel)
		if os.path.exists(local):
			return open(local, encoding="utf-8").read(), "local checkout (not on main yet)"
		return None, None


def slug_from_remote(path):
	r = subprocess.run(["git", "-C", path, "remote", "get-url", "origin"], capture_output=True, text=True)
	m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", r.stdout.strip())
	return m.group(1) if m else None


def rows_from_text(text, fmt):
	if fmt == "json":
		data = json.loads(text)
		if isinstance(data, dict):
			for k in ("rows", "data", "results"):
				if isinstance(data.get(k), list):
					return data[k]
			return [data]
		return data
	out = []
	for r in csv.DictReader(io.StringIO(text), delimiter="\t" if fmt == "tsv" else ","):
		out.append({k: (float(v) if re.match(r"^-?\d+(\.\d+)?$", v or "") else v) for k, v in r.items()})
	return out


# ---------- core collectors ----------

def c_project(cfg, repo, ctx):
	p = dict(cfg["project"])
	p.setdefault("repo", repo.slug)
	p.setdefault("repo_url", "https://github.com/%s" % repo.slug)
	p.setdefault("main_branch", repo.main_branch)
	return [p]


def c_src(name):
	def run(cfg, repo, ctx):
		path = os.path.join(ctx["cfg_dir"], cfg["sources"][name])
		return json.load(open(path)) if os.path.exists(path) else []
	return run


def c_commits(cfg, repo, ctx):
	start = (now_utc() - timedelta(days=int(cfg["commit_days"]) - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
	items = gh("repos/%s/commits?sha=%s&per_page=100&since=%s" % (repo.slug, repo.main_branch, iso(start)), paginate=True)
	rows = []
	for c in items:
		t = parse_time(c["commit"]["committer"]["date"]).astimezone(timezone.utc)
		rows.append({"sha": c["sha"][:7], "time": iso(t), "day": t.strftime("%Y-%m-%d"),
			"hour": t.strftime("%Y-%m-%dT%H:00:00Z"), "message": c["commit"]["message"].split("\n")[0][:160]})
	rows.sort(key=lambda r: r["time"], reverse=True)
	return rows


def c_branches(cfg, repo, ctx):
	rows = []
	pushed = set()
	for b in gh("repos/%s/branches?per_page=100" % repo.slug, paginate=True):
		name = b["name"]
		pushed.add(name)
		if name == repo.main_branch:
			continue
		cmp = gh("repos/%s/compare/%s...%s" % (repo.slug, repo.main_branch, name))
		head = cmp["commits"][-1] if cmp.get("commits") else gh("repos/%s/commits/%s" % (repo.slug, b["commit"]["sha"]))
		rows.append({"branch": name, "where": "Pushed", "ahead": cmp["ahead_by"], "behind": cmp["behind_by"],
			"last_commit": iso(parse_time(head["commit"]["committer"]["date"])),
			"last_message": head["commit"]["message"].split("\n")[0][:120]})
	if repo.has(repo.main_sha):
		for name in repo.git("for-each-ref", "--format=%(refname:short)", "refs/heads").split():
			if name in pushed or name == repo.main_branch:
				continue
			info = repo.git("log", "-1", "--format=%cI\t%s", name).strip().split("\t", 1)
			rows.append({"branch": name, "where": "Local only",
				"ahead": int(repo.git("rev-list", "--count", "%s..%s" % (repo.main_sha, name)).strip()),
				"behind": int(repo.git("rev-list", "--count", "%s..%s" % (name, repo.main_sha)).strip()),
				"last_commit": iso(parse_time(info[0])), "last_message": info[1][:120] if len(info) > 1 else ""})
	rows.sort(key=lambda r: (-r["ahead"], r["branch"]))
	return rows


def c_runs(cfg, repo, ctx):
	rows = []
	for r in gh("repos/%s/actions/runs?per_page=100" % repo.slug)["workflow_runs"]:
		start = parse_time(r.get("run_started_at") or r["created_at"])
		dur = round((parse_time(r["updated_at"]) - start).total_seconds()) if r["status"] == "completed" else None
		rows.append({"run_id": str(r["id"]), "workflow": r["name"], "branch": r["head_branch"], "sha": r["head_sha"][:7],
			"event": r["event"], "status": r["status"], "result": r["conclusion"] or r["status"],
			"started": iso(start), "duration_s": dur})
	rows.sort(key=lambda r: r["started"])
	ctx["runs"] = rows
	return rows


def c_checks(cfg, repo, ctx):
	wf = cfg["project"].get("checks_workflow", "checks")
	latest = next((r for r in reversed(ctx.get("runs") or []) if r["workflow"] == wf and r["branch"] == repo.main_branch and r["status"] == "completed"), None)
	if latest is None:
		return []
	jobs = gh("repos/%s/actions/runs/%s/jobs?per_page=100" % (repo.slug, latest["run_id"]))["jobs"]
	by = {j["name"]: j["conclusion"] or j["status"] for j in jobs}
	paths, read = repo.tree()
	t = cfg["tests"]
	counts = {"unit": 0, "sims": 0, "lint": 0}
	for p in paths:
		if re.match(t["unit_file_regex"], p):
			counts["unit"] += len(re.findall(t["unit_func_regex"], read(p), re.M))
		if p.endswith(t["lint_ext"]):
			counts["lint"] += 1
	if t.get("sim_suite") and t["sim_suite"] in paths:
		m = re.search(t["sim_list_regex"], read(t["sim_suite"]), re.S)
		counts["sims"] = len(re.findall(t["sim_item_regex"], m.group(1))) if m else 0
	rows = []
	for spec in cfg["check_jobs"]:
		names = [n for n in by if (spec.get("job_prefix") and n.startswith(spec["job_prefix"])) or n in spec.get("jobs", [])]
		if not names:
			continue
		results = [by[n] for n in names]
		result = "success" if all(x == "success" for x in results) else next((x for x in results if x != "success"), "missing")
		count = len(names) if spec.get("count") == "jobs" else counts.get(spec.get("count"), None)
		rows.append({"role": spec["role"], "check": spec["check"], "result": result, "count": count,
			"count_what": spec.get("count_what", ""), "sha": latest["sha"], "run_id": latest["run_id"], "run_at": latest["started"]})
	return rows


def c_assets(cfg, repo, ctx):
	a = cfg["assets"]
	paths, _ = repo.tree()
	counts = {}
	for p in paths:
		if p.startswith(a["dir"].rstrip("/") + "/") and p.endswith(a["ext"]):
			parts = p.split("/")
			folder = parts[1] if len(parts) > 2 else "(top)"
			counts[folder] = counts.get(folder, 0) + 1
	return [{"folder": k, "files": v} for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def classify(status):
	s = status.lower().strip()
	if "missed" in s:
		return "Missed"
	if s.startswith("met"):
		return "Met"
	if "unmeasured" in s:
		return "Unmeasured"
	if s.startswith("no target"):
		return "No target"
	if s.startswith("weak"):
		return "Weak"
	return "Other"


def c_goals(cfg, repo, ctx):
	g = cfg.get("goals")
	if not g or not os.path.exists(g["md"]):
		return []
	part = open(g["md"], encoding="utf-8").read().split(g.get("heading", "## Goals table"), 1)
	if len(part) < 2:
		return []
	rows = []
	for line in part[1].splitlines():
		if not line.startswith("|") or re.match(r"^\|\s*-", line) or line.lower().startswith("| target"):
			continue
		cells = [c.strip() for c in line.strip().strip("|").split("|")]
		if len(cells) < 4:
			continue
		rows.append({"target": cells[0], "measured": cells[1], "status": cells[2], "status_class": classify(cells[2]),
			"source": " | ".join(cells[3:]).replace("`", "")})
	for live in g.get("live", []):  # a goal row kept current from a workflow's run history
		runs = [r for r in ctx.get("runs") or [] if r["workflow"] == live["workflow"] and r["branch"] == repo.main_branch and r["status"] == "completed"]
		if not runs:
			continue
		green, last = sum(1 for r in runs if r["result"] == "success"), runs[-1]
		for r in rows:
			if r["target"].lower().startswith(live["target_prefix"].lower()):
				ok = last["result"] == "success"
				r.update({"measured": "%d of %d green; latest %s at %s UTC" % (green, len(runs), last["result"], last["started"][11:16]),
					"status": "met (live)" if ok else "missed (live)", "status_class": "Met" if ok else "Missed",
					"source": "gh api actions/runs, workflow %s on %s (refresh.py)" % (live["workflow"], repo.main_branch)})
	for i, r in enumerate(rows):
		r["n"] = i + 1
	return rows


# ---------- optional collectors ----------

def parse_before_after_md(text, fights_default):
	"""Rows from a markdown table whose rows start 'vs ...': first number column = before, '(kept)' column = after."""
	rows, header = [], None
	for line in text.splitlines():
		if not line.startswith("|") or re.match(r"^\|\s*-", line):
			continue
		cells = [c.strip() for c in line.strip().strip("|").split("|")]
		if not cells[0].lower().startswith("vs "):
			header = cells
			continue
		if header is None:
			continue
		kept = next((i for i, h in enumerate(header) if "kept" in h.lower()), len(header) - 1)
		m = re.search(r"of (\d+)", header[0])
		fights = int(m.group(1)) if m else fights_default
		b, a = re.match(r"(\d+)", cells[1]), re.match(r"(\d+)", cells[kept])
		if b and a:
			rows.append({"matchup": cells[0], "fights": fights, "before_wins": int(b.group(1)), "after_wins": int(a.group(1)),
				"before": round(int(b.group(1)) / fights, 4), "after": round(int(a.group(1)) / fights, 4)})
	return rows


def c_balance(cfg, repo, ctx):
	"""Before/after studies, one row per matchup. Each study: a markdown doc in the repo or a JSON file here."""
	rows = []
	for st in cfg["balance"].get("studies", []):
		src, got, where = st["source"], [], None
		if "md" in src:
			text, where = repo.read_main_or_local(src["md"])
			got = parse_before_after_md(text, st.get("fights", 20)) if text else []
			where = "%s (%s)" % (src["md"], where) if text else None
		elif "json" in src:
			path = os.path.join(ctx["cfg_dir"], src["json"])
			if os.path.exists(path):
				got = json.load(open(path))
				where = src.get("label", os.path.basename(path))
		for r in got:
			r = dict(r)
			r.setdefault("fights", st.get("fights"))
			rows.append(dict(r, study=st["id"], study_title=st["title"], study_note=st.get("note", ""),
				band_lo=(st.get("band") or [None, None])[0], band_hi=(st.get("band") or [None, None])[1],
				ref=st.get("ref"), ref_label=st.get("ref_label", ""), read_from=where or ""))
	return rows


def c_extra(spec):
	def run(cfg, repo, ctx):
		fmt = spec.get("format") or os.path.splitext(spec["path"])[1].lstrip(".") or "json"
		text = None
		if spec.get("where", "main") == "main":
			paths, read = repo.tree()
			text = read(spec["path"]) if spec["path"] in paths else None
		else:
			local = os.path.join(repo.path, spec["path"])
			text = open(local, encoding="utf-8").read() if os.path.exists(local) else None
		if text is None:
			return [{"_empty": "missing: " + spec["path"]}] if spec.get("empty_if_missing") else None  # None = no-op
		return rows_from_text(text, fmt)
	return run


# ---------- run ----------

def load_config(path):
	cfg = json.load(open(path))
	merged = dict(DEFAULTS)
	merged.update(cfg)
	merged["tests"] = dict(DEFAULTS["tests"], **cfg.get("tests", {}))
	merged["sources"] = dict(DEFAULTS["sources"], **cfg.get("sources", {}))
	merged["assets"] = dict(DEFAULTS["assets"], **cfg.get("assets", {}))
	return merged


def refresh(args):
	cfg = load_config(args.config)
	cfg_dir = os.path.dirname(os.path.abspath(args.config))
	out_dir = os.path.normpath(os.path.join(cfg_dir, cfg["out_dir"]))
	os.makedirs(out_dir, exist_ok=True)
	slug = cfg.get("repo_slug") or slug_from_remote(args.repo)
	if not slug:
		print("PROBLEM no GitHub repo: set repo_slug in the config or an origin remote")
		return 1
	repo = Repo(args.repo, slug, cfg["main_branch"])
	try:
		repo.main_sha = gh("repos/%s/branches/%s" % (slug, cfg["main_branch"]))["commit"]["sha"]
	except RuntimeError as e:
		print("PROBLEM", e)
		return 1
	ctx = {"cfg_dir": cfg_dir}
	steps = [("project", c_project), ("workers", c_src("workers")), ("needs_owner", c_src("needs_owner")),
		("commits", c_commits), ("branches", c_branches), ("runs", c_runs), ("checks", c_checks),
		("assets_glb", c_assets), ("goals", c_goals)]
	if cfg.get("balance"):
		steps.append(("balance", c_balance))
	for spec in cfg.get("extra_files") or []:
		steps.append((spec["dataset"], c_extra(spec)))
	changed, written = [], []
	for ds_id, fn in steps:
		try:
			rows = fn(cfg, repo, ctx)
		except Exception as e:  # keep the last good file
			problems.append("%s: %s" % (ds_id, e))
			continue
		if rows is None:
			continue
		if not rows:  # the board reads an empty file as a failed source: write one marker row instead
			rows = [{"_empty": "no rows yet"}]
		path = os.path.join(out_dir, ds_id + ".json")
		text = json.dumps(rows, indent=1, ensure_ascii=False) + "\n"
		written.append(ds_id)
		if not os.path.exists(path) or open(path, encoding="utf-8").read() != text:
			open(path, "w", encoding="utf-8").write(text)
			changed.append(ds_id)
	stamp = iso(now_utc())
	json.dump({"at": stamp, "main": repo.main_sha[:7], "changed": changed, "written": written},
		open(os.path.join(out_dir, "changed.json"), "w"), indent=1)
	for p in problems:
		print("PROBLEM", p)
	for d in changed:
		print("CHANGED", d, os.path.join(out_dir, d + ".json"))
	print("REFRESH changed=%d problems=%d main=%s at=%s" % (len(changed), len(problems), repo.main_sha[:7], stamp))
	return 1 if problems else 0


def updates(args):
	"""After uploading changed files: one update doc per dataset, and the batch `writes` to send."""
	cfg = load_config(args.config)
	cfg_dir = os.path.dirname(os.path.abspath(args.config))
	out_dir = os.path.join(cfg_dir, cfg["out_dir"], "updates")
	os.makedirs(out_dir, exist_ok=True)
	stamp = iso(now_utc())
	writes = []
	template = {}
	tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets.json")
	if args.create and os.path.exists(tpl_path):
		template = json.load(open(tpl_path))
	for pair in args.updates:
		ds_id, url = pair.split("=", 1)
		doc = {"source": {"kind": "file", "url": url, "name": ds_id + ".json"}, "updated": {"at": stamp, "by": cfg["updated_by"]}}
		if args.create:  # a new dataset document: title, description and schedule from datasets.json
			doc = dict(template.get(ds_id, {"title": ds_id, "description": ""}), **doc)
		path = os.path.join(out_dir, ds_id + ".json")
		json.dump(doc, open(path, "w"), indent=1)
		w = {"op": "set" if args.create else "update", "collection": "datasets", "doc_id": ds_id, "file_path": path}
		if not args.create:
			w["if_version"] = "<version>"
		writes.append(w)
	print(json.dumps(writes, indent=1))
	print("UPDATES datasets=%d at=%s" % (len(writes), stamp))
	return 0


if __name__ == "__main__":
	ap = argparse.ArgumentParser(description="Rebuild the ops board's cloud-side datasets.")
	ap.add_argument("--repo", default=".", help="path to the project's git clone (default: current directory)")
	ap.add_argument("--config", required=True, help="board config JSON (see config.example.json)")
	ap.add_argument("--updates", nargs="+", metavar="ID=URL", help="write update docs for uploaded files instead of refreshing")
	ap.add_argument("--create", action="store_true", help="with --updates: whole new dataset docs (title and description from datasets.json)")
	a = ap.parse_args()
	sys.exit(updates(a) if a.updates else refresh(a))
