"""PreToolUse guard for Bash: refuse heavy commands when the PC is short of RAM or already running too many.

Why: on 2026-11-02 thirteen parallel team sessions each ran pytest, soaks, Godot and test servers at once; free RAM
fell to ~100 MB and the Claude app crashed three times, leaving orphaned processes behind. This hook runs for every
Bash call in this project (lead, teams and their workers) and blocks a heavy command before it starts.

Heavy = pytest, soak/bench scripts, net.server, Godot, Blender, check_all/run_sims/run_tests, exports.
Limits (the user's rule, 2026-11-03: "keep your caps at 60 percent so i can safely run a few things as well", so
Claude's work may use at most 60% of the 16 GB PC; override with env JAXSPIKE_GUARD_MIN_FREE_MB / _MAX_PYTHON /
_MAX_GODOT):
  Claude's heavy processes (python, Godot, Blender) together <= 60% of total RAM, counting the new command at
  1.5 GB (3 GB for a full suite or a soak); free RAM >= 25% of total (4 GB of 16 GB) left for the user's own
  programs (the Claude app itself holds 4-5 GB, so a higher free floor is unreachable even when idle; the user
  approved this measure on 2026-11-03); python.exe processes < 6 (full suite or soak: < 3); Godot < 2; Blender < 2.
Exit 2 blocks the call and the message below tells the model what to do instead. Standard library only.
"""
import ctypes
import json
import os
import re
import subprocess
import sys

HEAVY = re.compile(r"pytest|soak_|bench[/\\]|net\.server|[Gg]odot|blender|check_all|run_sims|run_tests|run_bot|"
                   r"run_lab|export_windows|export_android|server_load|world_scale", re.I)
FULL_SUITE = re.compile(r"pytest\s+(tests\s*(?:-|$|\n|;|\|)|-q\s*$)|pytest\s*$|pytest\s+-q\s+tests\s*(-|$)|"
                        r"pytest\s+tests/?\s*(-|$|&|\|)", re.M)
SOAK = re.compile(r"soak_check|soak_levels|server_load|bench_tick|bench_view|--batch\s+(\d+)", re.I)


def _stat():
    class Stat(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = Stat()
    st.dwLength = ctypes.sizeof(Stat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st


def free_mb() -> int:
    return int(_stat().ullAvailPhys // (1024 * 1024))


def total_mb() -> int:
    return int(_stat().ullTotalPhys // (1024 * 1024))


def counts() -> dict:
    """Process counts by name and the heavy processes' total working set in MB (tasklist's Mem Usage column)."""
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=20).stdout
    c = {"python": -1, "godot": 0, "blender": 0, "heavy_mb": 0}  # -1: this hook is itself a python.exe
    for ln in out.splitlines():
        if not ln.startswith('"'):
            continue
        cols = ln.strip().strip('"').split('","')
        name = cols[0].lower()
        kind = next((k for k in ("python", "godot", "blender") if name.startswith(k)), None)
        if kind is None:
            continue
        c[kind] += 1
        try:
            c["heavy_mb"] += int(re.sub(r"[^0-9]", "", cols[4])) // 1024  # "825,000 K"
        except (IndexError, ValueError):
            pass
    return c


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    cmd = str(data.get("tool_input", {}).get("command", ""))
    # git, echo and other read-only commands are never heavy, whatever words their arguments or messages contain
    first = re.match(r"\s*(?:cd\s+\S+\s*&&\s*)?(\S+)", cmd)
    light = ("git", "echo", "grep", "cat", "ls", "head", "tail", "sed", "rg", "wc", "find", "printf")
    if first and first.group(1) in light and not re.search(r"(&&|;|\|)\s*(?!" + "|".join(light) + r")\S", cmd):
        return 0
    if not HEAVY.search(cmd):
        return 0
    total = total_mb()
    min_free = int(os.environ.get("JAXSPIKE_GUARD_MIN_FREE_MB", total * 25 // 100))  # the user's own room
    budget = total * 60 // 100  # Claude's heavy processes together, including the one about to start
    max_py = int(os.environ.get("JAXSPIKE_GUARD_MAX_PYTHON", 6))
    max_godot = int(os.environ.get("JAXSPIKE_GUARD_MAX_GODOT", 2))
    big = bool(FULL_SUITE.search(cmd) or SOAK.search(cmd))
    need = 3000 if big else 1500  # MB the new command is assumed to take
    if big:
        max_py = min(max_py, 3)
    b = re.search(r"--batch\s+(\d+)", cmd)
    if b and int(b.group(1)) > 4 and "wsl" not in cmd.lower():
        print(f"guard_heavy: --batch {b.group(1)} is over the 4-world cap on this 16 GB PC; the lead runs bigger "
              f"soaks on the GPU in WSL. Use --batch 4 or less.", file=sys.stderr)
        return 2
    try:
        free = free_mb()
        c = counts()
    except Exception as e:  # never wedge the harness on a guard failure
        print(f"guard_heavy: could not measure ({e}); allowing", file=sys.stderr)
        return 0
    problems = []
    if free < min_free:
        problems.append(f"free RAM {free} MB < {min_free} MB (the floor kept for the user's own programs)")
    if c["heavy_mb"] + need > budget:
        problems.append(f"Claude's heavy processes use {c['heavy_mb']} MB; with this command's ~{need} MB that passes "
                        f"the 60% budget of {budget} MB")
    if c["python"] >= max_py:
        problems.append(f"{c['python']} python.exe processes running (limit {max_py} for this command)")
    if c["godot"] >= max_godot:
        problems.append(f"{c['godot']} Godot processes running (limit {max_godot})")
    if c["blender"] >= 2:
        problems.append(f"{c['blender']} Blender processes running (limit 1)")
    if not problems:
        return 0
    print("guard_heavy BLOCKED this heavy command to keep the PC from crashing: " + "; ".join(problems) + ". "
          "Do not retry in a loop. Wait for your own running process to finish (never start a second heavy one), "
          "run only the test files you changed instead of the full suite, leave soaks and full suites to the lead's "
          "integration, and kill any process you started that is no longer needed. If the box stays full, report "
          "to the lead and stop.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
