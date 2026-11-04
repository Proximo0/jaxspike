"""PreToolUse guard for Bash: refuse heavy commands when the PC is short of RAM or already running too many.

Why: on 2026-11-02 thirteen parallel team sessions each ran pytest, soaks, Godot and test servers at once; free RAM
fell to ~100 MB and the Claude app crashed three times, leaving orphaned processes behind. This hook runs for every
Bash call in this project (lead, teams and their workers) and blocks a heavy command before it starts.

Heavy = pytest, soak/bench scripts, net.server, Godot, Blender, check_all/run_sims/run_tests, exports.
Limits (the user's rule, 2026-11-03: "keep your caps at 60 percent so i can safely run a few things as well", so
Claude's work may use at most 60% of the 16 GB PC; override with env JAXSPIKE_GUARD_MIN_FREE_MB / _MAX_PYTHON /
_MAX_GODOT):
  free RAM >= 40% of total (6.4 GB of 16 GB) for any heavy command, >= 45% for a full pytest suite or a soak;
  python.exe processes < 6 (full suite or soak: < 3); Godot processes < 2; Blender < 2.
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
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=20).stdout
    names = [ln.split('","')[0].strip('"').lower() for ln in out.splitlines() if ln.startswith('"')]
    return {"python": sum(n.startswith("python") for n in names),
            "godot": sum(n.startswith("godot") for n in names),
            "blender": sum(n.startswith("blender") for n in names)}


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
    min_free = int(os.environ.get("JAXSPIKE_GUARD_MIN_FREE_MB", total * 40 // 100))  # Claude uses <= 60%
    max_py = int(os.environ.get("JAXSPIKE_GUARD_MAX_PYTHON", 6))
    max_godot = int(os.environ.get("JAXSPIKE_GUARD_MAX_GODOT", 2))
    big = bool(FULL_SUITE.search(cmd) or SOAK.search(cmd))
    if big:
        min_free = max(min_free, total * 45 // 100)
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
        problems.append(f"free RAM {free} MB < {min_free} MB (Claude's work may use at most 60% of the PC)")
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
