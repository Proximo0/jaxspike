"""The rename in docs/naming.md is complete: no Ragnarok Online proper noun is left in the repo's text files.

Scans the same files with the same mapping and skip rules as tools/rename_names.py (so the two cannot drift) and
fails on any old name still present, in file contents or in the paths the script renames.

Until the lead applies the rename (round 3 integration), the old names are still everywhere, so this test fails
by design. Run the suite with the environment variable NAMING_PENDING=1 to make it pass today:

    NAMING_PENDING=1 venv/Scripts/python.exe -m pytest tests -q          (Git Bash)
    $env:NAMING_PENDING = "1"; venv\\Scripts\\python.exe -m pytest tests -q (PowerShell)

With NAMING_PENDING=1 the test still runs the scan and checks that it finds the old names (so the scanner itself is
known to work); once the rename is applied the variable is no longer needed and the test must pass without it.
The mapping's own consistency (no new name contains an old one, no duplicate ids) is checked regardless.
"""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("rename_names", os.path.join(ROOT, "tools", "rename_names.py"))
rn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rn)

PENDING = os.environ.get("NAMING_PENDING") == "1"


def _scan_repo():
    hits = []
    for path, rel in rn.iter_files(ROOT):
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        hits.extend((rel, ln, old, line) for ln, old, line in rn.scan_text(text, rel))
    return hits, rn.scan_paths(ROOT)


def test_no_ro_names_left():
    hits, paths = _scan_repo()
    if PENDING:
        assert hits, "NAMING_PENDING=1 but no old name was found: the rename is applied, drop the variable"
        return
    by_file = {}
    for rel, ln, old, line in hits:
        by_file.setdefault(rel, []).append(f"  {ln}: [{old}] {line[:100]}")
    report = "\n".join(f"{rel}\n" + "\n".join(v[:5]) for rel, v in sorted(by_file.items()))
    assert not hits and not paths, (
        f"{len(hits)} Ragnarok Online names in {len(by_file)} files and {len(paths)} paths still to rename "
        f"(expected until the lead applies tools/rename_names.py; run with NAMING_PENDING=1 before that):\n"
        f"{report}\npaths: {paths}")


def test_mapping_is_consistent():
    olds = [old for old, _, _, _ in rn.MAPPING]
    assert len(olds) == len(set((o, m) for o, _, m, _ in rn.MAPPING)), "duplicate (old form, mode) entries"
    assert all(old != new for old, new, _, _ in rn.MAPPING), "an entry maps a name to itself"
    # no new name may contain an old form as a whole word: the sequential passes would cascade
    for _, new, _, _ in rn.MAPPING:
        for old, _, mode, _, pat in rn.PATTERNS:
            assert not pat.search(new), f"new name {new!r} contains old form {old!r}"
    snakes = [rn._snake(new) for _, new, _ in rn.MONSTERS]
    assert len(snakes) == len(set(snakes)), "two monsters share a snake id"
    assert len(rn.MONSTERS) >= 20 and len(rn.PATH_RENAMES) == 25
    assert all(o != n for o, n in rn.PATH_RENAMES)
    # the 20 table rows are covered, in order, whatever the data file currently says
    try:
        from game.data import monsters
        names = list(monsters.NAMES)
    except Exception:  # the engine is not importable here (no JAX): skip the cross-check only
        return
    mapped = [old for old, _, _ in rn.MONSTERS[:20]]
    unmapped = [n for n in names if n not in mapped and n not in [new for _, new, _ in rn.MONSTERS]]
    assert not unmapped, f"monster rows not in the naming map: {unmapped}"
