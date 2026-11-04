#!/usr/bin/env python3
"""Apply the naming bible (docs/naming.md): Ragnarok Online proper nouns -> Aurelune names, repo-wide.

Standard library only. Whole-word replacements over the repo's text files, plus the file and folder renames
that follow (sprite folders under client/assets/anim/, the kafra modules). Run from the repo root:

    python tools/rename_names.py --dry-run      # counts per mapping entry and per file, renames, ambiguous hits
    python tools/rename_names.py                # apply (the lead, at integration, when nobody else is editing)
    python tools/rename_names.py --root <dir>   # another checkout (a scratch copy, for a rehearsal)

Matching modes (every entry in MAPPING carries one):
  ident  the old form must not touch a letter or digit on either side; an underscore IS a boundary, so one entry
         covers `kafra`, `kafra_tiles`, `ro_kafra`, `npc_kafra` and `KAFRA` covers `T_KAFRA`, `KAFRA_TELEPORT`.
         Only for names that are unique words (nothing generic is spelled that way).
  word   strict whole word: letters, digits and underscores on either side block the match (zone ids, Title-case
         names of generic words such as `Wolf`, `Skeleton`).
  keyed  lowercase ids of generic words (`wolf`, `skeleton`, `spore`): replaced only between quote, dot or slash
         characters on both sides, i.e. `"skeleton"`, `monster.skeleton"`, `anim/skeleton/`, `anim.skeleton.walk`.
  raw    the literal string, bounded by non-identifier characters on the outside (`_spore(`, `def _spore`).
Entries may be scoped to a file set (the `drops` monster id inside the client's monster-id lists). `Drops` the
monster is guarded: not replaced when a space and a lowercase word follow ("Drops the path").

Left alone: URLs, lines citing rAthena / irowiki / iRO, lines holding NAMING_KEEP, the source docs in SKIP_FILES
(docs/ro_fidelity.md and the historical contracts describe RO as the design source), this script, the test and
the naming bible, and the excluded directories. "RO" and "Ragnarok" as the design source are not in the map.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

TEXT_EXTS = (".py", ".md", ".gd", ".tscn", ".tres", ".json", ".yml", ".html", ".txt", ".import")
SKIP_DIRS = {"venv", ".git", ".claude", "characters", "__pycache__", "node_modules", ".godot"}
# Files that document RO as the design source, or hold the old names on purpose (relative, forward slashes).
SKIP_FILES = {
    "docs/ro_fidelity.md",
    "docs/round2_contract.md",
    "docs/round3_contract.md",
    "docs/ro_contract.md",
    "docs/naming.md",
    "tools/rename_names.py",
    "tests/test_naming.py",
    "docs/production-handover.md",   # describes classic RO as the reference game
    "tools/ops/ops_data.json",       # mirrors git commit subjects
    "tools/ops/out/branches.json",
}
# A line holding any of these is left as it is (URLs, citations, explicit keep marker).
SKIP_LINE_TOKENS = ("http://", "https://", "rAthena", "rathena", "irowiki", "iRO", "NAMING_KEEP")

# ---------------------------------------------------------------------------------------------- the names
TOWNS = (  # old, new (Title). Lowercase zone names follow; `morocc` is RO's zone spelling of Morroc.
    ("Prontera", "Solmera"), ("Geffen", "Lunavell"), ("Payon", "Pinehollow"), ("Alberta", "Tidewick"),
    ("Morroc", "Kharsand"),
)
TOWN_ZONE_IDS = (("morocc", "kharsand"),)
ZONE_IDS = (
    ("prt_sewb1", "sol_sew01"), ("pay_dun00", "pin_dun00"), ("gef_fild00", "lun_fld00"),
    ("prt_fild01", "sol_fld01"), ("pay_fild01", "pin_fld01"), ("prt_fild05", "sol_fld05"),
    ("prt_fild08", "sol_fld08"), ("prt_fild06", "sol_fld06"), ("moc_fild01", "kha_fld01"),
    ("prt_fild02", "sol_fld02"), ("alb_fild01", "tid_fld01"),
)
CURRENCY = ("zeny", "ryn")
SERVICE = ("kafra", "waybell")

# The 20 monster kinds (game/data/monsters.py order) and the card-only / legacy-slot monsters.
# generic=True: the old name is an everyday English word, so lowercase and UPPER forms get restricted entries.
MONSTERS = (
    ("Poring", "Blobbin", False),
    ("Orc Warrior", "Grukk Brute", False),
    ("Fabre", "Wrigglet", False),
    ("Lunatic", "Moonbun", True),
    ("Drops", "Emberblob", True),
    ("Poporing", "Vexblob", False),
    ("Willow", "Bramblet", True),
    ("Rocker", "Fiddlehop", True),
    ("Spore", "Capling", True),
    ("Thief Bug", "Pilferbug", False),
    ("Zombie", "Rotling", True),
    ("Wolf", "Snarlfang", True),
    ("Skeleton", "Clackbone", True),
    ("Peco Peco", "Gallopeck", False),
    ("Hode", "Sandgulper", False),
    ("Soldier Skeleton", "Clackbone Guard", True),
    ("Bigfoot", "Grumblebear", True),
    ("Mummy", "Tombwrap", True),
    ("Golden Thief Bug", "Gildscarab", False),
    ("Eddga", "Scorchstripe", False),
    # card-only and legacy art slots
    ("Pupa", "Cocoonling", True),
    ("Andre", "Antlet", True),
    ("Chonchon", "Buzzeye", False),
    ("Picky", "Peepit", True),
    ("Roda Frog", "Croakle", False),
    ("Hornet", "Needlebuzz", True),
    ("Condor", "Cragwing", True),
    ("Baphomet Jr", "Imphorn", False),
)
# Lowercase ids of generic monsters that are also generic words in the code: keyed mode only (plus scoped lists).
KEYED_ONLY = {"spore", "skeleton"}
# Files whose quoted "drops" means the monster id (monster-id lists), never the loot block.
DROPS_ID_FILES = (
    "client/assets/build_sheets.py", "client/game/art/asset_slots.gd", "client/tests/unit/test_art_assets.gd",
    "client/tools/gen_placeholder_anim.py", "client/assets/anim/index.json", "client/assets/slots.json",
)
ITEMS = (  # old, new (Title); lower, UPPER_SNAKE forms are generated
    ("Jellopy", "Jelly Bead"), ("Fly Wing", "Flit Feather"), ("Butterfly Wing", "Homing Feather"),
    ("Phracon", "Tin Lode"), ("Emveretarcon", "Bronze Lode"), ("Oridecon", "Starsteel"),
    ("Elunium", "Moonstone"), ("Orcish Voucher", "Grukk Token"), ("Wolf Claw", "Snarlfang Claw"),
)
EXTRA = (  # (old, new, mode): forms the generators above do not produce
    ("GTB", "Gildscarab", "word"),          # abbreviation in comments and docs
    ("baphomet", "imphorn", "ident"),       # legacy slot spelled without the _jr in places
    ("Porings", "Blobbins", "ident"), ("porings", "blobbins", "ident"),
    ("Lunatics", "Moonbuns", "word"), ("Kafras", "Waybells", "ident"), ("kafras", "waybells", "ident"),
    ("draw_spore", "draw_capling", "raw"),  # the keyed `spore` entry cannot see this function name
)
# Paths renamed after the content pass: (old relative path, new relative path). Folders move whole.
PATH_RENAMES = tuple(
    ("client/assets/anim/" + old.lower().replace(" ", "_"), "client/assets/anim/" + new.lower().replace(" ", "_"))
    for old, new, _ in MONSTERS[:20]
) + (
    ("client/assets/anim/npc_kafra", "client/assets/anim/npc_waybell"),
    ("game/ro_kafra.py", "game/ro_waybell.py"),
    ("net/kafra.py", "net/waybell.py"),
    ("tests/test_ro_kafra.py", "tests/test_ro_waybell.py"),
    ("tests/test_kafra.py", "tests/test_waybell.py"),
)


def _snake(s):
    return s.lower().replace(" ", "_")


def _upper(s):
    return s.upper().replace(" ", "_")


def _build_mapping():
    """Return the explicit list of (old, new, mode, scope) entries, longest old form first."""
    out = []

    def add(old, new, mode, scope=None):
        out.append((old, new, mode, scope))

    for old, new in TOWNS:
        add(old, new, "ident")
        add(old.lower(), new.lower(), "ident")
        add(old.upper(), new.upper(), "ident")
    for old, new in TOWN_ZONE_IDS:
        add(old, new, "word")
    for old, new in ZONE_IDS:
        add(old, new, "word")
    for old, new in (CURRENCY, SERVICE):
        add(old, new, "ident")
        add(old.capitalize(), new.capitalize(), "ident")
        add(old.upper(), new.upper(), "ident")
    for old, new, generic in MONSTERS:
        if not generic:
            add(old, new, "ident")                              # Poring, Orc Warrior
            add(old.lower(), new.lower(), "ident")              # poring, orc warrior
            add(_snake(old), _snake(new), "ident")              # orc_warrior
            add(_upper(old), _upper(new), "ident")              # ORC_WARRIOR, ITEM_PORING_CARD
            continue
        add(old, new, "word")                                   # Wolf, Drops (guarded)
        add(old.lower() + " card", new.lower() + " card", "ident")
        add(_upper(old) + "_CARD", _upper(new) + "_CARD", "ident")
        if _snake(old) == "drops":                              # never keyed: world["drops"] is the loot block
            add('"drops"', '"emberblob"', "raw", DROPS_ID_FILES)
            add("monster.drops", "monster.emberblob", "raw")
            add("anim.drops", "anim.emberblob", "raw")
            add("anim/drops", "anim/emberblob", "raw")
        elif _snake(old) in KEYED_ONLY:
            add(_snake(old), _snake(new), "keyed")              # "skeleton", monster.spore, anim/spore/
        else:
            add(_snake(old), _snake(new), "ident")              # wolf, lunatic, soldier_skeleton, _willow
            add(_upper(old), _upper(new), "ident")              # WOLF, SOLDIER_SKELETON
    for old, new in ITEMS:
        add(old, new, "ident")
        add(old.lower(), new.lower(), "ident")
        add(_upper(old), _upper(new), "ident")
    for old, new, mode in EXTRA:
        add(old, new, mode)
    # longest first so "Golden Thief Bug" lands before "Thief Bug", "Soldier Skeleton" before "Skeleton"
    seen = OrderedDict()
    for e in sorted(out, key=lambda e: (-len(e[0]), e[0])):
        seen.setdefault((e[0], e[2]), e)
    return list(seen.values())


MAPPING = _build_mapping()
OLD_FORMS = tuple(e[0] for e in MAPPING)


def _pattern(old, mode):
    esc = re.escape(old)
    if mode == "ident":
        pat = r"(?<![A-Za-z0-9])" + esc + r"(?![A-Za-z0-9])"
    elif mode == "word":
        pat = r"(?<![A-Za-z0-9_])" + esc + r"(?![A-Za-z0-9_])"
    elif mode == "keyed":
        pat = r"(?<=[\"'./])" + esc + r"(?=[\"'./])"
    elif mode == "raw":
        pat = r"(?<![A-Za-z0-9])" + esc + r"(?![A-Za-z0-9_])"
    else:
        raise ValueError(mode)
    if old == "Drops":
        pat += r"(?! [a-z])"  # "Drops the path", "Drops from kills" are prose, not the monster
    return re.compile(pat)


PATTERNS = [(old, new, mode, scope, _pattern(old, mode)) for old, new, mode, scope in MAPPING]
GENERIC_TITLES = {old for old, _, generic in MONSTERS if generic}


def _rel(path, root):
    return os.path.relpath(path, root).replace("\\", "/")


def iter_files(root):
    """Yield (abs path, rel path) of every text file the rename covers, skipping the excluded dirs and files."""
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS)
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            rel = _rel(p, root)
            if os.path.splitext(fn)[1] in TEXT_EXTS and rel not in SKIP_FILES:
                yield p, rel


def _in_scope(rel, scope):
    return scope is None or any(fnmatch.fnmatch(rel, s) for s in scope)


def _line_kept(line):
    return any(tok in line for tok in SKIP_LINE_TOKENS)


def rename_text(text, rel):
    """Apply every entry to one file's text. Return (new text, Counter per old form, [ambiguous (old, line)])."""
    counts, ambiguous = Counter(), []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if _line_kept(line):
            continue
        original = line
        for old, new, mode, scope, pat in PATTERNS:
            if not _in_scope(rel, scope) or old not in line:
                continue
            line2, n = pat.subn(new, line)
            if n:
                counts[old] += n
                if old in GENERIC_TITLES or mode in ("keyed", "raw"):
                    ambiguous.append((old, original.strip()))
                line = line2
        lines[i] = line
    return "\n".join(lines), counts, ambiguous


def scan_text(text, rel):
    """Return [(line number, old form, line)] for every old form still present (the test's check)."""
    hits = []
    for i, line in enumerate(text.split("\n"), 1):
        if _line_kept(line):
            continue
        for old, new, mode, scope, pat in PATTERNS:
            if old in line and _in_scope(rel, scope) and pat.search(line):
                hits.append((i, old, line.strip()))
    return hits


def scan_paths(root):
    """Return the PATH_RENAMES whose old path still exists under root."""
    return [(o, n) for o, n in PATH_RENAMES if os.path.exists(os.path.join(root, o))]


def run(root, dry_run, verbose):
    per_entry, per_file, ambiguous, changed = Counter(), OrderedDict(), [], 0
    for path, rel in iter_files(root):
        try:
            with open(path, encoding="utf-8", newline="") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        new_text, counts, amb = rename_text(text, rel)
        if not counts:
            continue
        per_entry.update(counts)
        per_file[rel] = sum(counts.values())
        ambiguous.extend((rel, o, l) for o, l in amb)
        changed += 1
        if not dry_run and new_text != text:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(new_text)
    renames = scan_paths(root)
    if not dry_run:
        for old, new in renames:
            dst = os.path.join(root, new)
            if os.path.exists(dst):
                print(f"SKIP rename {old} -> {new}: target exists", file=sys.stderr)
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(os.path.join(root, old), dst)

    mode = "DRY RUN" if dry_run else "APPLIED"
    print(f"== rename_names {mode}: {sum(per_entry.values())} replacements in {changed} files, "
          f"{len(renames)} path renames ==")
    print("\n-- per mapping entry (old -> new: count) --")
    for old, new, m, scope in MAPPING:
        if per_entry[old]:
            print(f"{per_entry[old]:5d}  {old!r} -> {new!r}  [{m}]")
    unused = [old for old, _, _, _ in MAPPING if not per_entry[old]]
    print(f"\n-- entries with no hit today ({len(unused)}): " + ", ".join(repr(u) for u in unused))
    print("\n-- per file --")
    for rel, n in sorted(per_file.items(), key=lambda kv: -kv[1]):
        print(f"{n:5d}  {rel}")
    print("\n-- path renames --")
    for old, new in renames:
        print(f"  {old} -> {new}")
    print(f"\n-- ambiguous hits (generic words, keyed/raw forms): {len(ambiguous)} -- "
          "review by hand; each line shows the OLD text")
    if verbose:
        for rel, old, line in ambiguous:
            print(f"  {rel}: [{old}] {line[:140]}")
    else:
        by = Counter((rel, old) for rel, old, _ in ambiguous)
        for (rel, old), n in sorted(by.items()):
            print(f"  {n:4d}  {rel}  [{old}]")
        print("  (run with --verbose for the lines)")
    return per_entry, per_file, renames, ambiguous


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=REPO_ROOT, help="checkout to process (default: this repo)")
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    ap.add_argument("--verbose", action="store_true", help="print every ambiguous line")
    a = ap.parse_args(argv)
    run(os.path.abspath(a.root), a.dry_run, a.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
