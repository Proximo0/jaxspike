#!/usr/bin/env python3
"""Static lint for the project's GDScript (game/**/*.gd). Standard library only.

Usage: python3 tools/lint.py

Rules:
  no-print          print( outside game/debug/
  no-random         randf( randi( randomize( (and the _range variants) anywhere in game/
  no-clock          Time.get_ticks* / OS.get_ticks* / Time.get_unix* / OS.get_unix* in
                    the PURE_PREFIXES folders (default game/logic/)
  func-return-type  named func without "-> type"
  func-param-type   named func parameter without a ": type"
  class-var-type    class-level var without a type or ":="

Prints "path:line rule message" per problem, then "LINT problems=<n>".
Exits 1 when n > 0.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = "game"
DEBUG_PREFIX = "game/debug/"
# Folders of pure game logic that must not read the wall clock. Edit to match your layout.
PURE_PREFIXES = ("game/logic/",)

PRINT_RE = re.compile(r"(?<![\w.])print\s*\(")
RANDOM_RE = re.compile(r"(?<![\w.])(randf|randi|randomize|randf_range|randi_range)\s*\(")
CLOCK_RE = re.compile(r"\b(?:Time|OS)\.get_(?:ticks|unix)\w*")
FUNC_RE = re.compile(r"^[ \t]*(?:static[ \t]+)?func[ \t]+(\w+)[ \t]*\(", re.M)
PARAM_TYPED_RE = re.compile(r"^\s*\w+\s*:\s*\S")
CLASS_VAR_RE = re.compile(r"^(?:@\w+(?:\([^)]*\))?\s+)*(?:static\s+)?var\s+\w+(.*)$")
CLASS_VAR_OK_RE = re.compile(r"^(?::=|:\s*[A-Za-z_])")


def strip_code(line: str) -> str:
    """Blank out string literals and drop the trailing comment, keeping columns."""
    out = []
    quote = ""
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                out.append("  ")
                i += 2
                continue
            if ch == quote:
                quote = ""
            out.append(" ")
        elif ch in ("'", '"'):
            quote = ch
            out.append(" ")
        elif ch == "#":
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def split_top_level(params: str) -> list:
    """Split a parameter list on commas that sit outside brackets."""
    parts = []
    depth = 0
    cur = []
    for ch in params:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p for p in parts if p.strip()]


def matching_paren(text: str, open_pos: int) -> int:
    """Index of the ')' matching the '(' at open_pos, or -1."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def check_file(rel: str, source: str, problems: list) -> None:
    lines = [strip_code(raw) for raw in source.split("\n")]
    in_debug = rel.startswith(DEBUG_PREFIX)
    pure = rel.startswith(PURE_PREFIXES)

    for lineno, code in enumerate(lines, start=1):
        if not in_debug and PRINT_RE.search(code):
            problems.append((rel, lineno, "no-print", "print() outside game/debug/; use a debug helper"))
        if RANDOM_RE.search(code):
            problems.append((rel, lineno, "no-random", "global random call; use a seeded RandomNumberGenerator"))
        if pure and CLOCK_RE.search(code):
            problems.append((rel, lineno, "no-clock", "pure logic must not read the clock; take delta or a tick"))
        if code[:1] not in (" ", "\t"):
            m = CLASS_VAR_RE.match(code)
            if m and not CLASS_VAR_OK_RE.match(m.group(1).lstrip()):
                problems.append((rel, lineno, "class-var-type", "class-level var needs a type or ':='"))

    text = "\n".join(lines)
    for m in FUNC_RE.finditer(text):
        lineno = text.count("\n", 0, m.start()) + 1
        name = m.group(1)
        open_pos = m.end() - 1
        close = matching_paren(text, open_pos)
        if close < 0:
            problems.append((rel, lineno, "func-return-type", "unterminated parameter list in %s" % name))
            continue
        for param in split_top_level(text[open_pos + 1:close]):
            if not PARAM_TYPED_RE.match(param):
                problems.append((rel, lineno, "func-param-type",
                                 "parameter '%s' in %s has no type" % (param.strip(), name)))
        if not re.match(r"\s*->", text[close + 1:close + 40]):
            problems.append((rel, lineno, "func-return-type", "func %s has no return type" % name))


def main() -> int:
    problems = []
    game_root = os.path.join(ROOT, GAME)
    files = []
    for dirpath, dirnames, filenames in os.walk(game_root):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.endswith(".gd"):
                files.append(os.path.join(dirpath, fn))
    for path in files:
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        with open(path, encoding="utf-8") as fh:
            check_file(rel, fh.read(), problems)

    problems.sort(key=lambda p: (p[0], p[1]))
    for rel, lineno, rule, message in problems:
        print("%s:%d %s %s" % (rel, lineno, rule, message))
    print("LINT problems=%d" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
