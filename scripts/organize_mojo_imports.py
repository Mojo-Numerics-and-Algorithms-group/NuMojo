#!/usr/bin/env python3
"""Organize Mojo imports into the groups described in the style guide.

See `docs/developer-guide/style-guide.md` for the grouping this produces
(`Stdlib`, `External`, `NuMojo`, each under its own 80-character separator).

Usage:
    python3 scripts/organize_mojo_imports.py numojo

The script rewrites `.mojo` files under the given path by default. Use
`--check` or `--diff` for dry runs.

Sorting and grouping are safe and always applied. Dropping imported names
that appear unused is *opt-in* (`--remove-unused`), because "unused" is
decided by a text scan of the remaining source rather than by the compiler.

Output is line-width compatible with `pixi run mojo format`: a `from ... import`
that fits in 80 columns is emitted on one line, anything longer is
parenthesized with a trailing comma (which `mojo format` then leaves alone).
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

FROM_RE = re.compile(r"^from\s+(?P<module>\S+)\s+import\s+(?P<names>.+)$", re.DOTALL)
IMPORT_AS_RE = re.compile(r"^(?P<module>[^\s,]+)(?:\s+as\s+(?P<alias>\w+))?$")
IDENT_RE = re.compile(r"`[^`]+`|[A-Za-z_][A-Za-z0-9_]*")
BANNER_RE = re.compile(r"^# ===-+===\s*#?\s*$")

# The three groups from the style guide, in emission order.
GROUP_STDLIB = 0
GROUP_EXTERNAL = 1
GROUP_NUMOJO = 2
GROUP_TITLES = {
    GROUP_STDLIB: "Stdlib",
    GROUP_EXTERNAL: "External",
    GROUP_NUMOJO: "NuMojo",
}

# `mojo format` wraps at 80 columns; match it so the two tools agree.
MAX_LINE_LENGTH = 80

# The 80-character section separator from the style guide.
SEPARATOR = "# ===" + "-" * 70 + "=== #"


@dataclass(frozen=True)
class ImportedName:
    raw: str
    bound: str


@dataclass(frozen=True)
class ImportStatement:
    text: str
    kind: str
    module: str
    imported: tuple[ImportedName, ...]
    sortable: bool = True
    force_parenthesized: bool = False

    @property
    def has_wildcard(self) -> bool:
        return any(name.raw == "*" for name in self.imported)


def normalize_identifier(value: str) -> str:
    value = value.strip()
    if value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def split_top_level_commas(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in value:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
            continue
        current.append(ch)
    item = "".join(current).strip()
    if item:
        parts.append(item)
    return parts


def strip_outer_parens(value: str) -> str:
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        return value[1:-1]
    return value


def imported_bound_name(raw_name: str) -> str:
    name = raw_name.strip().rstrip(",")
    if " as " in name:
        return normalize_identifier(name.rsplit(" as ", 1)[1])
    return normalize_identifier(name)


def parse_from_import(text: str) -> ImportStatement | None:
    flattened = " ".join(line.strip() for line in text.splitlines())
    match = FROM_RE.match(flattened)
    if not match:
        return None
    module = match.group("module")
    raw_names = strip_outer_parens(match.group("names"))
    names = []
    for item in split_top_level_commas(raw_names):
        item = item.strip().rstrip(",")
        if not item:
            continue
        names.append(ImportedName(item, imported_bound_name(item)))
    return ImportStatement(
        text=text,
        kind="from",
        module=module,
        imported=tuple(names),
        # True only when the statement really spanned several lines — `text`
        # always ends in a newline, so test the stripped form.
        force_parenthesized="\n" in text.strip(),
    )


def parse_plain_import(text: str) -> ImportStatement | None:
    flattened = " ".join(line.strip() for line in text.splitlines())
    if not flattened.startswith("import "):
        return None
    names = []
    modules = []
    for item in split_top_level_commas(flattened[len("import ") :]):
        match = IMPORT_AS_RE.match(item.strip())
        if not match:
            return ImportStatement(
                text=text, kind="import", module=flattened, imported=(), sortable=False
            )
        module = match.group("module")
        alias = match.group("alias")
        modules.append(module)
        bound = alias if alias else module.split(".")[0]
        names.append(ImportedName(item.strip(), bound))
    return ImportStatement(
        text=text, kind="import", module=",".join(modules), imported=tuple(names)
    )


def parse_import_statement(text: str) -> ImportStatement | None:
    stripped = text.lstrip()
    if stripped.startswith("from "):
        return parse_from_import(text)
    if stripped.startswith("import "):
        return parse_plain_import(text)
    return None


def paren_delta(line: str) -> int:
    return line.count("(") - line.count(")")


def collect_import_statement(lines: list[str], start: int) -> tuple[str, int]:
    collected = [lines[start]]
    depth = paren_delta(lines[start])
    idx = start + 1
    while depth > 0 and idx < len(lines):
        collected.append(lines[idx])
        depth += paren_delta(lines[idx])
        idx += 1
    return "".join(collected), idx


def is_banner_comment(line: str) -> bool:
    return BANNER_RE.match(line.strip()) is not None


def is_generated_group_title(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("# "):
        return False
    return stripped[2:] in set(GROUP_TITLES.values())


def is_import_section_comment(lines: list[str], idx: int) -> bool:
    if idx + 2 >= len(lines):
        return False
    return (
        is_banner_comment(lines[idx])
        and is_generated_group_title(lines[idx + 1])
        and is_banner_comment(lines[idx + 2])
    )


def previous_import_section_start(lines: list[str], first_import: int) -> int:
    start = first_import
    idx = first_import - 1
    while idx >= 0 and lines[idx].strip() == "":
        idx -= 1
    if idx >= 2 and is_import_section_comment(lines, idx - 2):
        start = idx - 2
        idx = start - 1
        while idx >= 0 and lines[idx].strip() == "":
            start = idx
            idx -= 1
    return start


def lines_inside_triple_quoted_strings(lines: list[str]) -> set[int]:
    """Return indices of lines that fall (even partially) inside a
    triple-quoted string literal, so import-like text inside docstrings
    (e.g. example code in ```mojo fences) isn't mistaken for real imports.
    """
    inside: set[int] = set()
    in_string = False
    quote = ""
    for idx, line in enumerate(lines):
        pos = 0
        while pos < len(line):
            if not in_string:
                marker = None
                for candidate in ('"""', "'''"):
                    found = line.find(candidate, pos)
                    if found != -1 and (marker is None or found < marker[0]):
                        marker = (found, candidate)
                if marker is None:
                    break
                found, candidate = marker
                in_string = True
                quote = candidate
                pos = found + 3
            else:
                inside.add(idx)
                found = line.find(quote, pos)
                if found == -1:
                    pos = len(line)
                else:
                    in_string = False
                    pos = found + 3
        if in_string:
            inside.add(idx)
    return inside


def find_import_block(lines: list[str]) -> tuple[int, int, list[str]] | None:
    in_string = lines_inside_triple_quoted_strings(lines)
    start = None
    idx = 0
    while idx < len(lines):
        if idx in in_string:
            idx += 1
            continue
        if lines[idx].startswith("from ") or lines[idx].startswith("import "):
            start = idx
            break
        idx += 1
    if start is None:
        return None

    statements: list[str] = []
    first_import = start
    start = previous_import_section_start(lines, first_import)
    idx = start
    saw_comment = False
    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped == "":
            idx += 1
            continue
        if is_import_section_comment(lines, idx):
            idx += 3
            continue
        if stripped.startswith("#"):
            saw_comment = True
            idx += 1
            continue
        if lines[idx].startswith("from ") or lines[idx].startswith("import "):
            text, idx = collect_import_statement(lines, idx)
            statements.append(text)
            continue
        break

    if saw_comment:
        return None
    return start, idx, statements


def import_group(module: str) -> int:
    # Relative imports and the test helpers are in-tree, so they belong in
    # the same `NuMojo` block as absolute `numojo.*` imports rather than in a
    # second block under a duplicate title.
    if module.startswith("."):
        return GROUP_NUMOJO
    root = module.split(".", 1)[0].split(",", 1)[0]
    if root == "std":
        return GROUP_STDLIB
    if root == "numojo":
        return GROUP_NUMOJO
    if root == "utils_for_test":
        return GROUP_NUMOJO
    return GROUP_EXTERNAL


def sort_imported_names(names: tuple[ImportedName, ...]) -> tuple[ImportedName, ...]:
    return tuple(sorted(names, key=lambda item: item.raw.replace("`", "").lower()))


def render_parenthesized(module: str, names: tuple[ImportedName, ...]) -> str:
    body = "".join(f"    {name.raw},\n" for name in names)
    return f"from {module} import (\n{body})\n"


def render_from_import(module: str, names: tuple[ImportedName, ...]) -> str:
    names = sort_imported_names(names)
    # Measure the rendered line, not an approximation of it: the `from `,
    # ` import ` and `, ` separators all count against the 80-column budget
    # `mojo format` enforces. Getting this wrong makes the two tools undo
    # each other's work on every run.
    one_line = f"from {module} import {', '.join(name.raw for name in names)}\n"
    if len(one_line.rstrip("\n")) <= MAX_LINE_LENGTH:
        return one_line
    return render_parenthesized(module, names)


def render_import_statement(statement: ImportStatement) -> str:
    if statement.kind == "from":
        # A statement the author already wrote parenthesized keeps that shape:
        # the trailing comma is a "magic trailing comma" that `mojo format`
        # honours, so collapsing it here would just be re-expanded.
        if statement.force_parenthesized:
            return render_parenthesized(
                statement.module, sort_imported_names(statement.imported)
            )
        return render_from_import(statement.module, statement.imported)
    modules = ", ".join(name.raw for name in sort_imported_names(statement.imported))
    return f"import {modules}\n"


def strip_strings_and_comments(text: str) -> str:
    result: list[str] = []
    idx = 0
    quote: str | None = None
    triple = False
    while idx < len(text):
        ch = text[idx]
        if quote:
            if triple and text.startswith(quote * 3, idx):
                quote = None
                triple = False
                idx += 3
                result.append(" ")
                continue
            # A backslash escapes the next character, so `"a\"b"` is one
            # string, not two. Without this the scanner ends the string early
            # and starts reading code as string content (or vice versa).
            if ch == "\\":
                result.append("  ")
                idx += 2
                continue
            if not triple and ch == quote:
                quote = None
            result.append(" ")
            idx += 1
            continue
        if ch == "#":
            while idx < len(text) and text[idx] != "\n":
                result.append(" ")
                idx += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            triple = text.startswith(ch * 3, idx)
            idx += 3 if triple else 1
            result.append(" ")
            continue
        result.append(ch)
        idx += 1
    return "".join(result)


def used_identifiers(text: str) -> set[str]:
    stripped = strip_strings_and_comments(text)
    return {
        normalize_identifier(match.group(0)) for match in IDENT_RE.finditer(stripped)
    }


def remove_unused_imports(
    statements: list[ImportStatement], body: str
) -> list[ImportStatement]:
    used = used_identifiers(body)
    kept: list[ImportStatement] = []
    for statement in statements:
        if statement.has_wildcard or not statement.imported:
            kept.append(statement)
            continue
        names = tuple(name for name in statement.imported if name.bound in used)
        if not names:
            continue
        if names == statement.imported:
            kept.append(statement)
        else:
            kept.append(
                ImportStatement(
                    text=statement.text,
                    kind=statement.kind,
                    module=statement.module,
                    imported=names,
                    sortable=statement.sortable,
                    force_parenthesized=statement.force_parenthesized,
                )
            )
    return kept


def merge_from_imports(statements: list[ImportStatement]) -> list[ImportStatement]:
    merged: dict[str, ImportStatement] = {}
    output: list[ImportStatement] = []

    for statement in statements:
        if statement.kind != "from" or statement.has_wildcard:
            output.append(statement)
            continue

        existing = merged.get(statement.module)
        if existing is None:
            merged[statement.module] = statement
            output.append(statement)
            continue

        seen = {name.raw for name in existing.imported}
        names = list(existing.imported)
        for name in statement.imported:
            if name.raw in seen:
                continue
            names.append(name)
            seen.add(name.raw)

        combined = ImportStatement(
            text=existing.text,
            kind="from",
            module=existing.module,
            imported=tuple(names),
            sortable=existing.sortable and statement.sortable,
            force_parenthesized=(
                existing.force_parenthesized or statement.force_parenthesized
            ),
        )
        merged[statement.module] = combined
        output[output.index(existing)] = combined

    return output


def organized_import_block(statements: list[ImportStatement]) -> str:
    statements = merge_from_imports(statements)
    grouped: dict[int, list[ImportStatement]] = {}
    for statement in statements:
        grouped.setdefault(import_group(statement.module), []).append(statement)

    rendered_groups: list[str] = []
    for group in sorted(grouped):
        ordered = sorted(
            grouped[group],
            key=lambda stmt: (
                stmt.module.replace("`", "").lower(),
                render_import_statement(stmt).lower(),
            ),
        )
        title = GROUP_TITLES.get(group, "Imports")
        header = f"{SEPARATOR}\n# {title}\n{SEPARATOR}\n"
        rendered_groups.append(
            header + "".join(render_import_statement(stmt) for stmt in ordered)
        )
    return "\n".join(rendered_groups)


def organize_file(path: Path, *, remove_unused: bool) -> str | None:
    original = path.read_text()
    lines = original.splitlines(keepends=True)
    block = find_import_block(lines)
    if block is None:
        return None
    start, end, raw_statements = block
    statements = []
    for text in raw_statements:
        statement = parse_import_statement(text)
        if statement is None or not statement.sortable:
            return None
        statements.append(statement)

    body = "".join(lines[end:])
    # `__init__.mojo` re-exports on purpose, so nothing there is ever "unused".
    prune = remove_unused and path.name != "__init__.mojo" and bool(body.strip())
    if prune:
        statements = remove_unused_imports(statements, body)
    new_block = organized_import_block(statements)
    tail = "".join(lines[end:]).lstrip("\n")
    if new_block and tail.strip():
        new_block += "\n\n"
    head = "".join(lines[:start])
    if new_block and head.strip() and not head.endswith("\n\n"):
        head = head.rstrip("\n") + "\n\n"
    updated = head + new_block + tail
    if updated == original:
        return None
    return updated


def iter_mojo_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.mojo")))
        elif path.suffix == ".mojo":
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Top-level package/file path(s) to organize recursively.",
    )
    parser.add_argument(
        "--check", action="store_true", help="Exit non-zero if changes are needed."
    )
    parser.add_argument("--diff", action="store_true", help="Print unified diffs.")
    parser.add_argument(
        "--remove-unused",
        action="store_true",
        help=(
            "Also drop imported names that no longer appear in the file. This "
            "is decided by a text scan, not by the compiler, so review the "
            "result (pair it with --diff first)."
        ),
    )
    args = parser.parse_args()

    changed = []
    for path in iter_mojo_files(args.paths):
        updated = organize_file(path, remove_unused=args.remove_unused)
        if updated is None:
            continue
        changed.append(path)
        original = path.read_text()
        if args.diff:
            sys.stdout.writelines(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    fromfile=str(path),
                    tofile=str(path),
                )
            )
        if not args.check and not args.diff:
            path.write_text(updated)

    if args.check and changed:
        for path in changed:
            print(f"imports need organizing: {path}", file=sys.stderr)
        return 1
    if not args.diff and not args.check:
        print(f"organized imports in {len(changed)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
