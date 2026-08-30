#!/usr/bin/env python3
"""Organize Mojo imports and remove unused imported names.

Usage:
    python3 scripts/organize_mojo_imports.py numojo

The script rewrites `.mojo` files under the given path by default. Use
`--check` or `--diff` for dry runs.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

IMPORT_RE = re.compile(r"^(?P<indent>\s*)(?P<kind>from|import)\s+(?P<body>.+)$")
FROM_RE = re.compile(r"^from\s+(?P<module>\S+)\s+import\s+(?P<names>.+)$", re.DOTALL)
IMPORT_AS_RE = re.compile(r"^(?P<module>[^\s,]+)(?:\s+as\s+(?P<alias>\w+))?$")
IDENT_RE = re.compile(r"`[^`]+`|[A-Za-z_][A-Za-z0-9_]*")
BANNER_RE = re.compile(r"^# ===-+===\s*#?\s*$")

GROUP_TITLES = {
    0: "Stdlib",
    1: "External",
    2: "NuMojo",
    4: "NuMojo",
}


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
        force_parenthesized="\n" in text,
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
        stripped = lines[idx].strip()
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
    if module.startswith("."):
        return 4
    root = module.split(".", 1)[0].split(",", 1)[0]
    if root == "std":
        return 0
    if root == "max":
        return 1
    if root == "numojo":
        return 2
    if root == "utils_for_test":
        return 4
    return 1


def sort_imported_names(names: tuple[ImportedName, ...]) -> tuple[ImportedName, ...]:
    return tuple(sorted(names, key=lambda item: item.raw.replace("`", "").lower()))


def render_from_import(module: str, names: tuple[ImportedName, ...]) -> str:
    names = sort_imported_names(names)
    if len(names) == 1:
        return f"from {module} import {names[0].raw}\n"
    if len(names) <= 4 and sum(len(name.raw) for name in names) + len(module) < 88:
        return f"from {module} import {', '.join(name.raw for name in names)}\n"

    body = "".join(f"    {name.raw},\n" for name in names)
    return f"from {module} import (\n{body})\n"


def render_import_statement(statement: ImportStatement) -> str:
    if statement.kind == "from":
        if statement.force_parenthesized and len(statement.imported) > 1:
            names = sort_imported_names(statement.imported)
            body = "".join(f"    {name.raw},\n" for name in names)
            return f"from {statement.module} import (\n{body})\n"
        return render_from_import(statement.module, statement.imported)
    return f"import {', '.join(name.raw for name in sort_imported_names(statement.imported))}\n"


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
    statements: list[ImportStatement], body: str, *, skip_unused: bool
) -> list[ImportStatement]:
    if skip_unused:
        return statements
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
                existing.force_parenthesized
                or statement.force_parenthesized
                or len(names) > 1
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
        header = (
            "# ===----------------------------------------------------------------------=== #\n"
            f"# {title}\n"
            "# ===----------------------------------------------------------------------=== #\n"
        )
        rendered_groups.append(
            header + "".join(render_import_statement(stmt) for stmt in ordered)
        )
    return "\n".join(rendered_groups)


def organize_file(path: Path, *, keep_unused: bool) -> str | None:
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
    skip_unused = (
        path.name == "__init__.mojo" or keep_unused or not body.strip()
    )
    statements = remove_unused_imports(statements, body, skip_unused=skip_unused)
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
        "--keep-unused",
        action="store_true",
        help="Sort imports without removing unused imported names.",
    )
    args = parser.parse_args()

    changed = []
    for path in iter_mojo_files(args.paths):
        updated = organize_file(path, keep_unused=args.keep_unused)
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
