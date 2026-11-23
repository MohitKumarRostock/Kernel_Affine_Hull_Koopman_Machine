#!/usr/bin/env python3
"""Validate scientific Markdown mathematics and local links (stdlib only)."""
from pathlib import Path
from urllib.parse import unquote
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "README.md",
    ROOT / "REPRODUCIBILITY.md",
    *sorted((ROOT / "docs").glob("*.md")),
]
FENCE = re.compile(r"^\s*(?P<mark>~{3,}|" + re.escape(chr(96)) + r"{3,})")
MATH = re.compile(r"(?<!\\)\$")
INLINE = re.compile(r"(?<!\\)\$([^$]*)\$")
LINK = re.compile(r"!?\[[^\]]+\]\(([^)\s]+)")


def check_document(path: Path) -> list[str]:
    errors = []
    fence_char = None
    fence_size = 0
    in_display_math = False
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        loc = f"{path.relative_to(ROOT)}:{line_no}"
        found = FENCE.match(line)
        if found:
            mark = found.group("mark")
            if fence_char is None:
                fence_char, fence_size = mark[0], len(mark)
            elif mark[0] == fence_char and len(mark) >= fence_size:
                fence_char, fence_size = None, 0
            continue
        if fence_char is not None:
            continue
        if line.strip() == "$$":
            in_display_math = not in_display_math
            continue
        if in_display_math:
            continue
        if "$$" in line:
            errors.append(f"{loc}: display math must use standalone dollar delimiters")
        if len(MATH.findall(line)) % 2:
            errors.append(f"{loc}: unmatched inline math delimiter")
        if line.lstrip().startswith("|"):
            for token in INLINE.finditer(line):
                if re.search(r"(?<!\\)\|", token.group(1)):
                    errors.append(f"{loc}: unescaped table pipe inside math")
        for token in LINK.finditer(line):
            dest = token.group(1)
            if dest.startswith(("https:", "http:", "mailto:", "#", "data:")):
                continue
            relative = unquote(dest.split("#", 1)[0].split("?", 1)[0])
            if relative:
                resolved = (path.parent / relative).resolve()
                if not resolved.is_relative_to(ROOT) or not resolved.exists():
                    errors.append(f"{loc}: missing local target {dest}")
    if fence_char is not None:
        errors.append(f"{path.relative_to(ROOT)}: unclosed fenced code block")
    if in_display_math:
        errors.append(f"{path.relative_to(ROOT)}: unclosed display mathematics")
    return errors


def main() -> int:
    errors = [error for document in DOCS for error in check_document(document)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Verified Markdown mathematics and local links in {len(DOCS)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
