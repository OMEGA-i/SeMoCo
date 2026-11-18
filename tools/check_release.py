"""Audit a source distribution for machine paths, artifacts and forbidden text."""
import argparse
import ast
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--forbid-term", action="append", default=[])
    args = parser.parse_args()
    issues, files = [], 0
    ignored = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist"}
    extensions = {".py", ".md", ".yaml", ".toml", ".json", ".in", ""}
    for path in sorted(args.root.rglob("*")):
        rel = path.relative_to(args.root)
        if not path.is_file() or any(part in ignored or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.suffix not in extensions:
            issues.append(dict(file=str(rel), issue="unexpected binary or artifact"))
            continue
        files += 1
        text = path.read_text()
        for term in args.forbid_term:
            if term.casefold() in text.casefold():
                issues.append(dict(file=str(rel), issue="forbidden marker"))
        if path.suffix == ".py":
            tree = ast.parse(text, filename=str(rel))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value
                    if value.startswith(chr(47)) or re.match(r"^[A-Za-z]:[\\/]", value):
                        issues.append(dict(file=str(rel), line=node.lineno, issue="absolute path literal"))
        # Numeric machine addresses have no role in the portable core.
        if re.search(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", text):
            issues.append(dict(file=str(rel), issue="numeric network address"))
    print(json.dumps(dict(files=files, issues=issues), indent=2))
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
