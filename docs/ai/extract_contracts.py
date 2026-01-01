"""docs/ai/extract_contracts.py

Windows-safe helper script to extract high-level *contracts* from the repo for
the repo dossier:
  - FastAPI routes (@app.get/post/put/delete)
  - Celery task names (@app.task(name="..."))
  - MongoDB collection names accessed via db["..."] / mongo_db["..."]
  - File-based IO literals (open/read_csv/read_parquet/to_csv/to_parquet)

This script is intentionally best-effort and regex-based (no AST) so it can run
quickly without extra dependencies.

Outputs:
  docs/ai/_EXTRACTED_CONTRACTS.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]  # d:/Projects
AFN_ROOT = REPO_ROOT / "AlphaFusionNet"

TREE = REPO_ROOT / "docs" / "ai" / "REPO_TREE_ALL.txt"
OUT = REPO_ROOT / "docs" / "ai" / "_EXTRACTED_CONTRACTS.json"


def load_paths_from_tree(tree_path: Path) -> list[str]:
    lines = tree_path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def main() -> int:
    if not TREE.exists():
        raise SystemExit(f"Missing required file inventory: {TREE}")

    paths = load_paths_from_tree(TREE)

    apis: list[dict] = []
    celery_tasks: list[dict] = []
    mongo_collections: set[str] = set()
    file_io_literals: set[str] = set()

    route_re = re.compile(r"@app\.(get|post|put|delete)\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
    task_re = re.compile(r"@app\.task\(\s*name\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)

    # permissive pattern: db["X"], mongo_db["X"], windows_col["X"] etc.
    mongo_coll_re = re.compile(
        r"\b(?:db|mongo_db|windows_col|monthly_col|live_col|collection)\s*\[\s*['\"]([A-Za-z0-9_]+)['\"]\s*\]"
    )

    io_res = [
        re.compile(r"\bopen\(\s*['\"]([^'\"]+)['\"]"),
        re.compile(r"\bto_parquet\(\s*['\"]([^'\"]+)['\"]"),
        re.compile(r"\bto_csv\(\s*['\"]([^'\"]+)['\"]"),
        re.compile(r"\bread_parquet\(\s*['\"]([^'\"]+)['\"]"),
        re.compile(r"\bread_csv\(\s*['\"]([^'\"]+)['\"]"),
    ]

    for rel in paths:
        p = AFN_ROOT / rel
        if p.suffix.lower() != ".py":
            continue
        if not p.exists():
            # If a tree entry points to a missing file, treat as notable but don't crash.
            continue

        txt = p.read_text(encoding="utf-8", errors="ignore")

        for m in route_re.finditer(txt):
            apis.append({"file": rel, "method": m.group(1).upper(), "route": m.group(2)})

        for m in task_re.finditer(txt):
            celery_tasks.append({"file": rel, "name": m.group(1)})

        for m in mongo_coll_re.finditer(txt):
            mongo_collections.add(m.group(1))

        for rx in io_res:
            for m in rx.finditer(txt):
                val = m.group(1)
                if val and not val.lower().startswith(("http://", "https://")):
                    file_io_literals.add(val)

    # Stable ordering / dedupe
    apis = [
        {"file": f, "method": m, "route": r}
        for (f, m, r) in sorted({(a["file"], a["method"], a["route"]) for a in apis})
    ]
    celery_tasks = [
        {"file": f, "name": n}
        for (f, n) in sorted({(t["file"], t["name"]) for t in celery_tasks})
    ]

    OUT.write_text(
        json.dumps(
            {
                "apis": apis,
                "celery_tasks": celery_tasks,
                "mongo_collections": sorted(mongo_collections),
                "file_io_literals": sorted(file_io_literals),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUT} ({len(apis)} api routes, {len(celery_tasks)} celery tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

