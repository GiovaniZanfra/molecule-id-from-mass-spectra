"""Ranking do registry (o CV oficial do time).

    python src/top.py               # top 20
    python src/top.py -n 50 --family xgb --author w1
    python src/top.py --where       # onde ficam artefatos, reports, learnings
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, blocked_models, paths, read_registry, sort_registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--family")
    ap.add_argument("--author")
    ap.add_argument("--level", type=int)
    ap.add_argument("--where", action="store_true")
    args = ap.parse_args()
    if args.where:
        print(json.dumps(paths(), indent=2))
        return
    reg = read_registry()
    if reg.empty:
        print("registry vazio")
        return
    for col in ("family", "author", "level"):
        val = getattr(args, col)
        if val is not None:
            reg = reg[reg[col] == val]
    reg = sort_registry(reg)
    noise = float(np.median(reg["cv_std"])) / np.sqrt(max(1, int(CFG["n_folds"])))
    print(f"{len(reg)} experimentos | métrica {CFG['metric']} | ruído típico ≈ {noise:.6f} (std entre folds / √k)")
    print("BLOQ = fora de ensemble/stack (blocklist do auditor ou FAIL no audit)\n")
    blocked = blocked_models()
    reg["status"] = reg["name"].map(lambda n: "BLOQ" if n in blocked else "")
    cols = ["name", "cv", "cv_std", "level", "status", "where", "parent", "notes"]
    with_notes = reg[cols].head(args.n).copy()
    with_notes["notes"] = with_notes["notes"].astype(str).str.slice(0, 60)
    print(with_notes.to_string(index=False))


if __name__ == "__main__":
    main()
