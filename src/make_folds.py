"""Cria os folds fixos do time (artifacts/folds.npy). Rode UMA vez por competição.

    python src/make_folds.py            # cria
    python src/make_folds.py --force    # recria (invalida TODOS os artefatos existentes)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, FOLDS_PATH, get_target, load_train, read_registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if FOLDS_PATH.exists() and not args.force:
        sys.exit(f"{FOLDS_PATH} já existe. Folds são fixos durante a competição. "
                 "Use --force só se souber o que está fazendo.")
    if args.force and len(read_registry()):
        print("ATENÇÃO: recriar folds invalida todos os experimentos já registrados.")
    train = load_train()
    y = get_target()
    k, seed = int(CFG["n_folds"]), int(CFG["seed"])
    strat = y if CFG["task"] != "regression" else pd.qcut(y, q=10, labels=False, duplicates="drop")
    folds = np.full(len(y), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(skf.split(np.zeros(len(y)), strat)):
        folds[va] = f
    assert (folds >= 0).all()
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(FOLDS_PATH, folds)
    meta = {"n_rows": int(len(y)), "n_folds": k, "seed": seed, "task": CFG["task"],
            "sizes": np.bincount(folds).tolist(), "created_at": dt.datetime.now().isoformat(timespec="seconds")}
    FOLDS_PATH.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] {FOLDS_PATH}: {k} folds, {len(train)} linhas")
    for f in range(k):
        yf = y[folds == f]
        dist = (f"classes {np.bincount(yf, minlength=int(y.max()) + 1).tolist()}" if CFG["task"] == "multiclass"
                else f"target médio {float(np.mean(yf)):.5f}")
        print(f"  fold {f}: {len(yf)} linhas, {dist}")


if __name__ == "__main__":
    main()
