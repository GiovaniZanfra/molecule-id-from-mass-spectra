"""{NAME}: stacking nível 2 sobre OOFs do registry (roda LOCAL, não no Kaggle)."""
import os
import sys
from pathlib import Path

if os.environ.get("KIT_ON_KAGGLE") != "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from common import (CFG, finish, get_target, load_oof, load_pred, parse_args,  # noqa: E402
                    run_cv, usable_models)

PARENT = ""
NOTES = ""
PARAMS = dict(top=30, C=0.1)
MODELS = []  # deixe vazio para usar os PARAMS["top"] melhores de nível 1; ou liste nomes


def pick_models():
    if MODELS:
        return list(MODELS)
    return usable_models(levels=(1,))["name"].tolist()[: PARAMS["top"]]


def to_feat(p):
    p = np.asarray(p, dtype=np.float64)
    if CFG["task"] == "binary":
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))[:, None]
    return p if p.ndim == 2 else p[:, None]


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    from sklearn.linear_model import LogisticRegression, Ridge

    if CFG["task"] == "regression":
        model = Ridge(alpha=1.0 / PARAMS["C"]).fit(X_tr, y_tr)
        return model.predict(X_va), model.predict(X_te)
    model = LogisticRegression(C=PARAMS["C"], max_iter=5000).fit(X_tr, y_tr)
    p_va, p_te = model.predict_proba(X_va), model.predict_proba(X_te)
    if CFG["task"] == "binary":
        return p_va[:, 1], p_te[:, 1]
    return p_va, p_te


if __name__ == "__main__":
    args = parse_args()
    names = pick_models()
    print(f"stack sobre {len(names)} modelos: {names[:10]}{' ...' if len(names) > 10 else ''}")
    X = np.hstack([to_feat(load_oof(n)) for n in names])
    X_test = np.hstack([to_feat(load_pred(n)) for n in names])
    y = get_target()
    oof, pred, fold_scores = run_cv(fit_predict, X, y, X_test, args)
    finish(oof, pred, args, notes=NOTES or f"stack {len(names)} modelos", parent=PARENT,
           params=dict(PARAMS, models=names), fold_scores=fold_scores, level=2)
