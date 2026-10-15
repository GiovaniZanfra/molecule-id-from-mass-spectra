"""{NAME}: descreva em uma linha o que este experimento testa."""
import os
import sys
from pathlib import Path

if os.environ.get("KIT_ON_KAGGLE") != "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from common import CFG, finish, get_target, gpu_available, load_data, parse_args, run_cv  # noqa: E402

PARENT = ""  # experimento de onde este partiu (preenchido pelo new_exp.py --from)
NOTES = ""   # hipótese / o que mudou em relação ao pai

PARAMS = dict(
    n_estimators=5000, learning_rate=0.03, max_depth=6, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, early_stopping_rounds=200,
)


def features(train: pd.DataFrame, test: pd.DataFrame):
    """Feature engineering SEM usar o target. Coisas com target vão no fit_predict."""
    cols = [c for c in train.columns if c not in (CFG["id"], CFG["target"])]
    X, X_test = train[cols].copy(), test[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):  # object, string ou category
            cats = pd.Categorical(pd.concat([X[c], X_test[c]], ignore_index=True)).categories
            X[c] = pd.Categorical(X[c], categories=cats)
            X_test[c] = pd.Categorical(X_test[c], categories=cats)
    return X, X_test


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    import xgboost as xgb

    kw = dict(PARAMS, tree_method="hist", device="cuda" if gpu_available() else "cpu",
              enable_categorical=True, random_state=CFG["seed"])
    if CFG["task"] == "regression":
        model = xgb.XGBRegressor(**kw)
    else:
        model = xgb.XGBClassifier(**kw)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    if CFG["task"] == "regression":
        return model.predict(X_va), model.predict(X_te)
    p_va, p_te = model.predict_proba(X_va), model.predict_proba(X_te)
    if CFG["task"] == "binary":
        return p_va[:, 1], p_te[:, 1]
    return p_va, p_te


if __name__ == "__main__":
    args = parse_args()
    train, test, _ = load_data()
    y = get_target()
    X, X_test = features(train, test)
    oof, pred, fold_scores = run_cv(fit_predict, X, y, X_test, args)
    finish(oof, pred, args, notes=NOTES, parent=PARENT, params=PARAMS, fold_scores=fold_scores)
