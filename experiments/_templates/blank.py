"""{NAME}: descreva em uma linha o que este experimento testa."""
import os
import sys
from pathlib import Path

if os.environ.get("KIT_ON_KAGGLE") != "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import CFG, finish, get_target, load_data, parse_args, run_cv  # noqa: E402

PARENT = ""
NOTES = ""
PARAMS = dict(C=1.0)


def features(train: pd.DataFrame, test: pd.DataFrame):
    """Sem target aqui. Exemplo: numéricas padronizadas + one-hot das categóricas."""
    cols = [c for c in train.columns if c not in (CFG["id"], CFG["target"])]
    both = pd.concat([train[cols], test[cols]], ignore_index=True)
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(both[c])]
    num = [c for c in cols if c not in cat]
    both[num] = both[num].fillna(both[num].median())
    both[num] = (both[num] - both[num].mean()) / (both[num].std() + 1e-9)
    both = pd.get_dummies(both, columns=cat, dummy_na=True, dtype=np.float32)
    X = both.iloc[: len(train)].to_numpy(np.float32)
    X_test = both.iloc[len(train):].to_numpy(np.float32)
    return X, X_test


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    from sklearn.linear_model import LogisticRegression, Ridge

    if CFG["task"] == "regression":
        model = Ridge(alpha=1.0 / PARAMS["C"]).fit(X_tr, y_tr)
        return model.predict(X_va), model.predict(X_te)
    model = LogisticRegression(C=PARAMS["C"], max_iter=2000).fit(X_tr, y_tr)
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
