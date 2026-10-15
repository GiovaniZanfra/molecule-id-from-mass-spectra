"""O contrato do time em código: config, dados, folds, métrica, artefatos e registry.

Todo experimento importa daqui. Roda igual local e dentro de um kernel do Kaggle
(o kaggle_run.py embute src/ e comp.json no script enviado).

API principal usada pelos experimentos:
    CFG, parse_args, load_data, load_extra, get_target, get_folds,
    run_cv, finish, gpu_available, class_names, score
API usada pelas ferramentas / stacking:
    read_registry, load_oof, load_pred, register, write_submission, higher_is_better
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import metric as _m

_T0 = time.time()

ON_KAGGLE = os.environ.get("KIT_ON_KAGGLE") == "1"
KIT_ROOT = Path(os.environ.get("KIT_ROOT") or Path(__file__).resolve().parents[1])
KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
KAGGLE_WORKING = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))

NAME_RE = re.compile(r"^([a-z0-9]+)_([a-z0-9]+)_(\d{3,})$")
TOKEN_RE = re.compile(r"^[a-z0-9]+$")

DEFAULTS = {
    "task": "binary",
    "metric": "auc",
    "id": "id",
    "target": "target",
    "n_folds": 5,
    "seed": 42,
    "files": {"train": "train.csv", "test": "test.csv", "sample_submission": "sample_submission.csv"},
    "extra_files": [],
    "classes": None,
    "positive_label": None,
    "submission_format": "value",
    "submission_target": None,
    "artifacts_dir": "{repo}/artifacts",
    "kaggle_username": "",
    "kaggle_kernel_prefix": "",
    "kaggle_folds_dataset": "",
    "kaggle_kernel_extra": {},
    "debug_rows": 20000,
    "suspicious_gain": None,
}

REG_COLS = [
    "name", "family", "author", "level", "cv", "cv_std", "cv_claimed", "metric",
    "n_folds", "parent", "where", "runtime_s", "created_at", "notes",
]


# --------------------------------------------------------------------------- config

def _load_cfg() -> dict:
    path = KIT_ROOT / "comp.json"
    if not path.exists():
        raise FileNotFoundError(f"comp.json não encontrado em {KIT_ROOT}")
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(path.read_text(encoding="utf-8")))
    files = dict(DEFAULTS["files"])
    files.update(cfg.get("files") or {})
    cfg["files"] = files
    if cfg["task"] not in ("binary", "multiclass", "regression"):
        raise ValueError("comp.json: task deve ser binary, multiclass ou regression")
    if cfg["metric"] not in _m.METRICS:
        raise ValueError(f"comp.json: metric '{cfg['metric']}' não existe. Opções: {sorted(_m.METRICS)}")
    if cfg["task"] not in _m.METRICS[cfg["metric"]].tasks:
        raise ValueError(f"comp.json: metric '{cfg['metric']}' não combina com task '{cfg['task']}'")
    return cfg


CFG = _load_cfg()


def _main_checkout() -> Path:
    """Raiz do checkout principal do git (worktrees apontam para o mesmo lugar)."""
    for cmd in (["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                ["git", "rev-parse", "--git-common-dir"]):
        try:
            out = subprocess.run(cmd, cwd=KIT_ROOT, capture_output=True, text=True, timeout=5)
        except Exception:
            break
        if out.returncode == 0 and out.stdout.strip():
            common = Path(out.stdout.strip())
            if not common.is_absolute():
                common = (KIT_ROOT / common).resolve()
            if common.name == ".git":
                return common.parent
            return KIT_ROOT
    return KIT_ROOT


def _artifacts_root() -> Path:
    if ON_KAGGLE:
        return KAGGLE_WORKING / "out"
    raw = os.environ.get("KIT_ARTIFACTS") or CFG["artifacts_dir"]
    if "{repo}" in raw:
        raw = raw.replace("{repo}", str(_main_checkout()))
    raw = raw.replace("{competition}", CFG["competition"])
    return Path(os.path.expanduser(raw)).resolve()


ART = _artifacts_root()
DATA_DIR = ART / "data"
EXTRA_DIR = DATA_DIR / "extra"
FOLDS_PATH = ART / "folds.npy"
REGISTRY = ART / "registry.csv"
OOF_DIR = ART / "oof"
PRED_DIR = ART / "preds"
SCRIPTS_DIR = ART / "scripts"
LEARN_DIR = ART / "learnings"
REPORT_DIR = ART / "reports"
ENS_DIR = ART / "ensembles"
SUBS_DIR = ART / "subs"
KAGGLE_DIR = ART / "kaggle"


def paths() -> dict:
    return {
        "repo": str(KIT_ROOT), "artifacts": str(ART), "data": str(DATA_DIR),
        "registry": str(REGISTRY), "oof": str(OOF_DIR), "preds": str(PRED_DIR),
        "scripts": str(SCRIPTS_DIR), "learnings": str(LEARN_DIR),
        "reports": str(REPORT_DIR), "subs": str(SUBS_DIR),
    }


# --------------------------------------------------------------------------- dados

_CACHE: dict = {}


def _find_kaggle(fname: str, prefer: str | None = None) -> Path:
    hits = sorted(p for p in KAGGLE_INPUT.rglob(fname) if p.is_file())
    if not hits:
        raise FileNotFoundError(f"{fname} não encontrado em {KAGGLE_INPUT} (o kernel tem a fonte de dados certa?)")
    if prefer:
        pref = [h for h in hits if prefer in str(h)]
        if pref:
            return pref[0]
    return hits[0]


def _data_path(fname: str) -> Path:
    if ON_KAGGLE:
        return _find_kaggle(fname, prefer=CFG["competition"])
    p = DATA_DIR / fname
    if not p.exists():
        raise FileNotFoundError(f"{p} não existe. Coloque os arquivos da competição em {DATA_DIR} (veja README).")
    return p


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_train() -> pd.DataFrame:
    if "train" not in _CACHE:
        df = _read(_data_path(CFG["files"]["train"]))
        _CACHE["train"] = df
        _CACHE["y"] = _encode_target(df[CFG["target"]])
    return _CACHE["train"]


def load_test() -> pd.DataFrame:
    if "test" not in _CACHE:
        _CACHE["test"] = _read(_data_path(CFG["files"]["test"]))
    return _CACHE["test"]


def load_sub() -> pd.DataFrame:
    if "sub" not in _CACHE:
        _CACHE["sub"] = _read(_data_path(CFG["files"]["sample_submission"]))
    return _CACHE["sub"]


def load_data():
    """(train, test, sample_submission). Use .copy() se for alterar in place."""
    return load_train(), load_test(), load_sub()


def load_extra(fname: str) -> pd.DataFrame:
    """Arquivo extra (ex.: dataset original do playground) listado em comp.json extra_files."""
    if ON_KAGGLE:
        return _read(_find_kaggle(fname))
    p = EXTRA_DIR / fname
    if not p.exists():
        raise FileNotFoundError(f"{p} não existe. Coloque extras em {EXTRA_DIR}.")
    return _read(p)


def class_names(s: pd.Series | None = None) -> list:
    if CFG["classes"]:
        return list(CFG["classes"])
    if CFG["task"] == "regression":
        return []
    if s is None:
        s = load_train()[CFG["target"]]
    vals = s.dropna().unique().tolist()
    try:
        vals = sorted(vals)
    except TypeError:
        vals = sorted(vals, key=str)
    if CFG["task"] == "binary":
        if len(vals) != 2:
            raise ValueError(f"task binary, mas o target tem {len(vals)} valores: {vals[:10]}")
        pos = CFG["positive_label"]
        if pos is not None:
            vals = [v for v in vals if v != pos] + [pos]
    return vals


def n_classes() -> int | None:
    return len(class_names()) if CFG["task"] == "multiclass" else None


def _encode_target(s: pd.Series) -> np.ndarray:
    if CFG["task"] == "regression":
        return s.to_numpy(dtype=np.float64)
    classes = class_names(s)
    mapping = {c: i for i, c in enumerate(classes)}
    y = s.map(mapping)
    if y.isna().any():
        raise ValueError("target tem valores fora de class_names()")
    return y.to_numpy(dtype=np.int64)


def get_target(train: pd.DataFrame | None = None) -> np.ndarray:
    """Target codificado (binary: 0/1, multiclass: 0..k-1, regression: float)."""
    if train is None or train is _CACHE.get("train"):
        load_train()
        return _CACHE["y"]
    return _encode_target(train[CFG["target"]])


# --------------------------------------------------------------------------- folds e métrica

def get_folds() -> np.ndarray:
    path = _find_kaggle("folds.npy") if ON_KAGGLE else FOLDS_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} não existe. Rode: python src/make_folds.py")
    folds = np.load(path)
    n = len(load_train())
    if len(folds) != n:
        raise ValueError(f"folds.npy tem {len(folds)} linhas, train tem {n}. Folds desatualizados?")
    return folds


def score(y, pred) -> float:
    return float(_m.METRICS[CFG["metric"]].fn(y, pred))


def higher_is_better() -> bool:
    return _m.METRICS[CFG["metric"]].higher_is_better


def per_fold_scores(y, oof, folds) -> list[float]:
    return [score(y[folds == f], oof[folds == f]) for f in range(int(folds.max()) + 1)]


# --------------------------------------------------------------------------- execução

_GPU: dict = {}


def gpu_available() -> bool:
    if "ok" not in _GPU:
        ok = False
        if shutil.which("nvidia-smi"):
            try:
                ok = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10).returncode == 0
            except Exception:
                ok = False
        _GPU["ok"] = ok
    return _GPU["ok"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="fold 0 + amostra; não salva nada")
    args, _ = ap.parse_known_args(argv)
    return args


def exp_name() -> str:
    name = os.environ.get("KIT_EXP_NAME") or Path(sys.argv[0]).stem
    if not NAME_RE.match(name):
        raise ValueError(f"nome '{name}' fora do padrão <familia>_<autor>_<NNN>. Crie com src/new_exp.py")
    return name


def _len(X) -> int:
    return X.shape[0] if hasattr(X, "shape") else len(X)


def _take(X, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]


def _out_shape(n: int):
    k = n_classes()
    return (n,) if k is None else (n, k)


def run_cv(fit_predict, X, y, X_test, args=None, folds=None):
    """Loop de CV padrão do time.

    fit_predict(X_tr, y_tr, X_va, y_va, X_te) -> (pred_va, pred_te)
    Tudo que usa o target (target encoding, etc.) vai DENTRO do fit_predict.
    Retorna (oof, test_pred, fold_scores). Em --debug roda só o fold 0 numa amostra.
    """
    debug = bool(getattr(args, "debug", False))
    y = np.asarray(y)
    folds = get_folds() if folds is None else np.asarray(folds)
    n_folds = int(folds.max()) + 1
    fold_list = [0] if debug else list(range(n_folds))
    X_te = _take(X_test, np.arange(min(2000, _len(X_test)))) if debug else X_test
    oof = np.full(_out_shape(len(y)), np.nan)
    test_pred = np.zeros(_out_shape(_len(X_te)))
    rng = np.random.default_rng(CFG["seed"])
    scores = []
    for f in fold_list:
        tr = np.flatnonzero(folds != f)
        va = np.flatnonzero(folds == f)
        if debug:
            tr = np.sort(rng.choice(tr, min(len(tr), int(CFG["debug_rows"])), replace=False))
            va = np.sort(rng.choice(va, min(len(va), max(2000, int(CFG["debug_rows"]) // 4)), replace=False))
        t0 = time.time()
        p_va, p_te = fit_predict(_take(X, tr), y[tr], _take(X, va), y[va], X_te)
        p_va = np.asarray(p_va, dtype=np.float64)
        p_te = np.asarray(p_te, dtype=np.float64)
        if p_va.shape != _out_shape(len(va)) or p_te.shape != _out_shape(_len(X_te)):
            raise ValueError(f"fit_predict devolveu shapes {p_va.shape}/{p_te.shape}, esperado "
                             f"{_out_shape(len(va))}/{_out_shape(_len(X_te))}")
        oof[va] = p_va
        test_pred += p_te / len(fold_list)
        s = score(y[va], p_va)
        scores.append(s)
        print(f"fold {f}: {CFG['metric']}={s:.6f}  ({time.time() - t0:.0f}s)", flush=True)
    return oof, test_pred, scores


def finish(oof, test_pred, args=None, notes: str = "", parent: str = "", params=None,
           fold_scores=None, level: int = 1, name: str | None = None):
    """Fim obrigatório de todo experimento. Em --debug só imprime o score."""
    name = name or exp_name()
    if getattr(args, "debug", False):
        oof = np.asarray(oof, dtype=np.float64)
        mask = ~np.isnan(oof if oof.ndim == 1 else oof[:, 0])
        s = score(get_target()[mask], oof[mask])
        print(f"[debug] {name}: {CFG['metric']}={s:.6f} em {int(mask.sum())} linhas do fold 0. "
              "Nada foi salvo — rode via kaggle_run.py (ou local sem --debug) para registrar.")
        return s
    return save_artifacts(name, oof, test_pred, notes=notes, parent=parent, params=params,
                          fold_scores=fold_scores, level=level)


# --------------------------------------------------------------------------- artefatos

def _check_shape(arr: np.ndarray, n: int, what: str):
    exp = _out_shape(n)
    if arr.shape != exp:
        raise ValueError(f"{what} com shape {arr.shape}, esperado {exp}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{what} tem NaN/inf — toda linha precisa de previsão (rodou todos os folds?)")


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return repr(obj)


def save_artifacts(name: str, oof, test_pred, notes: str = "", parent: str = "", params=None,
                   fold_scores=None, level: int = 1) -> float:
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"nome '{name}' fora do padrão <familia>_<autor>_<NNN>")
    y = get_target()
    oof32 = np.asarray(oof, dtype=np.float64).astype(np.float32)
    pred32 = np.asarray(test_pred, dtype=np.float64).astype(np.float32)
    _check_shape(oof32, len(y), "OOF")
    _check_shape(pred32, len(load_test()), "test preds")
    for d in (OOF_DIR, PRED_DIR, SCRIPTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if (OOF_DIR / f"{name}.npy").exists():
        raise FileExistsError(f"{name} já tem artefatos. Artefatos nunca são sobrescritos: "
                              "crie outro experimento com src/new_exp.py --from")
    folds = get_folds()
    oof64 = oof32.astype(np.float64)
    cv = score(y, oof64)
    fs = per_fold_scores(y, oof64, folds)
    np.save(OOF_DIR / f"{name}.npy", oof32)
    np.save(PRED_DIR / f"{name}.npy", pred32)
    src = Path(sys.argv[0])
    if not ON_KAGGLE and src.suffix == ".py" and src.exists():
        shutil.copyfile(src, SCRIPTS_DIR / f"{name}.py")
    meta = {
        "name": name, "family": m.group(1), "author": m.group(2), "level": int(level),
        "metric": CFG["metric"], "cv": cv, "fold_scores": fs,
        "fold_scores_reported": _jsonable(fold_scores), "n_train": int(len(y)),
        "n_test": int(len(pred32)), "parent": parent, "notes": notes, "params": _jsonable(params),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "where": "kaggle" if ON_KAGGLE else "local", "runtime_s": round(time.time() - _T0, 1),
        "gpu": gpu_available(), "python": sys.version.split()[0],
    }
    (OOF_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if ON_KAGGLE:  # um arquivo único, fácil de achar no output do kernel
        np.savez_compressed(KAGGLE_WORKING / f"{name}_artifacts.npz", oof=oof32, preds=pred32,
                            meta=np.array(json.dumps(meta, ensure_ascii=False)))
    print(f"[ok] {name}: CV {CFG['metric']}={cv:.6f} ± {np.std(fs):.6f}  (folds: "
          + ", ".join(f"{s:.5f}" for s in fs) + ")", flush=True)
    if not ON_KAGGLE:
        register(name)
    return cv


def load_oof(name: str) -> np.ndarray:
    return np.load(OOF_DIR / f"{name}.npy").astype(np.float64)


def load_pred(name: str) -> np.ndarray:
    return np.load(PRED_DIR / f"{name}.npy").astype(np.float64)


def load_meta(name: str) -> dict:
    p = OOF_DIR / f"{name}.meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# --------------------------------------------------------------------------- registry

class FileLock:
    """Lock simples e portátil (arquivo exclusivo) para escrita concorrente no registry."""

    def __init__(self, path: Path, timeout: float = 60, stale: float = 300):
        self.path, self.timeout, self.stale = Path(str(path) + ".lock"), timeout, stale
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        self.path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.time() - t0 > self.timeout:
                    raise TimeoutError(f"lock ocupado: {self.path}")
                time.sleep(0.1)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def read_registry() -> pd.DataFrame:
    if not REGISTRY.exists():
        return pd.DataFrame(columns=REG_COLS)
    df = pd.read_csv(REGISTRY, dtype={"parent": str, "notes": str})
    for c in ("parent", "notes"):
        df[c] = df[c].fillna("")
    return df


def sort_registry(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("cv", ascending=not higher_is_better()).reset_index(drop=True)


def blocked_models(include_audit: bool = True) -> dict:
    """Modelos excluídos: blocklist do auditor (reports/blocklist.txt) + FAIL do último audit."""
    out = {}
    blk = REPORT_DIR / "blocklist.txt"
    if blk.exists():
        for line in blk.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                name, _, why = line.partition("#")
                out[name.strip()] = "blocklist: " + why.strip()
    audit = REPORT_DIR / "audit.json"
    if include_audit and audit.exists():
        for name, info in json.loads(audit.read_text(encoding="utf-8")).get("models", {}).items():
            if info.get("status") == "FAIL":
                out.setdefault(name, "audit FAIL")
    return out


def usable_models(levels=(1,)) -> pd.DataFrame:
    """Registry ordenado pelo CV, só níveis pedidos, sem modelos bloqueados."""
    reg = read_registry()
    reg = reg[reg["level"].isin(list(levels))]
    blocked = blocked_models()
    return sort_registry(reg[~reg["name"].isin(list(blocked))])


def register(name: str, where: str | None = None, quiet: bool = False) -> float:
    """Recalcula o CV a partir do OOF salvo e registra. Idempotente."""
    meta = load_meta(name)
    oof = load_oof(name)
    y = get_target()
    folds = get_folds()
    _check_shape(oof, len(y), "OOF")
    cv = score(y, oof)
    fs = per_fold_scores(y, oof, folds)
    m = NAME_RE.match(name)
    row = {
        "name": name, "family": m.group(1) if m else "", "author": m.group(2) if m else "",
        "level": int(meta.get("level", 1)), "cv": round(cv, 8), "cv_std": round(float(np.std(fs)), 8),
        "cv_claimed": meta.get("cv", np.nan), "metric": CFG["metric"], "n_folds": len(fs),
        "parent": meta.get("parent", ""), "where": where or meta.get("where", "local"),
        "runtime_s": meta.get("runtime_s", np.nan),
        "created_at": meta.get("created_at", dt.datetime.now().isoformat(timespec="seconds")),
        "notes": str(meta.get("notes", "")).replace("\n", " ")[:300],
    }
    with FileLock(REGISTRY):
        reg = read_registry()
        if name in set(reg["name"]):
            if not quiet:
                print(f"[registry] {name} já registrado")
            return float(reg.loc[reg["name"] == name, "cv"].iloc[0])
        new = pd.DataFrame([row], columns=REG_COLS)
        new.to_csv(REGISTRY, mode="a", header=not REGISTRY.exists(), index=False)
    if not quiet:
        best = sort_registry(read_registry())
        rank = int(np.flatnonzero(best["name"].to_numpy() == name)[0]) + 1
        print(f"[registry] {name}: CV={cv:.6f} → #{rank} de {len(best)}")
    return cv


# --------------------------------------------------------------------------- submissão

def write_submission(test_pred, path) -> Path:
    test_pred = np.asarray(test_pred, dtype=np.float64)
    sub = load_sub().copy()
    test = load_test()
    idc = CFG["id"]
    if idc in sub.columns and idc in test.columns and not sub[idc].equals(test[idc]):
        order = pd.Series(np.arange(len(test)), index=test[idc].to_numpy())
        test_pred = test_pred[order.loc[sub[idc].to_numpy()].to_numpy()]
    fmt = CFG["submission_format"]
    classes = class_names()
    tcol = CFG["submission_target"] or (CFG["target"] if CFG["target"] in sub.columns else sub.columns[-1])
    if fmt == "value":
        if test_pred.ndim != 1:
            raise ValueError("submission_format 'value' precisa de previsão 1-D")
        sub[tcol] = test_pred
    elif fmt == "label":
        if test_pred.ndim == 1:
            sub[tcol] = np.where(test_pred >= 0.5, classes[1], classes[0])
        else:
            sub[tcol] = np.asarray(classes, dtype=object)[test_pred.argmax(1)]
    elif fmt == "top3":
        top = np.argsort(-test_pred, axis=1)[:, :3]
        sub[tcol] = [" ".join(str(classes[i]) for i in row) for row in top]
    elif fmt == "proba_columns":
        for i, c in enumerate(classes):
            col = c if c in sub.columns else str(c)
            if col not in sub.columns:
                raise ValueError(f"coluna da classe {c} não existe no sample_submission")
            sub[col] = test_pred[:, i]
    else:
        raise ValueError(f"submission_format desconhecido: {fmt}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path
