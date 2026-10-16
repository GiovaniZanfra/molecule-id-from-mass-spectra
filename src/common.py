"""O contrato do time em código: config, dados, folds, métrica, artefatos e registry.

Todo experimento importa daqui. Roda igual local e dentro de um kernel do Kaggle
(o kaggle_run.py embute src/ e comp.json no script enviado).

API principal usada pelos experimentos:
    CFG, parse_args, load_data, load_extra, get_target, get_folds,
    run_cv, finish, gpu_available, class_names, score
API de retrieval (CASMI): run_cv(predict, args) chama predict(ctx) uma vez por fold e uma no
teste; ctx é um FoldCtx (linhas do train utilizáveis, chaves excluídas, consultas). Também:
    load_spectra, query_frame, candidate_pool, rows_by_struct, load_structures, dedupe
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
from dataclasses import dataclass
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
    "kaggle_datasets": [],  # datasets extras anexados a todo kernel (ex.: wheel do RDKit)
    "debug_rows": 20000,
    "suspicious_gain": None,
    # retrieval: quantas estruturas de cada subconjunto entram na validação (make_folds.py)
    "validation": {"np": 250, "e180pub": 150, "e180": 500, "nplib": 450,
                   "mass_range": [150.0, 1200.0], "max_abs_ppm": 20.0,
                   # peso de cada classe simulada no CV (estimativa da composição do teste)
                   "class_weights": [0.16, 0.45, 0.39], "debug_queries": 120},
}

TASKS = ("binary", "multiclass", "regression", "retrieval")
# colunas do train sem os arrays de picos (os picos não cabem como DataFrame em 11 GB)
SPECTRUM_COLS = ("ms2_mzs", "ms2_normalized_intensities")

REG_COLS = [
    "name", "family", "author", "level", "cv", "cv_std", "cv_claimed", "metric",
    "n_folds", "parent", "where", "runtime_s", "created_at", "notes",
    "cv_c1", "cv_c2", "cv_c3",
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
    if cfg["task"] not in TASKS:
        raise ValueError(f"comp.json: task deve ser uma de {TASKS}")
    cfg["validation"] = {**DEFAULTS["validation"], **(cfg.get("validation") or {})}
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
STRUCT_PATH = ART / "structures.parquet"      # retrieval: uma linha por estrutura (chave da métrica)
TRAIN_STRUCT_PATH = ART / "train_struct.npy"  # retrieval: estrutura de cada linha do train
QUERIES_PATH = ART / "queries.parquet"        # retrieval: consultas de validação
# no Kaggle o cache fica fora de /kaggle/working (senão vira saída do kernel e é baixado)
CACHE_DIR = Path(os.environ.get("KIT_CACHE_DIR", "/tmp/kit_cache")) if ON_KAGGLE else ART / "cache"
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
        pref = [h for h in hits if prefer.lower() in str(h).lower()]
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


def _read(path: Path, columns=None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def _art_file(name: str) -> Path:
    """Arquivo da pasta de artefatos; no Kaggle vem do dataset de folds."""
    path = _find_kaggle(name) if ON_KAGGLE else ART / name
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} não existe. Rode: python src/make_folds.py")
    return Path(path)


def train_columns() -> list[str]:
    import pyarrow.parquet as pq
    return list(pq.ParquetFile(_data_path(CFG["files"]["train"])).schema_arrow.names)


def n_train() -> int:
    """Linhas do train sem carregá-lo."""
    if "train" in _CACHE:
        return len(_CACHE["train"])
    if CFG["task"] != "retrieval":
        return len(load_train())
    import pyarrow.parquet as pq
    return pq.ParquetFile(_data_path(CFG["files"]["train"])).metadata.num_rows


def load_train() -> pd.DataFrame:
    """Train inteiro (tabular) ou só os metadados, sem os arrays de picos (retrieval)."""
    if "train" not in _CACHE:
        if CFG["task"] == "retrieval":
            cols = [c for c in train_columns() if c not in SPECTRUM_COLS]
            _CACHE["train"] = _read(_data_path(CFG["files"]["train"]), columns=cols)
        else:
            df = _read(_data_path(CFG["files"]["train"]))
            _CACHE["train"] = df
            _CACHE["y"] = _encode_target(df[CFG["target"]])
    return _CACHE["train"]


def load_structures() -> pd.DataFrame:
    """retrieval: uma linha por estrutura (índice = id da estrutura), feita pelo make_folds.py."""
    if "structures" not in _CACHE:
        _CACHE["structures"] = pd.read_parquet(_art_file(STRUCT_PATH.name))
    return _CACHE["structures"]


def train_struct() -> np.ndarray:
    """retrieval: id da estrutura de cada linha do train (-1 = SMILES que não parseia)."""
    if "train_struct" not in _CACHE:
        _CACHE["train_struct"] = np.load(_art_file(TRAIN_STRUCT_PATH.name))
    return _CACHE["train_struct"]


def load_queries() -> pd.DataFrame:
    """retrieval: consultas de validação (fold, classe simulada, resposta, linhas de espectro)."""
    if "queries" not in _CACHE:
        _CACHE["queries"] = pd.read_parquet(_art_file(QUERIES_PATH.name))
    return _CACHE["queries"]


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
    """Target codificado (binary: 0/1, multiclass: 0..k-1, regression: float,
    retrieval: chave da métrica da estrutura de cada linha do train)."""
    if CFG["task"] == "retrieval":
        if train is not None and train is not _CACHE.get("train"):
            raise ValueError("retrieval: get_target só existe para o train inteiro (use train_struct())")
        if "y" not in _CACHE:
            keys = load_structures()["key"].to_numpy(dtype=object)
            idx = train_struct()
            _CACHE["y"] = np.where(idx >= 0, keys[np.maximum(idx, 0)], None)
        return _CACHE["y"]
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
    n = n_train()
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
    ap.add_argument("--submit", action="store_true",
                    help="retrieval: pula o CV, prevê o teste e grava submission.csv (rerun do Kaggle)")
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


def run_cv(fit_predict, X=None, y=None, X_test=None, args=None, folds=None):
    """Loop de CV padrão do time.

    Retrieval: run_cv(predict, args) — veja _run_cv_retrieval.

    fit_predict(X_tr, y_tr, X_va, y_va, X_te) -> (pred_va, pred_te)
    Tudo que usa o target (target encoding, etc.) vai DENTRO do fit_predict.
    Retorna (oof, test_pred, fold_scores). Em --debug roda só o fold 0 numa amostra.
    """
    if CFG["task"] == "retrieval":
        return _run_cv_retrieval(fit_predict, args if args is not None else X)
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
    if CFG["task"] == "retrieval":
        return _finish_retrieval(name, oof, test_pred, args, notes=notes, parent=parent, params=params,
                                 fold_scores=fold_scores, level=level)
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


def load_oof(name: str):
    """Tabular: array alinhado ao train. Retrieval: DataFrame query_id, rank, smiles, key."""
    if CFG["task"] == "retrieval":
        return pd.read_parquet(OOF_DIR / f"{name}.parquet")
    return np.load(OOF_DIR / f"{name}.npy").astype(np.float64)


def load_pred(name: str):
    """Tabular: array alinhado ao test. Retrieval: DataFrame molecule_id, rank, smiles, key."""
    if CFG["task"] == "retrieval":
        return pd.read_parquet(PRED_DIR / f"{name}.parquet")
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
    if CFG["task"] == "retrieval":
        return _register_retrieval(name, where=where, quiet=quiet)
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
    if CFG["task"] == "retrieval":
        return _write_submission_retrieval(test_pred, path)
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


# =========================================================================== retrieval (CASMI)
#
# Unidade = consulta (uma molécula com 1–16 espectros); previsão = lista de até 25 SMILES.
# Validação: consultas fixas de queries.parquet (make_folds.py), com classe simulada 1/2/3.
# No fold f, um experimento só pode usar como biblioteca ou dado de treino as linhas do train com
# folds != f, e não pode propor como candidata nenhuma chave de ctx.excluded (classe 3 do fold).

QUERY_COLS = ["adduct", "ionization_mode", "instrument_type", "precursor_mz", "collision_energy_ev",
              "collision_energy_orig", "collision_energy_orig_units", "base_peak_intensity"]
MAX_RANK = _m.MAX_RANK


def load_spectra():
    """Picos limpos (spectra.CLEAN) de todas as linhas do train, em mmap. Monta o cache na 1ª vez (~4 min)."""
    import spectra as _sp

    if "spectra" not in _CACHE:
        d = CACHE_DIR / f"spectra_train_v{_sp.CACHE_VERSION}"
        if not (d / "off.npy").exists():
            print(f"montando o cache de espectros do train em {d} (~4 min, só na 1ª vez)...", flush=True)
            _sp.build_cache(_data_path(CFG["files"]["train"]), d)
        _CACHE["spectra"] = _sp.load_cache(d)
    return _CACHE["spectra"]


def load_test_full() -> pd.DataFrame:
    """test.parquet com os picos (pequeno). No rerun do Kaggle é o teste oculto."""
    if "test_full" not in _CACHE:
        _CACHE["test_full"] = _read(_data_path(CFG["files"]["test"]))
    return _CACHE["test_full"]


def query_frame(split: str = "val", fold: int | None = None, limit: int | None = None) -> pd.DataFrame:
    """Um espectro por linha: query_id, spectrum_id, QUERY_COLS e os picos limpos em `mz` e `inten`.

    split="val": consultas de validação (de um fold, se dado); spectrum_id = linha do train.
    split="test": moléculas do test.parquet (query_id = molecule_id).
    limit: só `limit` consultas (sorteadas com a seed) — usado no --debug.
    """
    import spectra as _sp

    rng = np.random.default_rng(CFG["seed"])
    if split == "val":
        q = load_queries()
        if fold is not None:
            q = q[q["fold"] == fold]
        if limit and len(q) > limit:
            q = q.iloc[np.sort(rng.choice(len(q), limit, replace=False))]
        rows = np.concatenate(q["rows"].to_numpy()) if len(q) else np.zeros(0, dtype=np.int64)
        df = load_train().iloc[rows][QUERY_COLS].reset_index(drop=True)
        df.insert(0, "query_id", np.repeat(q["query_id"].to_numpy(), q["rows"].map(len).to_numpy()))
        df.insert(1, "spectrum_id", rows)
        sp = load_spectra().take(rows)
    elif split == "test":
        t = load_test_full()
        if limit:
            t = t[t[CFG["id"]].isin(pd.unique(t[CFG["id"]])[:limit])]
        t = t.reset_index(drop=True)
        df = t[[CFG["id"], "spectrum_id"] + [c for c in QUERY_COLS if c in t.columns]].rename(
            columns={CFG["id"]: "query_id"})
        sp = _sp.clean_frame(t)
    else:
        raise ValueError("split deve ser 'val' ou 'test'")
    df["mz"] = [sp.get(i)[0] for i in range(len(sp))]
    df["inten"] = [sp.get(i)[1] for i in range(len(sp))]
    return df


def excluded_keys(fold: int) -> frozenset:
    """Chaves que não podem ser candidatas no fold (classe 3 simulada). Vale para TODO pool,
    inclusive bancos externos (COCONUT, PubChem...)."""
    q = load_queries()
    return frozenset(q.loc[(q["fold"] == fold) & (q["cls"] == 3), "key"])


class Pool:
    """Estruturas candidatas ordenadas por massa: struct (id em structures.parquet), key, smiles,
    formula, mass. Só identidade e massa: contagens/bibliotecas do train vazariam quem saiu do fold."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("mass", kind="stable").reset_index(drop=True)
        self.mass = self.df["mass"].to_numpy(dtype=np.float64)

    def __len__(self) -> int:
        return len(self.df)

    def window(self, mass: float, ppm: float = 10.0) -> pd.DataFrame:
        tol = abs(mass) * ppm * 1e-6
        lo = np.searchsorted(self.mass, mass - tol, side="left")
        hi = np.searchsorted(self.mass, mass + tol, side="right")
        return self.df.iloc[lo:hi]


def candidate_pool(excluded=frozenset()) -> Pool:
    """Estruturas do train como candidatas, menos as excluídas."""
    st = load_structures()
    df = st[["key", "smiles", "formula", "mass"]].copy()
    df.insert(0, "struct", np.arange(len(st), dtype=np.int32))
    if excluded:
        df = df[~df["key"].isin(excluded)]
    return Pool(df[np.isfinite(df["mass"])])


def rows_by_struct() -> tuple[np.ndarray, np.ndarray]:
    """(order, start): linhas do train da estrutura s = order[start[s]:start[s + 1]].
    Filtre sempre por ctx.train_mask antes de usar."""
    if "rows_by_struct" not in _CACHE:
        ts = train_struct()
        order = np.argsort(ts, kind="stable")
        start = np.searchsorted(ts[order], np.arange(len(load_structures()) + 1), side="left")
        _CACHE["rows_by_struct"] = (order, start)
    return _CACHE["rows_by_struct"]


@dataclass
class FoldCtx:
    """O que um experimento de retrieval pode usar numa rodada.

    fold        0..k-1 (validação) ou -1 (teste: o train inteiro vale)
    train_mask  bool por linha do train: só essas linhas valem como biblioteca ou dado de treino
    excluded    chaves que não podem ser candidatas (classe 3 simulada do fold)
    queries     query_frame: um espectro por linha; agrupe por query_id
    """
    fold: int
    train_mask: np.ndarray
    excluded: frozenset
    queries: pd.DataFrame

    @property
    def is_test(self) -> bool:
        return self.fold < 0

    def pool(self) -> Pool:
        return candidate_pool(self.excluded)

    def train_rows(self) -> np.ndarray:
        return np.flatnonzero(self.train_mask)


def smiles_keys(smiles) -> list:
    """Chave da métrica de cada SMILES, usando as estruturas conhecidas como cache."""
    from chem import metric_key

    if "smiles_key" not in _CACHE:
        st = load_structures()
        _CACHE["smiles_key"] = dict(zip(st["smiles"], st["key"]))
        info = CACHE_DIR / "smiles_info_rdkit2026.03.3.parquet"
        if info.exists() and not ON_KAGGLE:
            si = pd.read_parquet(info, columns=["smiles", "key"]).dropna()
            _CACHE["smiles_key"].update(zip(si["smiles"], si["key"]))
    cache = _CACHE["smiles_key"]
    out = []
    for smi in smiles:
        if smi not in cache:
            cache[smi] = metric_key(smi)
        out.append(cache[smi])
    return out


def dedupe(smiles, keys=None, max_rank: int = MAX_RANK) -> list:
    """Remove SMILES repetidos pela chave da métrica (repetição só desperdiça posição) e corta em max_rank."""
    smiles = list(smiles)
    keys = smiles_keys(smiles) if keys is None else list(keys)
    seen, out = set(), []
    for smi, k in zip(smiles, keys):
        if not smi or (k is not None and k in seen):
            continue
        if k is not None:
            seen.add(k)
        out.append(smi)
        if len(out) == max_rank:
            break
    return out


def lists_frame(lists: dict, ids, id_col: str) -> pd.DataFrame:
    """{id: [smiles, ...]} → DataFrame longo (id_col, rank, smiles, key). Todo id precisa estar lá.
    Consulta sem palpite vira uma linha com rank 0 (para continuar contando como 0 no CV)."""
    ids = list(ids)
    missing = [i for i in ids if i not in lists]
    if missing:
        raise ValueError(f"predict não devolveu lista para {len(missing)} consultas, ex.: {missing[:3]}")
    recs = []
    for i in ids:
        guesses = [g.strip() for g in lists[i] if isinstance(g, str) and g.strip()]
        if len(guesses) > MAX_RANK:
            raise ValueError(f"{i}: {len(guesses)} palpites (máximo {MAX_RANK}) — use dedupe()")
        recs += [(i, r, g) for r, g in enumerate(guesses, 1)] if guesses else [(i, 0, "")]
    df = pd.DataFrame(recs, columns=[id_col, "rank", "smiles"])
    df["key"] = [None if r == 0 else k for r, k in zip(df["rank"], smiles_keys(df["smiles"].tolist()))]
    return df


def evaluate(oof: pd.DataFrame) -> dict:
    """MRR@25 por consulta (oof longo) → CV ponderado por classe, por classe, por subconjunto e por fold."""
    q = load_queries().set_index("query_id")
    q = q.loc[q.index.intersection(pd.unique(oof["query_id"]))]
    g = oof[oof["rank"] >= 1].sort_values(["query_id", "rank"]).groupby("query_id")["key"].agg(list)
    lists = [g.get(i, []) for i in q.index]
    rr = pd.Series(_m.reciprocal_ranks(q["key"].to_numpy(dtype=object), lists), index=q.index)
    w = dict(zip((1, 2, 3), CFG["validation"]["class_weights"]))

    def weighted(sel) -> float:
        by = rr[sel].groupby(q.loc[sel, "cls"]).mean()
        tot = sum(w[c] for c in by.index)
        return float(sum(w[c] * v for c, v in by.items()) / tot) if tot else float("nan")

    return {
        "cv": weighted(np.ones(len(q), dtype=bool)),
        "n_queries": int(len(q)),
        "plain": float(rr.mean()) if len(rr) else float("nan"),
        "by_class": {int(c): float(v) for c, v in rr.groupby(q["cls"]).mean().items()},
        "by_subset": {f"{s}/{c}": float(v) for (s, c), v in rr.groupby([q["subset"], q["cls"]]).mean().items()},
        "fold_scores": [weighted((q["fold"] == f).to_numpy()) for f in sorted(q["fold"].unique())],
        "top1": float((rr == 1).mean()) if len(rr) else float("nan"),
        "in_top25": float((rr > 0).mean()) if len(rr) else float("nan"),
    }


def _fmt_report(r: dict) -> str:
    cls = " ".join(f"c{c}={v:.4f}" for c, v in sorted(r["by_class"].items()))
    return (f"mrr25={r['cv']:.4f} [{cls}] sem peso {r['plain']:.4f}, top1 {r['top1']:.3f}, "
            f"no top25 {r['in_top25']:.3f}, {r['n_queries']} consultas")


def _run_cv_retrieval(predict, args=None):
    """predict(ctx: FoldCtx) -> {query_id: [smiles, ...]} melhor primeiro, até 25.

    Chamado uma vez por fold (validação) e uma no teste (ctx.fold == -1, train inteiro).
    --debug: fold 0 com poucas consultas + 30 moléculas do teste. --submit: só o teste.
    Retorna (oof, test_pred, fold_scores) — DataFrames longos.
    """
    debug = bool(getattr(args, "debug", False))
    submit = bool(getattr(args, "submit", False))
    v = CFG["validation"]
    parts, scores = [], []
    if not submit:
        folds = get_folds()
        for f in ([0] if debug else range(int(CFG["n_folds"]))):
            ctx = FoldCtx(f, folds != f, excluded_keys(f),
                          query_frame("val", f, limit=int(v["debug_queries"]) if debug else None))
            t0 = time.time()
            part = lists_frame(predict(ctx), pd.unique(ctx.queries["query_id"]), "query_id")
            parts.append(part)
            r = evaluate(part)
            scores.append(r["cv"])
            print(f"fold {f}: {_fmt_report(r)} ({time.time() - t0:.0f}s)", flush=True)
    ctx = FoldCtx(-1, np.ones(n_train(), dtype=bool), frozenset(), query_frame("test", limit=30 if debug else None))
    t0 = time.time()
    test = lists_frame(predict(ctx), pd.unique(ctx.queries["query_id"]), CFG["id"])
    print(f"teste: {ctx.queries['query_id'].nunique()} moléculas ({time.time() - t0:.0f}s)", flush=True)
    oof = pd.concat(parts, ignore_index=True) if parts else None
    return oof, test, scores


def _finish_retrieval(name, oof, test_pred, args, notes="", parent="", params=None, fold_scores=None, level=1):
    if getattr(args, "submit", False):
        path = KAGGLE_WORKING / "submission.csv" if ON_KAGGLE else SUBS_DIR / f"{name}_submission.csv"
        write_submission(test_pred, path)
        print(f"[submit] {path}")
        return path
    if getattr(args, "debug", False):
        r = evaluate(oof)
        print(f"[debug] {name}: {_fmt_report(r)}. Nada foi salvo — rode sem --debug para registrar.")
        return r["cv"]
    return save_artifacts_retrieval(name, oof, test_pred, notes=notes, parent=parent, params=params,
                                    fold_scores=fold_scores, level=level)


def save_artifacts_retrieval(name, oof: pd.DataFrame, test_pred: pd.DataFrame, notes="", parent="",
                             params=None, fold_scores=None, level=1) -> float:
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"nome '{name}' fora do padrão <familia>_<autor>_<NNN>")
    missing = set(load_queries()["query_id"]) - set(oof["query_id"])
    if missing:
        raise ValueError(f"OOF sem {len(missing)} consultas — rodou todos os folds?")
    for d in (OOF_DIR, PRED_DIR, SCRIPTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if (OOF_DIR / f"{name}.parquet").exists():
        raise FileExistsError(f"{name} já tem artefatos. Artefatos nunca são sobrescritos: "
                              "crie outro experimento com src/new_exp.py --from")
    r = evaluate(oof)
    oof.to_parquet(OOF_DIR / f"{name}.parquet", index=False)
    test_pred.to_parquet(PRED_DIR / f"{name}.parquet", index=False)
    src = Path(sys.argv[0])
    if not ON_KAGGLE and src.suffix == ".py" and src.exists():
        shutil.copyfile(src, SCRIPTS_DIR / f"{name}.py")
    meta = {
        "name": name, "family": m.group(1), "author": m.group(2), "level": int(level),
        "metric": CFG["metric"], "cv": r["cv"], "report": r,
        "fold_scores_reported": _jsonable(fold_scores), "n_queries": r["n_queries"],
        "n_test": int(test_pred[CFG["id"]].nunique()), "parent": parent, "notes": notes,
        "params": _jsonable(params), "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "where": "kaggle" if ON_KAGGLE else "local", "runtime_s": round(time.time() - _T0, 1),
        "gpu": gpu_available(), "python": sys.version.split()[0],
    }
    (OOF_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if ON_KAGGLE:  # arquivos no output do kernel, para o kaggle_run.py baixar
        oof.to_parquet(KAGGLE_WORKING / f"{name}_oof.parquet", index=False)
        test_pred.to_parquet(KAGGLE_WORKING / f"{name}_preds.parquet", index=False)
        (KAGGLE_WORKING / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[ok] {name}: CV {_fmt_report(r)} ± {np.std(r['fold_scores']):.4f}", flush=True)
    if not ON_KAGGLE:
        register(name)
    return r["cv"]


def _register_retrieval(name: str, where: str | None = None, quiet: bool = False) -> float:
    meta = load_meta(name)
    r = evaluate(load_oof(name))
    if r["n_queries"] != len(load_queries()):
        raise ValueError(f"{name}: OOF cobre {r['n_queries']} de {len(load_queries())} consultas")
    m = NAME_RE.match(name)
    bc = r["by_class"]
    row = {
        "name": name, "family": m.group(1) if m else "", "author": m.group(2) if m else "",
        "level": int(meta.get("level", 1)), "cv": round(r["cv"], 6),
        "cv_std": round(float(np.std(r["fold_scores"])), 6), "cv_claimed": meta.get("cv", np.nan),
        "metric": CFG["metric"], "n_folds": len(r["fold_scores"]), "parent": meta.get("parent", ""),
        "where": where or meta.get("where", "local"), "runtime_s": meta.get("runtime_s", np.nan),
        "created_at": meta.get("created_at", dt.datetime.now().isoformat(timespec="seconds")),
        "notes": str(meta.get("notes", "")).replace("\n", " ")[:300],
        "cv_c1": round(bc.get(1, np.nan), 6), "cv_c2": round(bc.get(2, np.nan), 6),
        "cv_c3": round(bc.get(3, np.nan), 6),
    }
    with FileLock(REGISTRY):
        reg = read_registry()
        if name in set(reg["name"]):
            if not quiet:
                print(f"[registry] {name} já registrado")
            return float(reg.loc[reg["name"] == name, "cv"].iloc[0])
        pd.DataFrame([row], columns=REG_COLS).to_csv(REGISTRY, mode="a", header=not REGISTRY.exists(), index=False)
    if not quiet:
        best = sort_registry(read_registry())
        rank = int(np.flatnonzero(best["name"].to_numpy() == name)[0]) + 1
        print(f"[registry] {name}: CV={r['cv']:.4f} → #{rank} de {len(best)}")
    return r["cv"]


def _write_submission_retrieval(test_pred: pd.DataFrame, path) -> Path:
    """Uma linha por molécula do test.parquet, SMILES separados por ';'. Molécula sem palpite
    recebe 'C' (o Kaggle exige toda molecule_id; palpite errado vale 0, como nenhum)."""
    idc = CFG["id"]
    ids = pd.unique(load_test_full()[idc])
    tp = test_pred[test_pred["rank"] >= 1].sort_values([idc, "rank"])
    g = tp.groupby(idc)["smiles"].agg(lambda s: ";".join(s.tolist()[:MAX_RANK]))
    sub = pd.DataFrame({idc: ids, CFG.get("submission_target") or "smiles": [g.get(i) or "C" for i in ids]})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path


def fuse(frames: list, weights, id_col: str = "query_id", k0: float = 3.0, max_rank: int = MAX_RANK) -> pd.DataFrame:
    """Fusão por posição de listas longas (id_col, rank, smiles, key): score(chave) = Σ w_m / (k0 + rank_m).
    Empate: melhor posição em qualquer modelo, depois o modelo de menor índice, depois a chave.
    SMILES que não parseiam (key None) ficam de fora. Mesma regra no hill_climb e no template fusion."""
    parts, ids = [], set()
    for m, (df, w) in enumerate(zip(frames, weights)):
        ids |= set(df[id_col])
        if w <= 0:
            continue
        d = df[df["rank"] >= 1].dropna(subset=["key"])[[id_col, "rank", "smiles", "key"]].copy()
        d["c"] = float(w) / (k0 + d["rank"])
        d["m"] = m
        parts.append(d)
    if parts:
        a = pd.concat(parts, ignore_index=True)
        agg = a.groupby([id_col, "key"], sort=False).agg(score=("c", "sum"), best=("rank", "min"),
                                                         first_m=("m", "min")).reset_index()
        sm = a.sort_values(["rank", "m"]).drop_duplicates([id_col, "key"])[[id_col, "key", "smiles"]]
        agg = agg.merge(sm, on=[id_col, "key"])
        agg = agg.sort_values([id_col, "score", "best", "first_m", "key"], ascending=[True, False, True, True, True])
        agg["rank"] = agg.groupby(id_col).cumcount() + 1
        out = agg.loc[agg["rank"] <= max_rank, [id_col, "rank", "smiles", "key"]]
    else:
        out = pd.DataFrame(columns=[id_col, "rank", "smiles", "key"])
    empty = sorted(ids - set(out[id_col]))
    if empty:
        out = pd.concat([out, pd.DataFrame({id_col: empty, "rank": 0, "smiles": "", "key": None})], ignore_index=True)
    return out.reset_index(drop=True)


def load_member(name: str):
    """Importa o script de um experimento registrado (para a fusão chamar o predict dele)."""
    import importlib.util

    if "members" not in _CACHE:
        _CACHE["members"] = {}
    if name not in _CACHE["members"]:
        cands = [KIT_ROOT / "members" / f"{name}.py", KIT_ROOT / "experiments" / f"{name}.py", SCRIPTS_DIR / f"{name}.py"]
        path = next((c for c in cands if c.exists()), None)
        if path is None:
            raise FileNotFoundError(f"script de {name} não encontrado em {[str(c.parent) for c in cands]}")
        spec = importlib.util.spec_from_file_location(f"member_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "predict"):
            raise AttributeError(f"{name} não tem predict(ctx) — só experimentos de retrieval entram na fusão")
        _CACHE["members"][name] = mod
    return _CACHE["members"][name]
