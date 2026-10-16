"""Ensemble por hill climbing (seleção de Caruana com reposição) + submissão candidata.

    python src/hill_climb.py --nested            # recomendado: mede o CV honesto do ensemble
    python src/hill_climb.py --top 40 --levels 1,2
    python src/hill_climb.py --include a,b --exclude c

Saídas (pasta de artefatos): ensembles/hc_NNN.json (+ _oof.npy, _test.npy),
subs/hc_NNN.csv, ensembles/history.csv e reports/ensemble.md.

--nested: para cada fold f, escolhe os pesos SEM olhar o fold f e mede nele.
É esse CV que decide se vale submeter (o CV "in-sample" é otimista).

Retrieval (CASMI): o ensemble é uma fusão por posição (common.fuse: Σ peso / (k0 + rank)) das
listas OOF; os pesos são contagens da seleção de Caruana. Para submeter, crie um experimento
`python src/new_exp.py fusion --author <autor> --template fusion`, cole os pesos em MEMBERS e
mande pro Kaggle com `kaggle_run.py submit` — ele roda o predict de cada membro no teste oculto.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CFG, ENS_DIR, REGISTRY, REPORT_DIR, SUBS_DIR, blocked_models, get_folds,  # noqa: E402
                    get_target, higher_is_better, load_oof, load_pred, read_registry, score,
                    sort_registry, write_submission)


def rank01(a: np.ndarray) -> np.ndarray:
    return (rankdata(a) - 1) / max(1, len(a) - 1)


def caruana(P: np.ndarray, y: np.ndarray, max_iter: int, tol: float, sign: float):
    """P: (n, K) ou (n, K, C). Retorna contagens por modelo e o score final."""
    K = P.shape[1]
    singles = np.array([score(y, P[:, j]) for j in range(K)])
    j0 = int(np.argmax(sign * singles))
    counts = np.zeros(K, dtype=int)
    counts[j0] = 1
    S = P[:, j0].copy()
    cur = singles[j0]
    for _ in range(max_iter):
        k = counts.sum()
        cand = np.array([score(y, (S + P[:, j]) / (k + 1)) for j in range(K)])
        j = int(np.argmax(sign * cand))
        if sign * (cand[j] - cur) <= tol:
            break
        S += P[:, j]
        counts[j] += 1
        cur = cand[j]
    return counts, float(cur)


def blend(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.tensordot(P, w, axes=([1], [0]))


def main():
    if CFG["task"] == "retrieval":
        return main_retrieval()
    ap = argparse.ArgumentParser()
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--levels", default="1")
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--rows", type=int, default=200_000, help="amostra de linhas para escolher pesos")
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--no-rank", action="store_true", help="AUC: não converter previsões para rank")
    ap.add_argument("--include", default="")
    ap.add_argument("--exclude", default="")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x]
    reg = read_registry()
    if reg.empty:
        sys.exit("registry vazio")
    audit = REPORT_DIR / "audit.json"
    if not audit.exists():
        print("aviso: rode python src/audit.py antes (modelos com FAIL ficam de fora só depois do audit)")
    elif audit.stat().st_mtime < REGISTRY.stat().st_mtime:
        print("aviso: há modelos registrados depois do último audit — rode python src/audit.py")
    blocked = blocked_models()
    excl = set(x for x in args.exclude.split(",") if x)
    pool = sort_registry(reg[reg["level"].isin(levels) & ~reg["name"].isin(list(blocked) + list(excl))])
    names = pool["name"].tolist()[: args.top]
    for n in (x for x in args.include.split(",") if x):
        if n not in names and n in set(reg["name"]):
            names.append(n)
    if not names:
        sys.exit("nenhum modelo disponível")
    sign = 1.0 if higher_is_better() else -1.0
    use_rank = CFG["metric"] == "auc" and not args.no_rank

    y = get_target()
    folds = get_folds()
    oofs = [load_oof(n) for n in names]
    tests = [load_pred(n) for n in names]
    if use_rank:
        oofs = [rank01(o) for o in oofs]
        tests = [rank01(t) for t in tests]
    P = np.stack(oofs, axis=1)
    T = np.stack(tests, axis=1)
    rng = np.random.default_rng(CFG["seed"])
    cvs = reg.set_index("name").loc[names, "cv"].astype(float)
    single_best = float(cvs.max() if sign > 0 else cvs.min())
    print(f"{len(names)} candidatos | melhor modelo único CV={single_best:.6f} | rank={use_rank}")

    def select(idx):
        if len(idx) > args.rows:
            idx = np.sort(rng.choice(idx, args.rows, replace=False))
        return caruana(P[idx], y[idx], args.max_iter, args.tol, sign)

    nested_cv, nested_folds = None, []
    if args.nested:
        nested = np.zeros(P.shape[:1] + P.shape[2:])
        for f in range(int(folds.max()) + 1):
            tr, va = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            counts, _ = select(tr)
            nested[va] = blend(P[va], counts / counts.sum())
            nested_folds.append(score(y[va], nested[va]))
            print(f"  nested fold {f}: {nested_folds[-1]:.6f}")
        nested_cv = score(y, nested)
        print(f"CV nested (honesto) = {nested_cv:.6f}")

    counts, _ = select(np.arange(len(y)))
    w = counts / counts.sum()
    ens_oof, ens_test = blend(P, w), blend(T, w)
    cv_in = score(y, ens_oof)

    ENS_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = ENS_DIR / "history.csv"
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    eid = f"hc_{len(hist) + 1:03d}"
    weights = {n: round(float(wi), 4) for n, wi in sorted(zip(names, w), key=lambda t: -t[1]) if wi > 0}
    info = {"id": eid, "created_at": dt.datetime.now().isoformat(timespec="seconds"), "metric": CFG["metric"],
            "cv_insample": cv_in, "cv_nested": nested_cv, "nested_folds": nested_folds,
            "best_single": single_best, "rank": use_rank, "candidates": names, "weights": weights}
    (ENS_DIR / f"{eid}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    np.save(ENS_DIR / f"{eid}_oof.npy", ens_oof.astype(np.float32))
    np.save(ENS_DIR / f"{eid}_test.npy", ens_test.astype(np.float32))
    sub_path = write_submission(ens_test, SUBS_DIR / f"{eid}.csv")
    row = {"id": eid, "created_at": info["created_at"], "cv_insample": cv_in, "cv_nested": nested_cv,
           "best_single": single_best, "n_models": len(weights), "sub": sub_path.name}
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(hist_path, index=False)

    noise = float(np.std(nested_folds)) / np.sqrt(len(nested_folds)) if nested_folds else None
    lines = [f"# Ensemble {eid} — {info['created_at']}", "",
             f"- métrica: {CFG['metric']} | candidatos: {len(names)} | modelos com peso: {len(weights)}",
             f"- melhor modelo único: {single_best:.6f}",
             f"- CV in-sample (otimista): {cv_in:.6f}",
             f"- CV nested (honesto): {nested_cv:.6f}" if nested_cv is not None else "- CV nested: não calculado (use --nested)",
             f"- ruído ≈ {noise:.6f} (std entre folds nested / √k)" if noise else "",
             f"- submissão candidata: `{sub_path}`", "", "## Pesos", "", "| modelo | peso |", "|---|---|"]
    lines += [f"| {n} | {wi:.3f} |" for n, wi in weights.items()]
    try:
        hist_md = hist.to_markdown(index=False)
    except ImportError:
        hist_md = "```\n" + hist.to_string(index=False) + "\n```"
    lines += ["", "## Histórico", "", hist_md]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "ensemble.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {eid}: in-sample {cv_in:.6f}" + (f" | nested {nested_cv:.6f}" if nested_cv is not None else "")
          + f" | {len(weights)} modelos → {sub_path}")
    if len(hist) > 1 and nested_cv is not None and pd.notna(hist["cv_nested"].iloc[-2]):
        prev = float(hist["cv_nested"].iloc[-2])
        delta = sign * (nested_cv - prev)
        verdict = "vale submeter" if noise and delta > noise else "dentro do ruído — não gaste submissão"
        print(f"vs ensemble anterior (nested {prev:.6f}): {delta:+.6f} → {verdict}")



# --------------------------------------------------------------------------- retrieval

class FusionData:
    """Listas OOF de K modelos como matrizes (modelo × par consulta-chave), para avaliar a fusão
    com qualquer vetor de pesos sem refazer DataFrames. Desempate igual ao de common.fuse."""

    def __init__(self, oofs: list, k0: float):
        from common import load_queries

        q = load_queries()
        self.cls, self.fold = q["cls"].to_numpy(), q["fold"].to_numpy()
        self.nq, self.K = len(q), len(oofs)
        qidx = pd.Series(np.arange(self.nq), index=q["query_id"])
        parts = []
        for m, o in enumerate(oofs):
            d = o[o["rank"] >= 1].dropna(subset=["key"])[["query_id", "key", "rank"]].copy()
            d["m"] = m
            parts.append(d)
        a = pd.concat(parts, ignore_index=True)
        pairs = a[["query_id", "key"]].drop_duplicates().reset_index(drop=True)
        pairs["pid"] = np.arange(len(pairs))
        a = a.merge(pairs, on=["query_id", "key"])
        P = len(pairs)
        self.C = np.zeros((self.K, P))
        self.R = np.full((self.K, P), np.inf)
        self.C[a["m"].to_numpy(), a["pid"].to_numpy()] = 1.0 / (k0 + a["rank"].to_numpy())
        self.R[a["m"].to_numpy(), a["pid"].to_numpy()] = a["rank"].to_numpy()
        self.pq = qidx.loc[pairs["query_id"]].to_numpy()
        self.key_rank = pd.factorize(pairs["key"], sort=True)[0]
        truth = q["key"].to_numpy(dtype=object)
        self.is_truth = pairs["key"].to_numpy(dtype=object) == truth[self.pq]
        self.weights = dict(zip((1, 2, 3), CFG["validation"]["class_weights"]))

    def rr(self, w: np.ndarray) -> np.ndarray:
        sel = np.flatnonzero(w > 0)
        out = np.zeros(self.nq)
        if len(sel) == 0:
            return out
        present = np.isfinite(self.R[sel])
        valid = present.any(0)
        score = (w[sel, None] * self.C[sel]).sum(0)
        best = np.where(present, self.R[sel], np.inf).min(0)
        first_m = sel[np.argmax(present, axis=0)]
        idx = np.flatnonzero(valid)
        order = idx[np.lexsort((self.key_rank[idx], first_m[idx], best[idx], -score[idx], self.pq[idx]))]
        pq = self.pq[order]
        starts = np.searchsorted(pq, np.arange(self.nq))
        rank = np.arange(len(order)) - starts[pq] + 1
        hit = self.is_truth[order] & (rank <= MAX_RANK_RET)
        out[pq[hit]] = 1.0 / rank[hit]
        return out

    def cv(self, rr: np.ndarray, mask: np.ndarray) -> float:
        by = pd.Series(rr[mask]).groupby(self.cls[mask]).mean()
        tot = sum(self.weights[c] for c in by.index)
        return float(sum(self.weights[c] * v for c, v in by.items()) / tot) if tot else float("nan")


MAX_RANK_RET = 25


def caruana_retrieval(fd: FusionData, mask: np.ndarray, max_iter: int, tol: float):
    singles = np.array([fd.cv(fd.rr(np.eye(fd.K)[j]), mask) for j in range(fd.K)])
    counts = np.zeros(fd.K)
    counts[int(np.argmax(singles))] = 1
    cur = float(singles.max())
    for _ in range(max_iter):
        cand = np.array([fd.cv(fd.rr(counts + np.eye(fd.K)[j]), mask) for j in range(fd.K)])
        j = int(np.argmax(cand))
        if cand[j] - cur <= tol:
            break
        counts[j] += 1
        cur = float(cand[j])
    return counts, cur


def main_retrieval():
    from common import evaluate, fuse, load_queries

    ap = argparse.ArgumentParser()
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--levels", default="1")
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--k0", type=float, default=3.0, help="fusão: peso / (k0 + rank)")
    ap.add_argument("--include", default="")
    ap.add_argument("--exclude", default="")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x]
    reg = read_registry()
    if reg.empty:
        sys.exit("registry vazio")
    audit = REPORT_DIR / "audit.json"
    if not audit.exists():
        print("aviso: rode python src/audit.py antes (modelos com FAIL ficam de fora só depois do audit)")
    elif audit.stat().st_mtime < REGISTRY.stat().st_mtime:
        print("aviso: há modelos registrados depois do último audit — rode python src/audit.py")
    blocked = blocked_models()
    excl = set(x for x in args.exclude.split(",") if x)
    pool = sort_registry(reg[reg["level"].isin(levels) & ~reg["name"].isin(list(blocked) + list(excl))])
    names = pool["name"].tolist()[: args.top]
    for n in (x for x in args.include.split(",") if x):
        if n not in names and n in set(reg["name"]):
            names.append(n)
    if not names:
        sys.exit("nenhum modelo disponível")

    oofs = [load_oof(n) for n in names]
    tests = [load_pred(n) for n in names]
    fd = FusionData(oofs, args.k0)
    allq = np.ones(fd.nq, dtype=bool)
    single_best = float(reg.set_index("name").loc[names, "cv"].max())
    print(f"{len(names)} candidatos | melhor modelo único CV={single_best:.4f}")

    nested_cv, nested_folds, nested_cls = None, [], {}
    if args.nested:
        rr_nested = np.zeros(fd.nq)
        for f in sorted(set(fd.fold)):
            counts, _ = caruana_retrieval(fd, fd.fold != f, args.max_iter, args.tol)
            va = fd.fold == f
            rr_nested[va] = fd.rr(counts)[va]
            nested_folds.append(fd.cv(rr_nested, va))
            print(f"  nested fold {f}: {nested_folds[-1]:.4f}  pesos {dict(zip(names, counts.astype(int).tolist()))}")
        nested_cv = fd.cv(rr_nested, allq)
        nested_cls = {int(c): float(rr_nested[fd.cls == c].mean()) for c in sorted(set(fd.cls))}
        print(f"CV nested (honesto) = {nested_cv:.4f} {nested_cls}")

    counts, cv_in = caruana_retrieval(fd, allq, args.max_iter, args.tol)
    w = counts / counts.sum()
    ens_oof = fuse(oofs, w, "query_id", args.k0)
    ens_test = fuse(tests, w, CFG["id"], args.k0)
    rep = evaluate(ens_oof)
    if abs(rep["cv"] - cv_in) > 1e-9:
        print(f"aviso: CV da fusão {rep['cv']:.6f} ≠ CV vetorizado {cv_in:.6f} (desempate divergente?)")

    ENS_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = ENS_DIR / "history.csv"
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    eid = f"hc_{len(hist) + 1:03d}"
    members = {n: int(c) for n, c in sorted(zip(names, counts), key=lambda t: -t[1]) if c > 0}
    info = {"id": eid, "created_at": dt.datetime.now().isoformat(timespec="seconds"), "metric": CFG["metric"],
            "k0": args.k0, "cv_insample": rep["cv"], "cv_nested": nested_cv, "nested_folds": nested_folds,
            "nested_by_class": nested_cls, "by_class": rep["by_class"], "best_single": single_best,
            "candidates": names, "members": members}
    (ENS_DIR / f"{eid}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    ens_oof.to_parquet(ENS_DIR / f"{eid}_oof.parquet", index=False)
    ens_test.to_parquet(ENS_DIR / f"{eid}_test.parquet", index=False)
    sub_path = write_submission(ens_test, SUBS_DIR / f"{eid}.csv")
    row = {"id": eid, "created_at": info["created_at"], "cv_insample": rep["cv"], "cv_nested": nested_cv,
           "best_single": single_best, "n_models": len(members), "sub": sub_path.name}
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(hist_path, index=False)

    noise = float(np.std(nested_folds)) / np.sqrt(len(nested_folds)) if nested_folds else None
    lines = [f"# Ensemble {eid} — {info['created_at']}", "",
             f"- métrica: {CFG['metric']} (fusão por posição, k0={args.k0}) | candidatos: {len(names)} | "
             f"membros: {len(members)}",
             f"- melhor modelo único: {single_best:.4f}",
             f"- CV in-sample (otimista): {rep['cv']:.4f} {rep['by_class']}",
             f"- CV nested (honesto): {nested_cv:.4f} {nested_cls}" if nested_cv is not None
             else "- CV nested: não calculado (use --nested)",
             f"- ruído ≈ {noise:.4f} (std entre folds nested / √k)" if noise else "",
             f"- CSV só do teste visível (formato): `{sub_path}`", "",
             "## Para submeter (competição de código)", "",
             "```", f"python src/new_exp.py fusion --author <autor> --template fusion",
             f"# no script: MEMBERS = {json.dumps(members)}; PARAMS k0 = {args.k0}",
             "python src/kaggle_run.py submit experiments/fusion_<autor>_NNN.py", "```", "",
             "## Pesos", "", "| modelo | contagem |", "|---|---|"]
    lines += [f"| {n} | {c} |" for n, c in members.items()]
    try:
        hist_md = hist.to_markdown(index=False)
    except ImportError:
        hist_md = "```\n" + hist.to_string(index=False) + "\n```"
    lines += ["", "## Histórico", "", hist_md]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "ensemble.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {eid}: in-sample {rep['cv']:.4f}" + (f" | nested {nested_cv:.4f}" if nested_cv is not None else "")
          + f" | membros {members} → {REPORT_DIR / 'ensemble.md'}")
    if len(hist) > 1 and nested_cv is not None and pd.notna(hist["cv_nested"].iloc[-2]):
        prev = float(hist["cv_nested"].iloc[-2])
        verdict = "vale submeter" if noise and nested_cv - prev > noise else "dentro do ruído — não gaste submissão"
        print(f"vs ensemble anterior (nested {prev:.4f}): {nested_cv - prev:+.4f} → {verdict}")


if __name__ == "__main__":
    main()
