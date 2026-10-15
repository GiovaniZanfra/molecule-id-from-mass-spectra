"""Auditoria de todos os artefatos registrados → reports/audit.md e reports/audit.json.

    python src/audit.py

FAIL (fica fora de ensemble/stack automaticamente):
  arquivo faltando, shape errado, NaN/inf, CV do arquivo ≠ registry (artefato mexido),
  nome na blocklist do auditor (reports/blocklist.txt).
WARN (o auditor investiga):
  salto suspeito sobre o melhor anterior (possível vazamento), fold muito fora da curva,
  distribuição OOF × test diferente, quase duplicado de modelo melhor, sem snapshot do script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CFG, OOF_DIR, PRED_DIR, REPORT_DIR, SCRIPTS_DIR, _out_shape, blocked_models,  # noqa: E402
                    get_folds, get_target, higher_is_better, load_test, per_fold_scores,
                    read_registry, score)


def proxy_1d(p: np.ndarray) -> np.ndarray:
    return p.max(1) if p.ndim == 2 else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-6, help="tolerância CV arquivo × registry")
    ap.add_argument("--ks", type=float, default=0.1, help="limite KS entre OOF e test preds")
    ap.add_argument("--dup", type=float, default=0.9995, help="correlação acima disso = quase duplicado")
    ap.add_argument("--corr-models", type=int, default=80)
    ap.add_argument("--sample", type=int, default=100_000)
    args = ap.parse_args()

    reg = read_registry()
    if reg.empty:
        print("registry vazio — nada a auditar")
        return
    y = get_target()
    folds = get_folds()
    n_test = len(load_test())
    sign = 1.0 if higher_is_better() else -1.0
    rng = np.random.default_rng(0)
    samp = np.sort(rng.choice(len(y), min(len(y), args.sample), replace=False))
    blocked = blocked_models(include_audit=False)
    reg = reg.sort_values("created_at", kind="stable").reset_index(drop=True)

    res: dict[str, dict] = {}
    loaded: dict[str, np.ndarray] = {}
    for _, row in reg.iterrows():
        name = row["name"]
        r = {"name": name, "cv": float(row["cv"]), "level": int(row["level"]), "status": "OK",
             "flags": [], "created_at": row["created_at"], "author": row["author"]}
        res[name] = r

        def fail(msg, r=r):
            r["status"] = "FAIL"
            r["flags"].append("FAIL: " + msg)

        f_oof, f_pred = OOF_DIR / f"{name}.npy", PRED_DIR / f"{name}.npy"
        if not f_oof.exists() or not f_pred.exists():
            fail("arquivo OOF ou preds faltando")
            continue
        oof = np.load(f_oof).astype(np.float64)
        pred = np.load(f_pred).astype(np.float64)
        if oof.shape != _out_shape(len(y)) or pred.shape != _out_shape(n_test):
            fail(f"shape OOF {oof.shape} / preds {pred.shape} inesperado")
            continue
        if not (np.isfinite(oof).all() and np.isfinite(pred).all()):
            fail("NaN/inf nos artefatos")
            continue
        cv = score(y, oof)
        r["cv_file"] = cv
        if abs(cv - float(row["cv"])) > args.tol:
            fail(f"CV do arquivo {cv:.6f} ≠ registry {float(row['cv']):.6f} (artefato alterado?)")
        claimed = row.get("cv_claimed")
        if claimed == claimed and abs(float(claimed) - float(row["cv"])) > max(args.tol, 1e-5):
            r["flags"].append(f"WARN: CV calculado no script ({float(claimed):.6f}) ≠ registry")
        fs = per_fold_scores(y, oof, folds)
        r["fold_scores"] = fs
        r["fold_spread"] = float(max(fs) - min(fs))
        ks = float(ks_2samp(proxy_1d(oof)[samp], proxy_1d(pred)[: args.sample]).statistic)
        r["ks_oof_test"] = ks
        if ks > args.ks:
            r["flags"].append(f"WARN: distribuição OOF × test diferente (KS={ks:.3f}) — vazamento no OOF ou bug no test?")
        if not (SCRIPTS_DIR / f"{name}.py").exists():
            r["flags"].append("WARN: sem snapshot do script (reprodutibilidade)")
        if name in blocked:
            fail(blocked[name])
        if r["status"] != "FAIL":
            loaded[name] = oof

    # fold fora da curva e salto suspeito (precisam de todos os modelos)
    ok = [r for r in res.values() if "fold_spread" in r]
    if ok:
        med_spread = float(np.median([r["fold_spread"] for r in ok]))
        med_std = float(np.median(reg["cv_std"]))
        jump = CFG["suspicious_gain"] if CFG["suspicious_gain"] is not None else max(med_std, 1e-9)
        best = None
        for r in sorted(ok, key=lambda r: r["created_at"]):
            if med_spread > 0 and r["fold_spread"] > 3 * med_spread and len(ok) >= 5:
                r["flags"].append(f"WARN: diferença entre folds ({r['fold_spread']:.5f}) 3× maior que o típico")
            if r["status"] != "FAIL" and r["level"] == 1:
                if best is not None and sign * (r["cv"] - best) > jump:
                    r["flags"].append(f"WARN: salto de {sign * (r['cv'] - best):+.5f} sobre o melhor anterior "
                                      f"(limite {jump:.5f}) — conferir vazamento antes de comemorar")
                best = r["cv"] if best is None else (max(best, r["cv"]) if sign > 0 else min(best, r["cv"]))

    # diversidade: correlação com modelos melhores
    names = sorted(loaded, key=lambda n: -sign * res[n]["cv"])[: args.corr_models]
    if len(names) >= 2:
        M = np.vstack([proxy_1d(loaded[n])[samp] for n in names])
        C = np.corrcoef(M)
        for i, n in enumerate(names):
            if i == 0:
                continue
            j = int(np.argmax(C[i, :i]))
            res[n]["max_corr_better"] = float(C[i, j])
            res[n]["most_similar"] = names[j]
            if C[i, j] > args.dup:
                res[n]["flags"].append(f"WARN: quase duplicado de {names[j]} (corr={C[i, j]:.5f})")

    for r in res.values():
        if r["status"] == "OK" and any(f.startswith("WARN") for f in r["flags"]):
            r["status"] = "WARN"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().isoformat(timespec="seconds")
    (REPORT_DIR / "audit.json").write_text(json.dumps({"created_at": now, "models": res}, indent=2,
                                                      ensure_ascii=False), encoding="utf-8")
    counts = {s: sum(r["status"] == s for r in res.values()) for s in ("OK", "WARN", "FAIL")}
    ranked = sorted(res.values(), key=lambda r: -sign * r["cv"])
    best_cv = ranked[0]["cv"]
    lines = [f"# Audit — {now}", "",
             f"{len(res)} modelos | OK {counts['OK']} | WARN {counts['WARN']} | FAIL {counts['FAIL']} "
             f"| métrica {CFG['metric']}", "",
             "FAIL fica fora de ensemble/stack. WARN: o auditor investiga (prompts/40_auditor.md).", "",
             "## Pendências", ""]
    pend = [r for r in ranked if r["status"] != "OK"]
    if not pend:
        lines.append("- nenhuma")
    for r in pend:
        lines.append(f"- **{r['name']}** ({r['status']}, autor {r['author']}, CV {r['cv']:.6f})")
        lines += [f"  - {f}" for f in r["flags"]]
    lines += ["", "## Todos (por CV)", "", "| modelo | CV | Δ melhor | spread folds | corr máx c/ melhor | status |",
              "|---|---|---|---|---|---|"]
    for r in ranked:
        corr = f"{r['max_corr_better']:.4f} ({r['most_similar']})" if "max_corr_better" in r else ""
        spread = f"{r['fold_spread']:.5f}" if "fold_spread" in r else ""
        lines.append(f"| {r['name']} | {r['cv']:.6f} | {r['cv'] - best_cv:+.6f} | {spread} | {corr} | {r['status']} |")
    (REPORT_DIR / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[audit] OK {counts['OK']} | WARN {counts['WARN']} | FAIL {counts['FAIL']} → {REPORT_DIR / 'audit.md'}")
    for r in pend[:15]:
        print(f"  {r['status']:4s} {r['name']}: " + " | ".join(r["flags"]))


if __name__ == "__main__":
    main()
