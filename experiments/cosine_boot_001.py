"""cosine_boot_001: cosseno na biblioteca do fold + propagação por análogos (Tanimoto) dentro da janela de massa."""
import os
import sys
from pathlib import Path

if os.environ.get("KIT_ON_KAGGLE") != "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from chem import ensure_rdkit, neutral_mass  # noqa: E402
from common import (FoldCtx, dedupe, finish, load_spectra, load_train, parse_args,  # noqa: E402
                    rows_by_struct, run_cv)
from spectra import cosine_one_to_many, weigh, weigh_many  # noqa: E402

PARENT = "cosine_infra_001"
NOTES = ("candidata sem espectro (ou com cosseno baixo) herda cos(q,a)·Tanimoto(c,a)^gamma da melhor "
         "candidata a da janela que tem espectro; score = max(próprio, propagado). Ataca a classe 2.")

PARAMS = dict(ppm=10.0, tol=0.01, top=60, power=0.5, fp_radius=2, fp_bits=2048, gamma=1.0)

_FP: dict = {}


def fingerprints(structs, smiles):
    """Morgan (bit vector) por id de estrutura, com cache."""
    ensure_rdkit()
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    if "gen" not in _FP:
        _FP["gen"] = rdFingerprintGenerator.GetMorganGenerator(radius=PARAMS["fp_radius"], fpSize=PARAMS["fp_bits"])
    out = []
    for s, smi in zip(structs, smiles):
        if s not in _FP:
            mol = Chem.MolFromSmiles(smi)
            _FP[s] = _FP["gen"].GetFingerprint(mol) if mol is not None else None
        out.append(_FP[s])
    return out


def tanimoto(fa, fb) -> np.ndarray:
    from rdkit import DataStructs

    T = np.zeros((len(fa), len(fb)))
    ok_b = [j for j, f in enumerate(fb) if f is not None]
    if not ok_b:
        return T
    fb_ok = [fb[j] for j in ok_b]
    for i, f in enumerate(fa):
        if f is not None:
            T[i, ok_b] = DataStructs.BulkTanimotoSimilarity(f, fb_ok)
    return T


def predict(ctx: FoldCtx) -> dict:
    """{query_id: [smiles, ...]} melhor primeiro.

    Regras do fold: biblioteca e dados de treino só das linhas com ctx.train_mask;
    nenhuma chave de ctx.excluded como candidata (ctx.pool() já tira).
    """
    mode = load_train()["ionization_mode"].to_numpy()
    spectra = load_spectra()
    order, start = rows_by_struct()
    pool = ctx.pool()
    out = {}
    for qid, q in ctx.queries.groupby("query_id", sort=False):
        mass = float(np.nanmedian([neutral_mass(p, a) for p, a in zip(q["precursor_mz"], q["adduct"])]))
        cands = pool.window(mass, PARAMS["ppm"]) if np.isfinite(mass) else pool.df.iloc[:0]
        if cands.empty:
            out[qid] = []
            continue
        # biblioteca: espectros permitidos das candidatas, no(s) modo(s) de ionização da consulta
        modes = set(q["ionization_mode"])
        rows, owner = [], []
        for k, s in enumerate(cands["struct"].to_numpy()):
            r = order[start[s]:start[s + 1]]
            r = r[ctx.train_mask[r] & np.isin(mode[r], list(modes))]
            rows.append(r)
            owner.append(np.full(len(r), k))
        rows, owner = np.concatenate(rows), np.concatenate(owner)
        own = np.zeros(len(cands))
        if len(rows):
            lib = weigh_many(spectra.take(rows), PARAMS["top"], PARAMS["power"])
            for qmz, qint, qmode in zip(q["mz"], q["inten"], q["ionization_mode"]):
                cos = cosine_one_to_many(*weigh(qmz, qint, PARAMS["top"], PARAMS["power"]), lib, PARAMS["tol"])
                cos[mode[rows] != qmode] = 0.0
                best = np.zeros(len(cands))
                np.maximum.at(best, owner, cos)
                own += best / len(q)
        score = own.copy()
        has = np.flatnonzero(np.bincount(owner, minlength=len(cands)) > 0) if len(rows) else np.zeros(0, int)
        if len(has):
            fps = fingerprints(cands["struct"].to_numpy(), cands["smiles"].to_numpy())
            T = tanimoto(fps, [fps[j] for j in has])          # candidatas × candidatas com espectro
            T[has, np.arange(len(has))] = 0.0                 # o próprio já está em `own`
            prop = (own[has][None, :] * T ** PARAMS["gamma"]).max(axis=1)
            score = np.maximum(own, prop)
        err = np.abs(cands["mass"].to_numpy() - mass)
        rank = np.lexsort((err, -score))[:50]
        out[qid] = dedupe(cands["smiles"].to_numpy()[rank], cands["key"].to_numpy()[rank])
    return out


if __name__ == "__main__":
    args = parse_args()
    oof, pred, fold_scores = run_cv(predict, args)
    finish(oof, pred, args, notes=NOTES, parent=PARENT, params=PARAMS, fold_scores=fold_scores)
