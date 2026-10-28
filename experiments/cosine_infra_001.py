"""cosine_infra_001: busca por cosseno na biblioteca do fold + candidatos do pool por massa."""
import os
import sys
from pathlib import Path

if os.environ.get("KIT_ON_KAGGLE") != "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from chem import neutral_mass  # noqa: E402
from common import (FoldCtx, dedupe, finish, load_spectra, load_train, parse_args,  # noqa: E402
                    rows_by_struct, run_cv)
from spectra import cosine_one_to_many, weigh, weigh_many  # noqa: E402

PARENT = ""  # experimento de onde este partiu (preenchido pelo new_exp.py --from)
NOTES = ""   # hipótese / o que mudou em relação ao pai

PARAMS = dict(ppm=10.0, tol=0.01, top=60, power=0.5)


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
        score = np.zeros(len(cands))
        if len(rows):
            lib = weigh_many(spectra.take(rows), PARAMS["top"], PARAMS["power"])
            for qmz, qint, qmode in zip(q["mz"], q["inten"], q["ionization_mode"]):
                cos = cosine_one_to_many(*weigh(qmz, qint, PARAMS["top"], PARAMS["power"]), lib, PARAMS["tol"])
                cos[mode[rows] != qmode] = 0.0
                best = np.zeros(len(cands))
                np.maximum.at(best, owner, cos)
                score += best / len(q)
        err = np.abs(cands["mass"].to_numpy() - mass)
        rank = np.lexsort((err, -score))[:50]
        out[qid] = dedupe(cands["smiles"].to_numpy()[rank], cands["key"].to_numpy()[rank])
    return out


if __name__ == "__main__":
    args = parse_args()
    oof, pred, fold_scores = run_cv(predict, args)
    finish(oof, pred, args, notes=NOTES, parent=PARENT, params=PARAMS, fold_scores=fold_scores)
