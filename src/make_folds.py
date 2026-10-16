"""Cria a validação fixa do time. Rode UMA vez por competição.

    python src/make_folds.py            # cria
    python src/make_folds.py --force    # recria (invalida TODOS os artefatos existentes)

Tabular: folds.npy estratificado por linha.

Retrieval (CASMI) — gera na pasta de artefatos:
  structures.parquet  uma linha por estrutura, identificada pela chave da métrica (chem.metric_key):
                      smiles, fórmula, massa, bibliotecas onde aparece
  train_struct.npy    id da estrutura de cada linha do train
  folds.npy           por linha do train: -1 = sempre utilizável; f = fora de treino e biblioteca no fold f
  queries.parquet     consultas de validação: resposta, linhas de espectro, fold, classe simulada
  folds.meta.json     parâmetros e contagens

Classes simuladas — o que sai de cena quando o fold f é avaliado:
  1  os espectros da biblioteca da consulta; a estrutura continua nas outras bibliotecas
  2  todos os espectros da estrutura; ela continua entre os candidatos
  3  como 2, e a estrutura sai também de todo pool de candidatos (common.excluded_keys)

Subconjuntos (quantidades em comp.json → validation):
  np       enveda-np-examples: timsTOF + produto natural (classes 1/2/3)
  e180pub  enveda-180 com espectro público (classe 1)
  e180     enveda-180 sem espectro público, parecido com a classe 2 do teste (classes 2/3)
  nplib    só em bibliotecas públicas que não são de triagem (classes 2/3)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE_DIR, CFG, FOLDS_PATH, QUERIES_PATH, STRUCT_PATH, TRAIN_STRUCT_PATH,  # noqa: E402
                    get_target, load_train, read_registry)

PUBLIC_LIBS = {"pluskal_ms2", "riken", "gnps", "massbank", "mona", "spectraverse", "msdial",
               "drug_plus", "masaryk"}
SCREENING_LIBS = {"enveda-180", "pluskal_ms2", "drug_plus"}   # química de triagem / fármacos
NP_QUERY_LIBS = ["riken", "gnps", "massbank", "mona", "msdial", "spectraverse"]
SUBSET_CLASSES = {"np": [1, 2, 3], "e180pub": [1], "e180": [2, 3], "nplib": [2, 3]}


def make_tabular(args):
    from sklearn.model_selection import StratifiedKFold

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


# --------------------------------------------------------------------------- retrieval

def smiles_info(smiles: np.ndarray) -> pd.DataFrame:
    """(chave, massa, fórmula) por SMILES único, com cache em disco (leva alguns minutos)."""
    from chem import RDKIT_VERSION, structure_infos

    cache = CACHE_DIR / f"smiles_info_rdkit{RDKIT_VERSION}.parquet"
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["smiles", "key", "mass", "formula"])
    todo = sorted(set(smiles) - set(done["smiles"]))
    if todo:
        t0 = time.time()
        print(f"calculando a chave da métrica de {len(todo):,} SMILES...", flush=True)
        info = structure_infos(todo)
        new = pd.DataFrame(info, columns=["key", "mass", "formula"])
        new.insert(0, "smiles", todo)
        done = new if done.empty else pd.concat([done, new], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        done.to_parquet(cache, index=False)
        print(f"  {time.time() - t0:.0f}s", flush=True)
    return done.set_index("smiles")


def build_structures(meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    info = smiles_info(meta["normalized_smiles"].unique())
    row_key = info["key"].reindex(meta["normalized_smiles"]).to_numpy(dtype=object)
    ok = pd.notna(row_key)
    if (~ok).any():
        print(f"aviso: {int((~ok).sum())} linhas com SMILES que não parseia ficam sem estrutura (-1)")
    df = pd.DataFrame({"key": row_key[ok], "smiles": meta["normalized_smiles"].to_numpy()[ok],
                       "lib": meta["ingest_lib"].to_numpy()[ok]})
    # SMILES representativo = o mais frequente entre as linhas da estrutura
    rep = (df.groupby(["key", "smiles"]).size().rename("n").reset_index()
           .sort_values(["key", "n", "smiles"], ascending=[True, False, True])
           .drop_duplicates("key").set_index("key")["smiles"])
    libs = df.groupby("key")["lib"].agg(lambda s: "|".join(sorted(set(s))))
    st = pd.DataFrame({"key": rep.index, "smiles": rep.to_numpy()})
    st["formula"] = info["formula"].reindex(st["smiles"]).to_numpy()
    st["mass"] = info["mass"].reindex(st["smiles"]).to_numpy(dtype=np.float64)
    st["n_spectra"] = df.groupby("key").size().reindex(st["key"]).to_numpy()
    st["libs"] = libs.reindex(st["key"]).to_numpy()
    lib_sets = st["libs"].str.split("|").map(set)
    st["has_public"] = lib_sets.map(lambda s: bool(s & PUBLIC_LIBS))
    st["in_e180"] = lib_sets.map(lambda s: "enveda-180" in s)
    st["in_np"] = lib_sets.map(lambda s: "enveda-np-examples" in s)
    st = st.sort_values("key").reset_index(drop=True)
    sid = pd.Series(np.arange(len(st), dtype=np.int32), index=st["key"])
    row_struct = np.full(len(meta), -1, dtype=np.int32)
    row_struct[ok] = sid.reindex(row_key[ok]).to_numpy()
    return st, row_struct


def precursor_ppm(meta: pd.DataFrame, row_mass: np.ndarray) -> np.ndarray:
    """Erro do precursor contra a estrutura rotulada, recalculado: o precursor_error_ppm do train
    é em módulo e está errado para [M+CH2O2-H]- (mediana ~2.000 ppm)."""
    from chem import ADDUCT_SHIFT

    shift = meta["adduct"].map(ADDUCT_SHIFT).to_numpy(dtype=np.float64)
    ion = row_mass + shift
    return (meta["precursor_mz"].to_numpy(dtype=np.float64) - ion) / ion * 1e6


def eligible_rows(meta: pd.DataFrame, row_mass: np.ndarray) -> np.ndarray:
    """Espectros que podem ser consulta: aduto do teste, precursor coerente, ao menos um pico."""
    v = CFG["validation"]
    ppm = np.abs(precursor_ppm(meta, row_mass))
    return (np.isfinite(ppm) & (ppm <= float(v["max_abs_ppm"]))
            & (meta["num_peaks"].fillna(0).to_numpy() >= 1))


def n_query_spectra(rng, n_avail: int) -> int:
    """Teste oculto: 1–16 espectros por molécula, mediana 3."""
    return int(min(n_avail, np.clip(1 + rng.poisson(2.0), 1, 16)))


def make_retrieval(args):
    v = CFG["validation"]
    k, seed = int(CFG["n_folds"]), int(CFG["seed"])
    rng = np.random.default_rng(seed)
    t0 = time.time()
    meta = load_train()
    st, row_struct = build_structures(meta)
    lo, hi = map(float, v["mass_range"])
    in_range = st["mass"].between(lo, hi).to_numpy()

    row_mass = np.where(row_struct >= 0, st["mass"].to_numpy()[np.maximum(row_struct, 0)], np.nan)
    elig = eligible_rows(meta, row_mass) & (row_struct >= 0)
    lib = meta["ingest_lib"].to_numpy()
    # linhas elegíveis por (estrutura, biblioteca)
    el = pd.DataFrame({"row": np.flatnonzero(elig), "struct": row_struct[elig], "lib": lib[elig]})
    by_struct_lib = el.groupby(["struct", "lib"])["row"].apply(np.array)
    has = lambda libname: set(el.loc[el["lib"] == libname, "struct"])  # noqa: E731

    lib_sets = st["libs"].str.split("|").map(set)
    pools = {
        "np": [(s, "enveda-np-examples") for s in sorted(has("enveda-np-examples"))],
        "e180pub": [(s, "enveda-180") for s in sorted(has("enveda-180"))
                    if st.at[s, "has_public"] and not st.at[s, "in_np"]],
        "e180": [(s, "enveda-180") for s in sorted(has("enveda-180"))
                 if not st.at[s, "has_public"] and not st.at[s, "in_np"]],
    }
    np_cands = []
    for s in np.flatnonzero(~st["in_e180"].to_numpy() & ~st["in_np"].to_numpy()):
        if lib_sets.iat[s] & SCREENING_LIBS:
            continue
        libs_ok = [l for l in NP_QUERY_LIBS if (s, l) in by_struct_lib.index]
        if libs_ok:
            np_cands.append((s, libs_ok[rng.integers(len(libs_ok))]))
    pools["nplib"] = np_cands

    rows_out, folds = [], np.full(len(meta), -1, dtype=np.int8)
    struct_lib = lib
    for subset, classes in SUBSET_CLASSES.items():
        cands = [c for c in pools[subset] if in_range[c[0]]]
        n_want = min(int(v[subset]), len(cands))
        if n_want < int(v[subset]):
            print(f"aviso: {subset} tem só {len(cands)} estruturas elegíveis (pedido {v[subset]})")
        pick = [cands[i] for i in rng.choice(len(cands), n_want, replace=False)]
        cls = np.resize(np.array(classes), n_want)
        rng.shuffle(cls)
        for c in classes:  # folds balanceados dentro de cada (subconjunto, classe)
            idx = np.flatnonzero(cls == c)
            fold_of = np.resize(np.arange(k), len(idx))
            rng.shuffle(fold_of)
            for i, f in zip(idx, fold_of):
                s, qlib = pick[i]
                avail = by_struct_lib[(s, qlib)]
                q_rows = np.sort(rng.choice(avail, n_query_spectra(rng, len(avail)), replace=False))
                mine = row_struct == s
                drop = mine & (struct_lib == qlib) if c == 1 else mine
                assert (folds[drop] == -1).all()
                folds[drop] = f
                rows_out.append({"fold": int(f), "cls": int(c), "subset": subset, "struct": int(s),
                                 "key": st.at[s, "key"], "smiles": st.at[s, "smiles"],
                                 "formula": st.at[s, "formula"], "mass": float(st.at[s, "mass"]),
                                 "query_lib": qlib, "n_spectra": int(len(q_rows)),
                                 "rows": q_rows.astype(np.int64)})
    q = pd.DataFrame(rows_out).sort_values(["fold", "subset", "cls", "struct"]).reset_index(drop=True)
    q.insert(0, "query_id", [f"q{i:05d}" for i in range(len(q))])

    # sanidade: nenhuma consulta usa linha que continua disponível no próprio fold
    for r in q.itertuples():
        assert (folds[r.rows] == r.fold).all(), r.query_id
    for d in (STRUCT_PATH, TRAIN_STRUCT_PATH, FOLDS_PATH, QUERIES_PATH):
        d.parent.mkdir(parents=True, exist_ok=True)
    st.to_parquet(STRUCT_PATH, index=False)
    np.save(TRAIN_STRUCT_PATH, row_struct)
    np.save(FOLDS_PATH, folds)
    q.to_parquet(QUERIES_PATH, index=False)

    summary = q.groupby(["subset", "cls"]).agg(queries=("query_id", "size"), spectra=("n_spectra", "sum"),
                                                med_spectra=("n_spectra", "median")).reset_index()
    meta_out = {
        "task": CFG["task"], "n_rows": int(len(meta)), "n_folds": k, "seed": seed,
        "n_structures": int(len(st)), "n_queries": int(len(q)), "validation": v,
        "rows_held_out_per_fold": np.bincount(folds[folds >= 0], minlength=k).tolist(),
        "queries_per_fold": q["fold"].value_counts().sort_index().tolist(),
        "summary": summary.to_dict(orient="records"),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    FOLDS_PATH.with_suffix(".meta.json").write_text(json.dumps(meta_out, indent=2, default=str))
    print(f"[ok] {len(st):,} estruturas, {len(q)} consultas em {k} folds ({time.time() - t0:.0f}s)")
    print(summary.to_string(index=False))
    print("linhas fora do treino por fold:", meta_out["rows_held_out_per_fold"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if FOLDS_PATH.exists() and not args.force:
        sys.exit(f"{FOLDS_PATH} já existe. Folds são fixos durante a competição. "
                 "Use --force só se souber o que está fazendo.")
    if args.force and len(read_registry()):
        print("ATENÇÃO: recriar folds invalida todos os experimentos já registrados.")
    if CFG["task"] == "retrieval":
        make_retrieval(args)
    else:
        make_tabular(args)


if __name__ == "__main__":
    main()
