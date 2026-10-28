"""EDA 3 — consultas de validação: classes/subconjuntos, adutos, candidatos por consulta e o que o
fold deixa visível de cada candidata (vazamento possível via "candidata sem espectro na biblioteca")."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from chem import neutral_mass  # noqa: E402
from common import (candidate_pool, excluded_keys, get_folds, load_queries, load_structures,  # noqa: E402
                    load_train, rows_by_struct)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

q = load_queries()
st = load_structures()
m = load_train()
folds = get_folds()
order, start = rows_by_struct()
mode = m["ionization_mode"].to_numpy()

print("=== consultas por subconjunto × classe × fold")
print(pd.crosstab([q["subset"], q["cls"]], q["fold"], margins=True))
print("\nespectros por consulta:", q["n_spectra"].value_counts().sort_index().to_dict())
print("biblioteca de origem:", q["query_lib"].value_counts().to_dict())
rows = np.concatenate(q["rows"].to_numpy())
print("adutos das consultas:", m["adduct"].iloc[rows].value_counts().to_dict())
print("modo das consultas:", m["ionization_mode"].iloc[rows].value_counts(normalize=True).round(2).to_dict())
print("massa da resposta por subconjunto:")
print(q.groupby("subset")["mass"].describe()[["min", "25%", "50%", "75%", "max"]].round(0))

print("\n=== candidatas na janela de ±10 ppm (massa da consulta pelo precursor)")
recs = []
pools = {f: candidate_pool(excluded_keys(f)) for f in range(5)}
for r in q.itertuples():
    mm = m.iloc[r.rows]
    mass = float(np.nanmedian([neutral_mass(p, a) for p, a in zip(mm["precursor_mz"], mm["adduct"])]))
    w = pools[r.fold].window(mass, 10.0)
    qmodes = set(mm["ionization_mode"])
    n_lib = []
    for s in w["struct"].to_numpy():
        rr = order[start[s]:start[s + 1]]
        n_lib.append(int(((folds[rr] != r.fold) & np.isin(mode[rr], list(qmodes))).sum()))
    n_lib = np.array(n_lib)
    is_ans = (w["key"] == r.key).to_numpy()
    same_f = (w["formula"] == r.formula).to_numpy()
    recs.append({
        "subset": r.subset, "cls": r.cls, "n_cand": len(w), "ans_in": bool(is_ans.any()),
        "n_same_formula": int(same_f.sum()), "n_formulas": w["formula"].nunique(),
        "n_nolib": int((n_lib == 0).sum()),
        "ans_nolib": bool(is_ans.any() and n_lib[is_ans][0] == 0),
        "n_nolib_same_formula": int(((n_lib == 0) & same_f).sum()),
    })
d = pd.DataFrame(recs)
agg = d.groupby(["subset", "cls"]).agg(
    n=("n_cand", "size"), cand_med=("n_cand", "median"), ans_in=("ans_in", "mean"),
    same_formula_med=("n_same_formula", "median"), formulas_med=("n_formulas", "median"),
    nolib_med=("n_nolib", "median"), ans_nolib=("ans_nolib", "mean"),
    nolib_same_formula_med=("n_nolib_same_formula", "median"))
print(agg.round(2))
c2 = d[d["cls"] == 2]
print("\nclasse 2: candidatas sem espectro (mesmo modo) na biblioteca do fold, incluindo a resposta:",
      c2["n_nolib"].value_counts().sort_index().head(10).to_dict())
print("classe 2: dessas, com a mesma fórmula da resposta:",
      c2["n_nolib_same_formula"].value_counts().sort_index().head(10).to_dict())
print("classe 3: candidatas sem espectro:", d.loc[d["cls"] == 3, "n_nolib"].value_counts().sort_index().head(10).to_dict())
