"""EDA 2 — estruturas: sobreposição entre bibliotecas, isômeros (mesma fórmula) e candidatos por janela de massa."""
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import candidate_pool, load_structures  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

st = load_structures()
libs = st["libs"].str.split("|")
n_libs = libs.map(len)
print("estruturas:", len(st), "| em quantas bibliotecas:", n_libs.value_counts().sort_index().to_dict())

print("\n=== estruturas exclusivas de cada biblioteca e sobreposição com e180 / np")
all_libs = sorted({l for ls in libs for l in ls})
sets = {l: set(np.flatnonzero(libs.map(lambda ls, l=l: l in ls))) for l in all_libs}
rows = []
for l in all_libs:
    s = sets[l]
    only = sum(1 for i in s if n_libs.iat[i] == 1)
    rows.append({"lib": l, "structs": len(s), "only_here%": round(100 * only / len(s), 1),
                 "in_e180%": round(100 * len(s & sets["enveda-180"]) / len(s), 1)})
print(pd.DataFrame(rows).sort_values("structs", ascending=False).to_string(index=False))
pairs = Counter()
for ls in libs:
    for a, b in combinations(sorted(ls), 2):
        pairs[(a, b)] += 1
print("pares mais comuns:", pairs.most_common(8))

e180 = st["in_e180"]
print(f"\ne180: {int(e180.sum())} estruturas; com espectro público: {st.loc[e180, 'has_public'].mean():.3f}")
print(f"np-examples: {int(st['in_np'].sum())}; com espectro público: {st.loc[st['in_np'], 'has_public'].mean():.3f}; "
      f"em e180: {st.loc[st['in_np'], 'in_e180'].mean():.3f}")

print("\n=== isômeros: estruturas por fórmula")
fc = st.groupby("formula").size()
st["n_iso"] = st["formula"].map(fc)
for name, sel in [("todas", np.ones(len(st), bool)), ("e180", e180.to_numpy()), ("np", st["in_np"].to_numpy())]:
    x = st.loc[sel, "n_iso"]
    print(f"{name:6s} isômeros no train (mesma fórmula): mediana {x.median():.0f}, p75 {x.quantile(.75):.0f}, "
          f"p90 {x.quantile(.9):.0f}, único {100 * (x == 1).mean():.0f}%")

print("\n=== candidatos do pool do train numa janela de massa (sem o próprio)")
pool = candidate_pool()
rng = np.random.default_rng(0)
for name, sel in [("e180", e180.to_numpy()), ("np", st["in_np"].to_numpy()), ("outras", ~e180.to_numpy())]:
    idx = rng.choice(np.flatnonzero(sel), min(2000, int(sel.sum())), replace=False)
    for ppm in (5, 10):
        n_c, n_f, n_same = [], [], []
        for i in idx:
            w = pool.window(float(st.at[i, "mass"]), ppm)
            n_c.append(len(w) - 1)
            n_f.append(w["formula"].nunique())
            n_same.append(int((w["formula"] == st.at[i, "formula"]).sum()) - 1)
        n_c, n_f, n_same = map(np.array, (n_c, n_f, n_same))
        print(f"{name:6s} ±{ppm:2d} ppm: candidatos mediana {np.median(n_c):.0f} (p90 {np.quantile(n_c, .9):.0f}); "
              f"fórmulas na janela mediana {np.median(n_f):.0f}; isômeros mesma fórmula mediana {np.median(n_same):.0f}; "
              f"fração dos candidatos com a mesma fórmula {n_same.sum() / max(n_c.sum(), 1):.2f}")
