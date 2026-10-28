"""EDA 4 — onde a resposta cai no ranking dos baselines (por subconjunto/classe) e quão parecido com a
resposta é o top-1 errado (Tanimoto), para c2/c3. Uso: python eda/04_oof.py cosine_boot_001 [outro ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from chem import ensure_rdkit  # noqa: E402
from common import load_oof, load_queries  # noqa: E402

pd.set_option("display.width", 220)
ensure_rdkit()
from rdkit import Chem, DataStructs  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return gen.GetFingerprint(m) if m is not None else None


q = load_queries().set_index("query_id")
for name in sys.argv[1:] or ["cosine_boot_001"]:
    oof = load_oof(name)
    hit = oof.merge(q[["key"]], left_on="query_id", right_index=True, suffixes=("", "_ans"))
    hit = hit[(hit["rank"] >= 1) & (hit["key"] == hit["key_ans"])].groupby("query_id")["rank"].min()
    q["rank"] = hit.reindex(q.index).fillna(0).astype(int)
    q["n_list"] = oof[oof["rank"] >= 1].groupby("query_id").size().reindex(q.index).fillna(0).astype(int)
    bins = pd.cut(q["rank"].replace(0, 99), [0, 1, 3, 10, 25, 100], labels=["1", "2-3", "4-10", "11-25", "fora"])
    print(f"\n=== {name}: posição da resposta (fração das consultas)")
    print(pd.crosstab([q["subset"], q["cls"]], bins, normalize="index").round(2))
    top1 = oof[oof["rank"] == 1].set_index("query_id")["smiles"]
    sims = []
    for qid, r in q.iterrows():
        a, b = fp(r["smiles"]), fp(top1.get(qid))
        sims.append(DataStructs.TanimotoSimilarity(a, b) if a is not None and b is not None else np.nan)
    q["tan_top1"] = sims
    wrong = q[q["rank"] != 1]
    print("Tanimoto(resposta, top-1) quando o top-1 está errado — mediana e fração ≥ 0,7:")
    print(wrong.groupby(["subset", "cls"])["tan_top1"].agg(
        ["median", lambda s: (s >= 0.7).mean(), "count"]).round(2))
