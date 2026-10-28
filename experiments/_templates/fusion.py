"""{NAME}: fusão por posição de experimentos registrados (o ensemble que vai para a submissão)."""
import os
import sys
from pathlib import Path

if os.environ.get("KIT_ON_KAGGLE") != "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import FoldCtx, finish, fuse, lists_frame, load_member, parse_args, run_cv  # noqa: E402

PARENT = ""
NOTES = ""
# nome do experimento → peso (contagens de reports/ensemble.md / ensembles/hc_NNN.json)
MEMBERS = {}
PARAMS = dict(k0=3.0)


def predict(ctx: FoldCtx) -> dict:
    """Roda o predict de cada membro no mesmo ctx e funde as listas."""
    if not MEMBERS:
        raise ValueError("preencha MEMBERS com os pesos do hill_climb")
    ids = list(dict.fromkeys(ctx.queries["query_id"]))
    frames = [lists_frame(load_member(name).predict(ctx), ids, "query_id") for name in MEMBERS]
    fused = fuse(frames, list(MEMBERS.values()), "query_id", PARAMS["k0"])
    fused = fused[fused["rank"] >= 1].sort_values(["query_id", "rank"])
    lists = fused.groupby("query_id")["smiles"].agg(list).to_dict()
    return {i: lists.get(i, []) for i in ids}


if __name__ == "__main__":
    args = parse_args()
    oof, pred, fold_scores = run_cv(predict, args)
    # nível 2: os pesos foram escolhidos olhando os OOFs — o CV honesto é o nested do hill_climb
    finish(oof, pred, args, notes=NOTES or f"fusão {MEMBERS}", parent=PARENT,
           params=dict(PARAMS, members=MEMBERS), fold_scores=fold_scores, level=2)
