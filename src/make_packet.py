"""Monta um pacote + prompt para consultar um modelo Pro no navegador (ChatGPT Pro, Claude etc.).

    python src/make_packet.py "Escreva um experimento XGB melhor que xgb_w1_004"
    python src/make_packet.py "..." --top 3 --include experiments/nn_w2_002.py

Saída (pasta de artefatos): packets/<data>_<slug>/packet.zip e PROMPT.md.
Você sobe o zip + cola o PROMPT.md numa conversa NOVA (turno único). O script que
voltar vira experiments/<familia>_pro_<NNN>.py (crie com new_exp.py e cole o código).
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ART, CFG, KIT_ROOT, REPORT_DIR, SCRIPTS_DIR, usable_models  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="a tarefa, numa frase concreta")
    ap.add_argument("--top", type=int, default=3, help="quantos scripts dos melhores modelos incluir")
    ap.add_argument("--include", nargs="*", default=[])
    args = ap.parse_args()

    files: list[tuple[Path, str]] = []

    def add(path: Path, arc: str):
        if path.exists():
            files.append((path, arc))

    tpl = "retrieval" if CFG["task"] == "retrieval" else "gbdt"
    for rel in ("AGENTS.md", "comp.json", "LEARNINGS.md", "notes/eda.md", "notes/competition.md",
                "notes/research.md", "src/common.py", "src/metric.py", f"experiments/_templates/{tpl}.py"):
        add(KIT_ROOT / rel, rel)
    add(REPORT_DIR / "audit.md", "reports/audit.md")
    add(REPORT_DIR / "ensemble.md", "reports/ensemble.md")
    top = usable_models(levels=(1,)).head(args.top)
    for name in top["name"]:
        add(SCRIPTS_DIR / f"{name}.py", f"top_models/{name}.py")
    for inc in args.include:
        p = Path(inc)
        add(p if p.is_absolute() else KIT_ROOT / p, f"extra/{Path(inc).name}")

    slug = re.sub(r"[^a-z0-9]+", "-", args.task.lower())[:40].strip("-")
    out = ART / "packets" / f"{dt.datetime.now():%Y%m%d_%H%M}_{slug}"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out / "packet.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path, arc in files:
            z.write(path, arc)

    tpl = (KIT_ROOT / "prompts" / "50_pro_packet.md").read_text(encoding="utf-8")
    body = tpl.split("<!-- PROMPT -->", 1)[-1].strip()
    top_lines = "\n".join(f"- {r.name}: CV {r.cv:.6f}" for r in top.itertuples()) or "- (nenhum ainda)"
    prompt = (body.replace("{TASK}", args.task)
                  .replace("{COMPETITION}", CFG["competition"])
                  .replace("{METRIC}", CFG["metric"])
                  .replace("{TOP}", top_lines)
                  .replace("{FILES}", "\n".join(f"- {arc}" for _, arc in files)))
    (out / "PROMPT.md").write_text(prompt + "\n", encoding="utf-8")
    print(f"[ok] {out / 'packet.zip'} ({len(files)} arquivos)\n     {out / 'PROMPT.md'}")


if __name__ == "__main__":
    main()
