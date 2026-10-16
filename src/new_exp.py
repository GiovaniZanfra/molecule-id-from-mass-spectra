"""Cria um experimento novo com o nome certo e o próximo número livre.

    python src/new_exp.py cosine --author w1                  # retrieval: template retrieval (padrão)
    python src/new_exp.py xgb --author w1                     # tabular: template gbdt (padrão)
    python src/new_exp.py lr --author w1 --template blank     # template mínimo (sklearn)
    python src/new_exp.py stack --author mgr --template stack # stacking nível 2 (roda local)
    python src/new_exp.py fusion --author mgr --template fusion  # retrieval: ensemble que vai pra submissão
    python src/new_exp.py xgb --author w2 --from xgb_boot_001 # parte de um experimento existente
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, KIT_ROOT, NAME_RE, OOF_DIR, SCRIPTS_DIR, TOKEN_RE  # noqa: E402

EXP_DIR = KIT_ROOT / "experiments"
TPL_DIR = EXP_DIR / "_templates"


def next_number(family: str, author: str) -> int:
    pat = re.compile(rf"^{family}_{author}_(\d+)$")
    nums = [0]
    for d, ext in ((EXP_DIR, "*.py"), (SCRIPTS_DIR, "*.py"), (OOF_DIR, "*.npy"), (OOF_DIR, "*.parquet")):
        if d.exists():
            for p in d.glob(ext):
                m = pat.match(p.stem)
                if m:
                    nums.append(int(m.group(1)))
    return max(nums) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--author", required=True)
    ap.add_argument("--from", dest="parent", default="")
    ap.add_argument("--template", default="retrieval" if CFG["task"] == "retrieval" else "gbdt",
                    choices=["gbdt", "blank", "stack", "retrieval", "fusion"])
    args = ap.parse_args()
    for tok, what in ((args.family, "família"), (args.author, "autor")):
        if not TOKEN_RE.match(tok):
            sys.exit(f"{what} '{tok}' inválido: use só letras minúsculas e números (sem _ ou -)")

    name = f"{args.family}_{args.author}_{next_number(args.family, args.author):03d}"
    if args.parent:
        if not NAME_RE.match(args.parent):
            sys.exit(f"--from '{args.parent}' não é um nome de experimento válido")
        src = EXP_DIR / f"{args.parent}.py"
        if not src.exists():
            src = SCRIPTS_DIR / f"{args.parent}.py"
        if not src.exists():
            sys.exit(f"não achei o script de {args.parent} em experiments/ nem em {SCRIPTS_DIR}")
        text = src.read_text(encoding="utf-8")
        lines = text.split("\n")
        lines[0] = lines[0].replace(args.parent, name)
        text = "\n".join(lines)
        text, n = re.subn(r'(?m)^PARENT = .*$', f'PARENT = "{args.parent}"', text)
        if n == 0:
            print("aviso: o script pai não tem a linha PARENT = ...; adicione o pai no finish(parent=...)")
    else:
        text = (TPL_DIR / f"{args.template}.py").read_text(encoding="utf-8").replace("{NAME}", name)

    out = EXP_DIR / f"{name}.py"
    out.write_text(text, encoding="utf-8")
    print(out.relative_to(KIT_ROOT))


if __name__ == "__main__":
    main()
