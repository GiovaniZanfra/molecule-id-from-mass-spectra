"""Cria o arquivo de learnings de um job (na pasta de artefatos, compartilhada).

    python src/new_note.py --author w1 --job j3
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LEARN_DIR, TOKEN_RE  # noqa: E402

TEMPLATE = """# {job} — {author} — {date}

Objetivo do job:

Experimentos registrados (nome → CV copiado do registry / python src/top.py):
-

O que funcionou:
-

O que não funcionou:
-

Suspeitas / bugs encontrados:
-

Próxima ideia (1–3, concretas):
-
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", required=True)
    ap.add_argument("--job", required=True)
    args = ap.parse_args()
    for tok in (args.author, args.job):
        if not TOKEN_RE.match(tok):
            sys.exit(f"'{tok}' inválido: use só letras minúsculas e números")
    today = dt.date.today()
    LEARN_DIR.mkdir(parents=True, exist_ok=True)
    path = LEARN_DIR / f"{today:%Y%m%d}_{args.author}_{args.job}.md"
    i = 2
    while path.exists():
        path = LEARN_DIR / f"{today:%Y%m%d}_{args.author}_{args.job}_{i}.md"
        i += 1
    path.write_text(TEMPLATE.format(job=args.job, author=args.author, date=today.isoformat()), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
