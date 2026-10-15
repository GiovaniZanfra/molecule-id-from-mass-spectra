# 20 — Duelo (/best-of-n)

**Quando:** depois do bootstrap, para a pergunta "qual o melhor modelo ÚNICO da família X?". Também quando o time empacar numa família.
**Onde:** no Cursor, `/best-of-n` com 2 modelos (ex.: 1 frontier + Composer). Cada um roda no seu worktree; os artefatos vão para a mesma pasta compartilhada.
**Depois:** os OOFs de todos ficam no registry (os "perdedores" entram no ensemble). Do vencedor você mergeia o código, se quiser.
**Como:** troque {FAMILIA}, {PAI} e {N} e cole.

<!-- PROMPT -->
Tarefa: construir o melhor modelo ÚNICO da família {FAMILIA} nesta competição.
Seu AUTOR é o nome curto do seu modelo (ex.: opus, gpt, composer, gemini) — só letras minúsculas e números.

1. Leia AGENTS.md, LEARNINGS.md, notes/eda.md e `python src/top.py`.
2. Comece de {PAI}: `python src/new_exp.py {FAMILIA} --author <AUTOR> --from {PAI}` (ou do template, se {PAI} estiver vazio).
3. Itere em feature engineering e hiperparâmetros. Cada versão que muda algo vira um experimento novo e registrado: --debug local, depois `python src/kaggle_run.py run ...`.
4. Orçamento: até {N} experimentos registrados.
5. Fim: `python src/new_note.py --author <AUTOR> --job duelo{FAMILIA}` preenchido, e resposta curta: melhor nome + CV do registry + o que fez diferença.

Não faça ensemble. Não submeta.
