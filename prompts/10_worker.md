# 10 — Worker (1 por job, vários em paralelo)

**Quando:** depois que JOBS.md tem a campanha.
**Onde:** no Cursor, `/multitask` no checkout principal — um subagente por job. Não precisa de worktree: pelo contrato cada job só cria arquivos novos.
**Modelo:** Composer / Auto (uso incluído). Frontier só para job difícil.
**Como:** no /multitask, mande algo como:

> Rode os jobs j1 a j6 de JOBS.md em paralelo, um subagente por job. Cada subagente segue prompts/10_worker.md com AUTOR = id do job e JOB = id do job.

Para um worker avulso, cole o bloco abaixo trocando {AUTOR} e {JOB}.

<!-- PROMPT -->
Você é um worker do time. AUTOR = {AUTOR} | JOB = {JOB}
(Se você foi criado por /multitask, AUTOR e JOB são o id do job que te deram, ex.: j3.)

1. Leia AGENTS.md, a linha do job em JOBS.md, LEARNINGS.md e notes/eda.md. Rode `python src/top.py -n 15`. Se existir, leia reports/audit.md (pasta em `python src/top.py --where`).
2. Para cada experimento:
   - crie com `python src/new_exp.py <familia> --author <AUTOR>` (ou `--from <pai>`) e escreva a hipótese em NOTES;
   - `python experiments/<nome>.py --debug` até rodar limpo;
   - modelo pesado (GBDT grande, NN): `python src/kaggle_run.py run experiments/<nome>.py`; modelo leve: rode local sem --debug.
   Pode enviar vários para o Kaggle e depois esperar todos com `python src/kaggle_run.py wait <nomes>`.
3. Orçamento: no máximo o número de experimentos do job em JOBS.md. Registre os intermediários que mudam algo (servem para o ensemble), não só o melhor.
4. Falhou no Kaggle? O `wait` mostra o final do log. Corrija num experimento NOVO (`--from`) e tente de novo — no máximo 2 vezes por ideia.
5. Ao terminar: `python src/new_note.py --author <AUTOR> --job <JOB>` e preencha. CVs vêm de `python src/top.py --author <AUTOR>`, nunca da memória.
6. Responda em até 5 linhas: nomes registrados + CV do registry, o que aprendeu, próxima ideia.

Não edite JOBS.md, LEARNINGS.md, src/, comp.json nem scripts de outros autores. Não apague artefatos. Não submeta. Bug em src/? Descreva no learnings.
