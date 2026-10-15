# Kit: time de agentes para Kaggle tabular

Um repo-modelo para rodar competições tabulares com vários agentes ao mesmo
tempo, no estilo que o Chris Deotte usou em 2026: os agentes escrevem o código,
a GPU do Kaggle executa, e um contrato de arquivos (folds fixos + OOF salvo +
registry recalculado) deixa o time combinar e auditar o trabalho de todo mundo.

Funciona no **Cursor** e no **Claude Code** — o mesmo repo, as mesmas regras.

| Arquivo | Para quê |
|---|---|
| `AGENTS.md` | regras curtas que todo agente segue (o Cursor e o Claude Code carregam sozinhos) |
| `PLAYBOOK.md` | papéis, fases, cardápio de jobs, anti-overfit, orçamento |
| `prompts/` | um prompt por papel: bootstrap, worker, duelo, gerente, auditor, consultor Pro |
| `comp.json` | configuração da competição |
| `JOBS.md` / `LEARNINGS.md` | fila da campanha e memória do time (o gerente mantém) |
| `src/` | o contrato em código: folds, CV, registry, Kaggle, audit, ensemble |
| `experiments/_templates/` | `gbdt.py` (XGBoost, GPU), `blank.py` (sklearn), `stack.py` (nível 2) |

---

## Setup (uma vez por competição, ~15 min)

1. **Repo novo:** copie esta pasta, `git init`, commit inicial.
2. **Dependências:** `pip install -r requirements.txt`.
3. **Kaggle:** configure o token da API (kaggle.com → Settings → API) e
   **aceite as regras da competição no site** — sem isso nem o download nem os
   kernels acessam os dados.
4. **`comp.json`:** preencha pelo menos `competition` (slug da URL), `task`,
   `metric`, `id`, `target` e `kaggle_username`. Tabela de campos no fim.
5. **Dados:** `kaggle competitions download -c <slug> -p artifacts/data` e
   descompacte lá. Dataset original do playground? Coloque em
   `artifacts/data/extra/` e liste em `extra_files`.
6. **Folds e Kaggle:**
   ```
   python src/make_folds.py
   python src/kaggle_run.py setup     # sobe folds (+ extras) como dataset privado
   ```
   Espere alguns minutos até o dataset ficar disponível.
7. **Teste rápido:** `python src/new_exp.py xgb --author teste` e
   `python experiments/xgb_teste_001.py --debug`. Rodou? Apague o arquivo de
   teste e faça commit.

---

## Dia a dia no Cursor

```
 bootstrap ──► duelo (opcional) ──► campanha: /multitask com N workers
                                          │
                       ┌──────────────────┘
                       ▼
             audit (+ auditor se tiver WARN/FAIL)
                       │
                       ▼
     gerente: ensemble nested, avalia workers, consolida memória, escreve JOBS.md
                       │
                       ├──► você decide se submete
                       └──► próxima campanha
```

| Passo | Onde no Cursor | Prompt | Modelo (plano Pro) |
|---|---|---|---|
| 1. Bootstrap | conversa de agente normal | `00_bootstrap.md` | frontier |
| 2. Duelo (opcional) | `/best-of-n` com 2 modelos | `20_best_of_n.md` | frontier + Composer |
| 3. Campanha | `/multitask`, checkout principal | `10_worker.md` | Composer / Auto |
| 4. Auditoria | conversa nova, se o audit apontar algo | `40_auditor.md` | Composer |
| 5. Gerente | conversa nova | `30_manager.md` | frontier |
| 6. Submissão | terminal (você) | — | — |

**Como usar um prompt:** o jeito mais simples é mandar o agente ler o arquivo:

> Siga prompts/10_worker.md com AUTOR = j3 e JOB = j3.

Ou copie o texto abaixo da linha `<!-- PROMPT -->` e troque os `{CAMPOS}`.

**Campanha em paralelo:** no `/multitask`, algo como:

> Rode os jobs j1 a j6 de JOBS.md em paralelo, um subagente por job. Cada subagente segue prompts/10_worker.md com AUTOR = id do job e JOB = id do job.

Não precisa de worktree para isso: pelo contrato cada job só cria arquivos novos
e o registry tem lock. Worktree só aparece no `/best-of-n`, e mesmo lá os
artefatos vão para a pasta compartilhada do checkout principal.

**Submissão (só você):**
```
python src/top.py                      # estado do time
cat artifacts/reports/ensemble.md      # CV nested e veredito vs. ensemble anterior
kaggle competitions submit -c <slug> -f artifacts/subs/hc_NNN.csv -m "hc_NNN nested 0.xxxxx"
```

**Consultor Pro (quando travar):** `python src/make_packet.py "tarefa concreta"`
gera `packet.zip` + `PROMPT.md`. Conversa nova no ChatGPT Pro / Claude no
navegador, sobe o zip, cola o prompt, turno único. Detalhes em `prompts/50_pro_packet.md`.

---

## No Claude Code

Igual. O `CLAUDE.md` importa o `AGENTS.md`. Para a campanha, peça para rodar os
jobs em paralelo com subagentes, cada um seguindo `prompts/10_worker.md`. Dá para
misturar: workers no Cursor e gerente no Claude Code, por exemplo — o que une o
time é a pasta de artefatos, não a ferramenta.

---

## Orçamento

- O treino roda no Kaggle: o agente gasta token escrevendo e lendo, não esperando.
- Workers em Composer/Auto; frontier só no bootstrap, no gerente e em auditoria difícil.
- Agentes leem `LEARNINGS.md`, `python src/top.py` e `reports/audit.md` — não notebooks inteiros.
- O Kaggle limita sessões de GPU simultâneas e horas por semana. Se o push falhar
  por limite, `python src/kaggle_run.py wait` e tente de novo.

---

## Limites conhecidos

- O `kaggle_run.py` foi testado com um CLI do Kaggle simulado, não contra a API
  real. Na primeira execução de verdade, abra a página do kernel e confira o log.
  O `common.py` procura `train.csv` e `folds.npy` recursivamente em
  `/kaggle/input`, então não depende do caminho exato de montagem.
- Cloud agents do Cursor não enxergam a pasta `artifacts/` local: use agentes locais.
- Stacking (nível 2) roda local; o `kaggle_run.py` recusa scripts que leem OOFs.
- Se o sandbox do Cursor bloquear escrita fora do worktree no `/best-of-n`,
  aprove o comando ou defina `KIT_ARTIFACTS` com um caminho permitido.

---

## Campos do `comp.json`

| Campo | Exemplo | Nota |
|---|---|---|
| `competition` | `playground-series-s6e10` | slug da URL |
| `task` | `binary` / `multiclass` / `regression` | |
| `metric` | `auc`, `logloss`, `accuracy`, `mlogloss`, `map3`, `rmse`, `rmsle`, `mae` | nova métrica: `src/metric.py` |
| `id`, `target` | `id`, `Churn` | nomes das colunas |
| `classes` | `["No","Yes"]` | opcional; ordem das classes (multiclass/proba) |
| `positive_label` | `"Yes"` | binary com rótulo texto |
| `submission_format` | `value` / `label` / `top3` / `proba_columns` | como escrever o CSV |
| `submission_target` | `null` | coluna do sample_submission, se diferente do target |
| `extra_files` | `["original.csv"]` | ficam em `artifacts/data/extra/` e sobem no `setup` |
| `artifacts_dir` | `{repo}/artifacts` | `{repo}` = checkout principal; aceita `{competition}` |
| `kaggle_username` | `seu-usuario` | |
| `kaggle_kernel_prefix` | `""` | padrão: `s6e10` extraído do slug |
| `kaggle_kernel_extra` | `{}` | campos extras do kernel-metadata.json |
| `debug_rows` | `20000` | tamanho da amostra no `--debug` |
| `suspicious_gain` | `null` | salto de CV que o audit marca; `null` = automático |
