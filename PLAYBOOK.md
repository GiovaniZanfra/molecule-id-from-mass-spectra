# PLAYBOOK — time de agentes para Kaggle tabular

Inspirado no jeito que o Chris Deotte usa agentes em 2026: agentes escrevem
código rápido, a GPU executa rápido, e um **contrato de arquivos** deixa
qualquer agente usar o trabalho de qualquer outro — sem precisar confiar no que
eles dizem. Regras curtas: `AGENTS.md`. Este arquivo explica o porquê e o fluxo.

---

## 1. As três peças que fazem o time funcionar

**Contrato de arquivos.** Todo experimento usa os mesmos 5 folds (`folds.npy`)
e salva OOF + test preds com nome padronizado. Isso permite que o gerente
combine centenas de modelos, que o auditor recalcule tudo, e que dois agentes
trabalhem em paralelo sem conflito (cada um só cria arquivos novos).

**Memória compartilhada.** Cada job termina num arquivo de learnings. O gerente
consolida tudo em `LEARNINGS.md`, que todo agente lê antes de começar. É a
"working memory" do time.

**Verificação por números, não por palavras.** O CV oficial é recalculado do OOF
salvo (`registry.csv`). O auditor compara o que o agente disse com o registry,
procura vazamento e duplicata. É assim que se pega agente mentindo ou enrolando.

---

## 2. Onde as coisas ficam

```
repo/                         (versionado no git)
  AGENTS.md  PLAYBOOK.md  CLAUDE.md  comp.json
  JOBS.md                     fila da campanha atual (só o gerente escreve)
  LEARNINGS.md                memória consolidada (só o gerente escreve)
  notes/eda.md                EDA do bootstrap
  prompts/                    um prompt por papel
  experiments/                scripts de experimento (<familia>_<autor>_<NNN>.py)
  experiments/_templates/     gbdt.py, blank.py, stack.py
  src/                        o contrato em código (ninguém edita além de você)

artifacts/                    (fora do git; compartilhado entre worktrees)
  data/                       train/test/sample_submission (+ data/extra/)
  folds.npy                   os 5 folds — nunca recriar no meio da competição
  structures.parquet          retrieval: uma linha por estrutura (chave da métrica)
  queries.parquet             retrieval: consultas de validação (fold, classe simulada 1/2/3)
  train_struct.npy            retrieval: estrutura de cada linha do train
  cache/                      retrieval: picos limpos do train (mmap) e chaves já calculadas
  registry.csv                um registro por experimento, CV recalculado
  oof/ preds/                 <nome>.npy (+ <nome>.meta.json em oof/)
  scripts/                    cópia exata do código que gerou cada artefato
  learnings/                  um arquivo por job (learnings/_done/ após consolidar)
  reports/                    audit.md, ensemble.md, workers.md, kaggle_logs/
  ensembles/ subs/            pesos dos ensembles e CSVs candidatos à submissão
  kaggle/                     pacotes enviados ao Kaggle e fila de pendentes
```

Por padrão `artifacts/` fica dentro do checkout principal do repo. Agentes em
worktrees (ex.: `/best-of-n` no Cursor) escrevem no mesmo lugar — o `common.py`
descobre o checkout principal pelo git. Caminhos: `python src/top.py --where`.

---

## 3. Papéis

| Papel | Faz | Prompt | Modelo sugerido (Cursor Pro) |
|---|---|---|---|
| **Bootstrap** | EDA, baseline, primeira lista de jobs | `00_bootstrap.md` | frontier, 1× por competição |
| **Worker** | executa 1 job de `JOBS.md` | `10_worker.md` | Composer / Auto (uso incluído) |
| **Duelista** | mesma tarefa, modelos diferentes, cada um no seu worktree | `20_best_of_n.md` | 2 modelos, só em tarefa de "melhor modelo único" |
| **Gerente** | avalia workers, consolida memória, escreve a próxima campanha, monta ensemble | `30_manager.md` | frontier, 1 conversa por campanha |
| **Auditor** | investiga cada FAIL/WARN do `audit.py` e dá veredito | `40_auditor.md` | Composer; frontier só se o caso for cabeludo |
| **Consultor Pro** | resolve UMA tarefa difícil em turno único, fora do Cursor | `50_pro_packet.md` | ChatGPT Pro / Claude no navegador |
| **Batedor** | todo dia: lê o delta do fórum e dos notebooks, atualiza `notes/research.md` e propõe 0–1 job | `60_scout.md` | Sonnet, headless pelo `src/scout.sh` |

O humano (você) é dono do processo: define as fases, aprova `JOBS.md` se
quiser, faz as submissões e leva o pacote pro modelo Pro.

---

## 4. Fases

**Fase 0 — Setup (você, uma vez).** Ver `README.md`: `comp.json`, dados em
`artifacts/data/`, `python src/make_folds.py`, `python src/kaggle_run.py setup`.

**Fase 1 — Bootstrap (1 agente).** EDA → `notes/eda.md`, baseline registrado,
primeiro `LEARNINGS.md` e proposta de jobs em `JOBS.md`. O agente aprende
fazendo: deixe ele tentar sozinho antes de dar dicas.

**Fase 2 — Duelo (opcional).** `/best-of-n` com 2 modelos na tarefa "melhor
GBDT único" (ou NN). Os perdedores não são lixo: os OOFs deles já estão no
registry e entram no ensemble. Do vencedor, você aproveita o código.

**Fase 3 — Campanha.** O gerente escreve 4–8 jobs independentes em `JOBS.md`.
Você dispara os workers em paralelo (no Cursor: `/multitask`, um subagente por
job; autor = id do job). Cada job gera experimentos novos + 1 learnings.

**Fase 4 — Auditoria + ensemble.** `python src/audit.py`. Se houver FAIL/WARN,
auditor investiga. Depois o gerente roda `python src/hill_climb.py --nested`,
avalia os workers, consolida `LEARNINGS.md` e propõe a próxima campanha.

**Fase 5 — Submissão (você).** Só se o CV *nested* do ensemble melhorar o
anterior por mais que o ruído. Volta para a Fase 3.

**Todo dia — Batedor (automático).** O agendador do Windows roda `src/scout.sh`:
`src/kaggle_intel.py` baixa só o que mudou (tópicos e top-30 notebooks por score) para
`artifacts/intel/<data>/`; se houver novidade, o batedor cruza com o registry e o `LEARNINGS.md`,
acrescenta `## Atualização <data>` em `notes/research.md` e escreve `reports/intel/<data>.md` com
0 ou 1 job proposto (id `p<data>`). Ele não roda experimento nem gasta GPU: o job só vira trabalho
quando você aprova ("rode o job proposto em reports/intel/<data>.md como worker") ou quando o
gerente o inclui na campanha. Ideia pública não é copiada: vira experimento nosso, com os nossos
folds, a partir da família `meta` (job j7).

Supervisão: não deixe agente rodando mais que algumas horas sem olhar.
O próprio Chris diz que não passa de 12h sem supervisionar.

---

## 5. Cardápio de jobs (o gerente escolhe daqui)

| Tipo | O que é |
|---|---|
| `port` | reescrever um notebook público usando os nossos folds e o contrato |
| `reuse` | converter uma solução de competição anterior para esta |
| `loop` | campanha de N experimentos melhorando um modelo; registrar 4–8 intermediários, não só o melhor |
| `diverse` | "um modelo que ainda não está no nosso ensemble" — o agente escolhe |
| `variant` | variantes de uma família (features, hiperparâmetros, seeds, target transformado) |
| `fe` | feature engineering guiada por EDA (relação feature × target, interações, contagens) |
| `original` | usar o dataset original (playground): concatenar, encoding a partir do original etc. |
| `distill` | treinar um modelo único usando OOF/preds de vários modelos como alvo suave |
| `stack` | meta-modelo nível 2 sobre os OOFs (template `stack`), roda local |
| `research` | ler papers/fórum/notebooks e propor jobs concretos (não treina nada) |

Cerca de metade das ideias boas do Chris veio de pedir "um modelo que não está
no meu ensemble". Sempre tenha ao menos um job `diverse` por campanha.

---

## 6. Formato do learnings (um por job)

`python src/new_note.py --author <autor> --job <id>` cria o arquivo já com este
esqueleto:

```
# <job> — <autor> — <data>
Objetivo do job:
Experimentos registrados (nome → CV do registry):
O que funcionou:
O que não funcionou:
Suspeitas / bugs encontrados:
Próxima ideia (1–3, concretas):
```

---

## 7. Anti-overfit

- Decisão pelo CV no fold fixo. Para ensemble, só vale o **CV nested** do
  `hill_climb.py` (pesos escolhidos sem ver o fold avaliado).
- Ganho menor que o ruído não é ganho. Ruído ≈ desvio entre folds / √5
  (aparece no `top.py` e no `ensemble.md`).
- LB público é sanity check, não alvo. Poucas submissões.
- CASMI: o LB público tem ~130 moléculas (1 molécula ≈ 0,008). Decida pelo CV, olhando cada
  classe simulada (c1/c2/c3) — ganho que só aparece numa classe pode sumir no teste.
- Lição do Chris (S6E4): depois do 3º dia, mais trabalho subiu o CV e derrubou o
  private. Agentes batem o teto rápido; campanha longa demais vira overfit.

---

## 8. Orçamento (Cursor Pro)

- Treino roda no Kaggle: o agente gasta token escrevendo e lendo, não esperando.
- Workers em Composer/Auto. Modelo frontier só no bootstrap, no gerente e em
  auditoria difícil.
- Contexto curto: agentes leem `LEARNINGS.md`, `top.py` e `reports/audit.md`,
  não notebooks inteiros.
- Cloud agents não enxergam a pasta `artifacts/` local; use agentes locais.
- O Kaggle limita sessões de GPU simultâneas e horas por semana. Se o push
  falhar por limite, espere os pendentes (`kaggle_run.py wait`).
