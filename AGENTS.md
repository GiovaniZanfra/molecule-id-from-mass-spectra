# Regras para agentes — leia antes de qualquer tarefa

Este repo é um time de agentes para competições tabulares do Kaggle.
Papéis, fases e cardápio de jobs: `PLAYBOOK.md`. Sua tarefa específica vem do
prompt (pasta `prompts/`) e, se você for worker, da sua linha em `JOBS.md`.
Competição atual: Enveda CASMI 2026 (espectro MS/MS → estrutura, MRR@25, competição de código).
Regras e respostas da organização: `notes/competition.md`; estado da arte público: `notes/research.md`.

## Contrato (inegociável)

1. **Folds**: use só `get_folds()` / `run_cv()` de `src/common.py`. Nunca crie
   outro split — nem ao reescrever notebook público (troque o split dele pelo nosso).
   Retrieval (esta competição): a validação é `queries.parquet` + `folds.npy`; o experimento
   é `predict(ctx)` chamado pelo `run_cv(predict, args)` e recebe tudo pelo `ctx`.
2. **Um experimento = um script** `experiments/<familia>_<autor>_<NNN>.py`,
   criado com `python src/new_exp.py <familia> --author <autor>`
   (ou `--from <experimento>` para partir de outro). Família e autor: só letras
   minúsculas e números.
3. **Todo experimento termina em `finish(...)`**, bom ou ruim. Isso salva OOF +
   test preds + meta e registra no registry. Experimento ruim também fica: serve
   para diversidade do ensemble e para o auditor.
4. **O CV que vale é o do registry** (`python src/top.py`), recalculado a partir
   do OOF salvo. Ao reportar resultado, copie de lá — nunca de memória.
5. **Nada que use a resposta fora do fold.** Tabular: target encoding, pseudo-label,
   seleção de feature por importância, stacking: tudo dentro do `fit_predict`,
   que só recebe o treino do fold. `features()` não pode tocar no target.
   Retrieval: no fold f, biblioteca, dado de treino e estatística por estrutura (contagens,
   bibliotecas, popularidade tirada do train) só das linhas `ctx.train_mask`; nenhum candidato
   com chave em `ctx.excluded`, inclusive de bancos externos (COCONUT, PubChem); modelo
   pré-treinado não pode ter visto as estruturas de validação do fold (peso treinado no train
   inteiro só vale no teste); nada pré-calculado das massas do `test.parquet` visível.
6. **Treino pesado roda no Kaggle.** Teste local com
   `python experiments/<nome>.py --debug` (fold 0, amostra, não salva nada) e
   rode de verdade com `python src/kaggle_run.py run experiments/<nome>.py`.
   Stacking (nível 2) e modelos leves podem rodar local sem `--debug`.
   Pesos treinados em outro kernel: publique como dataset e declare
   `KAGGLE_DATASETS = ["usuario/dataset"]` no experimento (o `kaggle_run.py` anexa).
7. **Não mexa no que não é seu**: scripts de outros autores, `src/`,
   `comp.json`, `JOBS.md`, `LEARNINGS.md`, `folds.npy`. Achou bug em `src/`?
   Descreva no seu learnings.
8. **Nunca apague, sobrescreva ou edite artefatos** (pasta de artefatos,
   `registry.csv`). Quer rodar de novo? Crie um experimento novo.
9. **Nunca submeta no Kaggle** (`kaggle competitions submit`). Submissão é humana.
   `kaggle_run.py submit` só monta e envia o kernel de submissão; quem clica Submit é o humano.
10. **Job só termina com um learnings** criado por `python src/new_note.py`,
    preenchido, com os nomes registrados e o CV copiado do registry.
11. **Conformidade (CASMI):** nada de NIST, METLIN ou bibliotecas de fabricantes, nem modelo
    treinado com eles. Peso de terceiros (inclusive de outros competidores) só com código de
    treino público e dados permitidos. Na dúvida, veja `notes/competition.md`.

## Comandos

| Comando | O que faz |
|---|---|
| `python src/top.py` | ranking do registry (`--where` mostra onde ficam os artefatos) |
| `python src/new_exp.py cosine --author w1` | cria `experiments/cosine_w1_001.py` a partir do template (`retrieval` nesta competição) |
| `python src/new_exp.py fusion --author mgr --template fusion` | ensemble que vai para a submissão (pesos do `hill_climb.py`) |
| `python src/new_exp.py xgb --author w1 --from xgb_boot_001` | cria a partir de um experimento existente |
| `python experiments/xgb_w1_001.py --debug` | teste rápido, não salva |
| `python src/kaggle_run.py run experiments/xgb_w1_001.py` | envia para a GPU do Kaggle (não espera) |
| `python src/kaggle_run.py wait` | espera os pendentes, baixa e registra |
| `python src/kaggle_run.py submit experiments/x.py` | kernel de submissão (só teste → `submission.csv`); quem submete é o humano |
| `python src/audit.py` | audita todos os artefatos → `reports/audit.md` |
| `python src/hill_climb.py --nested` | ensemble + submissão candidata → `reports/ensemble.md` |
| `python src/new_note.py --author w1 --job j3` | cria o arquivo de learnings do job |
| `python src/make_packet.py "tarefa"` | pacote + prompt para consultar um modelo Pro no navegador |

`reports/`, `learnings/`, `scripts/`, `subs/` ficam na pasta de artefatos
(veja `python src/top.py --where`), que é compartilhada entre worktrees.
