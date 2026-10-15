# 00 — Bootstrap (1 agente, 1× por competição)

**Quando:** logo depois do setup (README, passos 1–6).
**Onde:** uma conversa de agente normal, no checkout principal (sem worktree).
**Modelo:** o melhor frontier que você tiver — é a fundação do resto.
**Como:** cole o bloco abaixo.

<!-- PROMPT -->
Você é o primeiro agente desta competição. Autor: boot.
Nesta tarefa você PODE escrever LEARNINGS.md, JOBS.md e notes/eda.md (exceção à regra 7 do AGENTS.md).

Antes de tudo leia AGENTS.md, PLAYBOOK.md e comp.json. Rode `python src/top.py --where` para saber onde ficam dados e artefatos.

1. EDA (ainda sem modelo)
   - Escreva e rode scripts em `eda/` (não em experiments/). Investigue: tamanho, tipos, target (balanço/distribuição), missing, cardinalidade das categóricas, numéricas com poucos valores distintos, diferença train × test (adversarial validation rápida) e a relação de cada feature com o target.
   - Se comp.json tiver extra_files (dataset original), compare original × train e veja se o original ajuda.
   - Resuma em notes/eda.md: no máximo 1 página, só o que muda decisão de modelagem.
2. Baseline
   - `python src/new_exp.py xgb --author boot`; ajuste `features()` com o que a EDA mostrou (sem usar target ali).
   - `python experiments/xgb_boot_001.py --debug` até rodar limpo.
   - Rode de verdade: `python src/kaggle_run.py run experiments/xgb_boot_001.py --wait`.
   - Um modelo barato e diferente: `python src/new_exp.py lr --author boot --template blank` e rode local sem --debug.
3. Memória e plano
   - `python src/new_note.py --author boot --job bootstrap` e preencha.
   - Preencha LEARNINGS.md (seções Dados e O que funciona) com fatos curtos.
   - Escreva a campanha 1 em JOBS.md: 5–7 jobs independentes do cardápio do PLAYBOOK, ids j1, j2, ... Inclua ao menos 1 `diverse` e 1 `fe`; `port` se houver notebook público bom; `original` se houver dataset original. Cada job com objetivo concreto, pai (se houver), orçamento (máx. de experimentos) e critério de sucesso.
4. Pare e responda em até 10 linhas: achados principais da EDA, CV do baseline copiado de `python src/top.py`, e a campanha proposta.

Não faça ensemble nem stacking. Não submeta nada.
