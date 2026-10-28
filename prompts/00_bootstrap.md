# 00 — Bootstrap (1 agente, 1× por competição)

**Quando:** logo depois do setup (README, passos 1–6).
**Onde:** uma conversa de agente normal, no checkout principal (sem worktree).
**Modelo:** o melhor frontier que você tiver — é a fundação do resto.
**Como:** cole o bloco abaixo.

<!-- PROMPT -->
Você é o primeiro agente desta competição. Autor: boot.
Nesta tarefa você PODE escrever LEARNINGS.md, JOBS.md e notes/eda.md (exceção à regra 7 do AGENTS.md).

Antes de tudo leia AGENTS.md, PLAYBOOK.md, comp.json, notes/competition.md e notes/research.md. Rode `python src/top.py --where` para saber onde ficam dados e artefatos.

1. EDA (ainda sem modelo)
   - Escreva e rode scripts em `eda/` (não em experiments/). Use `common.load_train()` (metadados, sem picos), `load_spectra()` (picos limpos, mmap), `load_structures()` e `query_frame("val")`. O train não cabe inteiro no pandas com os picos.
   - Investigue: bibliotecas × instrumento × aduto × energia de colisão, picos por espectro, massas, sobreposição de estruturas entre bibliotecas, e como as consultas de validação se distribuem por classe/subconjunto. O `test.parquet` visível é amostra do enveda-180: não ajuste nada nele.
   - Resuma em notes/eda.md: no máximo 1 página, só o que muda decisão de modelagem.
2. Baseline
   - Já existe `cosine_infra_001` (busca por cosseno, CV no `top.py`). Faça 1–2 variações baratas: `python src/new_exp.py cosine --author boot`, `--debug` até rodar limpo, depois rode local sem --debug (o CV completo leva ~30 s).
   - Algo pesado (modelo espectro → fingerprint, por exemplo): `python src/kaggle_run.py run experiments/<nome>.py --wait`.
3. Memória e plano
   - `python src/new_note.py --author boot --job bootstrap` e preencha.
   - Preencha LEARNINGS.md (seções Dados e O que funciona) com fatos curtos.
   - Escreva a campanha 1 em JOBS.md: 5–7 jobs independentes do cardápio do PLAYBOOK, ids j1, j2, ... Use a seção 7 de notes/research.md. Inclua ao menos 1 `diverse`; `port` só de notebook cujo código e pesos sejam reprodutíveis (regra 11 do AGENTS.md). Cada job com objetivo concreto, pai (se houver), orçamento (máx. de experimentos) e critério de sucesso (diga em qual classe: c1, c2 ou c3).
4. Pare e responda em até 10 linhas: achados principais da EDA, CV do baseline copiado de `python src/top.py`, e a campanha proposta.

Não faça ensemble nem stacking. Não submeta nada.
