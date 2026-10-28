# LEARNINGS — memória consolidada do time

> Só o gerente reescreve este arquivo (máx. ~80 linhas). Todo agente lê antes de começar.
> Fonte: arquivos em `learnings/` na pasta de artefatos. Detalhes da EDA: `notes/eda.md`.

## Dados
- Train: 2,54M espectros, 274k estruturas. `load_train()` (sem picos) = 2 GB; picos via `load_spectra()` (mmap).
  Local tem 11 GB: **no máximo 3 experimentos locais ao mesmo tempo**; o resto no Kaggle (`--no-gpu` se for CPU).
- enveda-180 = 67% das estruturas, massa 235–499, só 1,3% com espectro público. O `test.parquet` visível
  é amostra dele; o teste oculto vai de 157 a 1.159 Da. O melhor proxy é o subconjunto `np` (enveda-np-examples).
- Precursor do e180/np positivo está +0,7 mDa (≈ massa do elétron + 0,15). Na validação, subtraia
  `chem.ELECTRON_MASS` das consultas timsTOF positivas; no teste (`ctx.is_test`) não (já vem corrigido).
- Janela de ±10 ppm: no e180 ~80 candidatas e ~37 isômeros da mesma fórmula; no np/nplib, 7–13 e 3–7.
  A resposta da c2 está na janela em 99% dos casos.
- Treino pesado: as linhas com `folds == -1` (98% do train) valem em todos os folds e no teste →
  **um único treino** serve para o CV inteiro (opcional: re-treinar com tudo só para o teste).

## O que funciona
- Busca por cosseno na janela de massa: c1 ≈ 0,88 (`cosine_infra_001`).
- Propagação por análogos na janela, max_a cos(q,a)·Tanimoto(c,a): c2 vai de 0,141 para 0,206 (`cosine_boot_001`).
- Prior de ppm exp(−0,5·(ppm/3)²) mais a correção do elétron: c2 0,246 e c1 0,890; CV 0,2533 (`cosine_boot_002`, melhor até agora).

## O que não funciona
- Análogos de dentro da janela não põem a resposta da c2 em 1º. O top-1 errado tem Tanimoto mediano
  0,2–0,3 com a resposta (`eda/04_oof.py`). Falta um modelo espectro→estrutura.

## Bugs / armadilhas conhecidas
- **Vazamento da c2 no painel**: o pool do CV só tem estruturas do train. Na c2, a resposta é quase
  sempre a única candidata sem espectro no fold (53% das consultas; 68% entre as da mesma fórmula).
  É proibido score, feature ou ranker que favoreça candidata por não ter espectro. Uma candidata sem
  espectro só ganha pontos por evidência que vale igual para as com espectro (FP previsto, fragmentação,
  análogos). LambdaRank só depois que o pool tiver estruturas externas sem espectro.
- A c3 simulada mantém os análogos da resposta no pool, o que é otimista para geração.
- `precursor_error_ppm` do train está em módulo e errado para [M+CH2O2-H]-: recalcule com `chem.ADDUCT_SHIFT`.
- Peso pré-treinado de terceiros (MassSpecGym, DreaMS…) pode ter visto estruturas da validação, e aí o CV
  infla. Reporte por subconjunto e desconfie de ganho só em c1/nplib.
- Todo membro da fusão roda o `predict` no rerun (9 h, sem internet). Pesos precisam estar em dataset do Kaggle.

## Ideias abertas
- Modelo espectro→fingerprint (j1), fragmentação in silico (j2), análogos globais (j3), pool externo (j4),
  geração restrita à fórmula para a c3 (j5), loop no canal de cosseno (j6). Depois: LambdaRank, ICEBERG/CFM-ID, DreaMS.
