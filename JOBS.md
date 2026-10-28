# JOBS — campanha atual

> Só o gerente edita este arquivo. Workers leem a própria linha.
> Cada job é independente (não depende de outro job desta campanha) e só cria arquivos novos.

Campanha: 1 (escrita pelo bootstrap em 2026-10-16). Referência: `cosine_boot_002` CV 0.2533
(c1 0.890 · c2 0.246 · c3 0). Ruído ≈ 0.006. Leia em LEARNINGS.md a armadilha da c2 antes de começar.

| id | tipo | objetivo | pai | orçamento | sucesso = |
|----|------|----------|-----|-----------|-----------|
| j1 | port/diverse | modelo espectro → fingerprint (GPU Kaggle) e score f·z das candidatas | cosine_boot_002 | 4 exp | c2 ≥ 0,32 com c1 ≥ 0,88 |
| j2 | fe | fragmentação in silico para ordenar isômeros | cosine_boot_002 | 4 exp | c2 ≥ 0,29 com c1 ≥ 0,88 |
| j3 | diverse | busca de análogos global (sem filtro de massa) e propagação por Tanimoto | cosine_boot_002 | 3 exp | c2 ≥ 0,27 com c1 ≥ 0,88 |
| j4 | original | pool externo (COCONUT + PubChemLite) como dataset e CV com o pool realista | cosine_boot_002 | 2 exp + dataset | dataset publicado + c2 realista medida |
| j5 | diverse | c3: gerar candidatas restritas à fórmula a partir de análogos | cosine_boot_002 | 3 exp | c3 ≥ 0,02 sem c1/c2 caírem mais que 0,01 |
| j6 | loop | melhorar o canal de cosseno (c1 e propagação) | cosine_boot_002 | 6 exp | CV ≥ 0,260 ou c1 ≥ 0,92 |
| j7 | port | reconstruir a linhagem pública (0,464) como família `meta`, um bloco por experimento | cosine_boot_002 | 6 exp | cada bloco com ganho medido por classe; base `meta` para os jobs do batedor |

## Detalhes

**j1 — fpmodel.** MLP (ou transformer pequeno) de picos binados → Morgan 2048 (+ MACCS opcional),
em torch, no Kaggle GPU (`kaggle_run.py run`). Treine **uma vez** nas linhas `folds == -1`, que valem
para todos os folds e para o teste. O treino roda no módulo, com cache, e o `predict(ctx)` só pontua.
Score da candidata = similaridade entre FP previsto e FP da candidata, combinado com o score de
`cosine_boot_002` (ex.: soma ponderada de ranks). Pode portar `aryanawasthiteykik/casmi26-gpu-fingerprint-fragmentation-ranker`
**se** o notebook treinar do zero no próprio código (regra 11); senão, implementação própria. Salve os
pesos no output do kernel. Registre ao menos o FP sozinho e o FP combinado.

**j2 — frag.** Para cada candidata da janela: quebras de 1–2 ligações (RDKit), massas dos fragmentos
±H, fração da intensidade da consulta explicada (tolerância max(0,002 Da, 10 ppm); use o modo/aduto
da consulta). Use como diferença para a melhor candidata da consulta e combine com o score de `cosine_boot_002`.
Pondere quebras raras. Cache por estrutura. CPU: local (máx. 3 processos) ou Kaggle `--no-gpu`.

**j3 — analog.** Índice esparso binado (ex.: 0,01 Da, sqrt da intensidade, L2) de todas as linhas
com `ctx.train_mask`, para cosseno contra o train inteiro sem filtro de massa. Teste também o cosseno
modificado (deslocamento pela diferença de precursor). Propague para as candidatas da janela com
Σ sim(q,a)^k · Tanimoto(c,a) nos top-N hits. Respeite `ctx.train_mask` em cada fold (o índice pode
ser global, filtrado por máscara).

**j4 — pool externo.** Baixe COCONUT e PubChemLite (ou um subconjunto do PubChem por massa
150–1.200 Da). Calcule `chem.structure_infos` (chave, massa, fórmula) e publique como dataset
**privado** no Kaggle; guarde uma cópia em `artifacts/data/extra/`. Experimento: `cosine_boot_002`
com pool = train ∪ externo, **menos `ctx.excluded`** (regra 5). Reporte quanto a c2 cai com o pool
realista e a cobertura das respostas por subconjunto (e180/np/nplib) no externo. No learnings, proponha
a integração em `common.candidate_pool` (src/ é do humano).

**j5 — c3 generativo leve.** A partir dos melhores análogos (top cosseno e top propagação), gere
estruturas novas **com a fórmula da consulta**: mover substituinte, trocar grupos isoméricos,
metilação/desmetilação + hidroxilação etc. (RDKit, SMARTS de reação). Pontue com a mesma evidência
espectral, nunca por "não existe no pool". Insira só a partir do rank 2, ou do rank 1 só com
confiança alta. O gerador não pode ler `ctx.excluded`, nem bancos externos que contenham essas
chaves. Gerar a resposta é o objetivo, mas ela tem que sair só das regras de edição. O audit dá
WARN quando c3 > 0: explique o gerador no learnings para o auditor. Meça a c3 por subconjunto.

**j6 — cosine loop.** Varie, um fator por experimento: similaridade de entropia × cosseno, tolerância
em ppm, top-N picos/potência, casar só o mesmo aduto (fallback para o modo), agregação de vários
espectros (média × máx × top-2), gamma/soma na propagação. Registre os intermediários. Não ajuste
nada no `test.parquet` visível.

**j7 — port-meta.** Base no nosso contrato para o que o batedor (`prompts/60_scout.md`) propuser todo dia.
Siga a linhagem descrita em `notes/research.md` §0 e §3, **um bloco por experimento** (`meta_j7_001`,
`--from` o anterior), para que cada ganho apareça no registry por classe: (1) fusão RRF por posição
`1/(3+r1) + 0,6/(3+r2)` sobre o motor de cosseno e a propagação; (2) FPNet próprio `f·z` (reuse o
modelo do j1 se já estiver registrado, senão treine do zero no Kaggle); (3) simuladores GLACIER/ICEBERG
com checkpoints MassSpecGym oficiais do `coleygroup/ms-pred`, reordenando dentro do grupo de fórmula;
(4) rank 1 único entre moléculas; (5) canal PubChem com promoção `rel > 150` e prior de popularidade
(só depois do pool do j4; antes disso, deixe pronto e meça com o pool do train). **Não use** pesos de
outros competidores sem código de treino público (`casmi26-fpnet-full1`, `v4b-models` etc., regra 11).
Os checkpoints MassSpecGym podem ter visto estruturas de validação: meça a sobreposição com
`ctx.excluded` e, se houver, registre o bloco como "só-teste" nas NOTES e no learnings (regra 5).
