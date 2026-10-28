# EDA — bootstrap (2026-10-16)

Scripts: `eda/01_meta.py` (bibliotecas, adutos, energia, precursor), `02_structures.py` (sobreposição,
isômeros, janela de massa), `03_queries.py` (painel de validação, o que o fold deixa ver),
`04_oof.py <exp>` (posição da resposta e Tanimoto do top-1 errado).

## Dados
- Train: 2,54M espectros, 274k estruturas (chave da métrica). Só metadados = 2 GB no pandas;
  picos limpos em mmap (107M picos, ~38 por espectro no enveda-180 após a limpeza padrão).
- `enveda-180` = 45% dos espectros e 67% das estruturas; 98,7% das estruturas dele só existem lá,
  e só 1,3% tem espectro público. Massa 235–499 Da (química de triagem). O `test.parquet` visível
  é amostra dele (mesma faixa 245–460, CE 20/40/60, 27% com 3 CEs fundidas, 3 espectros/molécula).
- O **teste oculto** vai de 157 a 1.159 Da (produtos naturais?). Melhor proxy: `enveda-np-examples`
  (250 estruturas, timsTOF, massa 133–1.109, CEs diferentes: 20/35/40/60/80). Olhe os subconjuntos
  `np` e `nplib` do CV, não só o total: o e180 pesa 500 das 1.346 consultas.
- 29% dos espectros do e180 têm aduto fora dos 10 do teste (dímeros etc.). Adutos das consultas:
  73% [M+H]+, 20% [M-H]-.
- Precursor contra a estrutura rotulada (mDa, mediana): e180 positivo **+0,71**, negativo +0,25;
  np-examples +0,68 / +0,16; bibliotecas públicas ~0. Positivo menos a massa do elétron (0,549)
  fica em ~+0,15. O teste oculto já vem corrigido; na validação, corrija as consultas timsTOF positivas.
  A janela de ±10 ppm sempre contém a resposta (só 3% das nplib/2 ficam fora).

## Dificuldade (janela de ±10 ppm no pool do train)
- e180: ~80 candidatas, 5 fórmulas, ~37 isômeros da mesma fórmula (bibliotecas combinatórias).
  np/nplib: 7–13 candidatas, 2–3 fórmulas, 3–7 isômeros. Separar isômeros é o problema da c2.
- Um prior de ppm (σ = 3), junto com a correção do elétron, corta as outras fórmulas: c2 vai de
  0,206 para 0,246 (`cosine_boot_001` → `cosine_boot_002`).
- Quando o top-1 está errado, ele é pouco parecido com a resposta (Tanimoto mediano de 0,20 no e180
  e de 0,21–0,30 no np/nplib): o espectro mais parecido na janela é de um isômero sem relação.

## Armadilha do painel (muda decisão de modelagem)
- **Na c2, a resposta é sempre uma candidata sem espectro (no modo da consulta) na biblioteca do
  fold.** Em 53% das c2 ela é a única candidata assim na janela, e em 68% a única com a mesma fórmula.
  Pôr "candidatas sem espectro" na frente levaria a c2 de 0,25 para ~0,65 sem saber nada de química.
  No teste real, o pool (COCONUT/PubChem) tem milhares de candidatas sem espectro, então esse sinal não existe.
- Consequência: nenhum score, feature ou ranker pode ganhar por "não ter espectro". Ranker
  aprendido (LambdaRank) com feature de cosseno próprio vai achar esse atalho. Precisa de pool com
  estruturas externas (sem espectro) antes de treinar ranker.
- A c3 simulada só tira a resposta do pool: os análogos (Tanimoto ≥ 0,7) continuam. É otimista
  para geração e para análogos, mas hoje todo método de recuperação dá c3 = 0.
