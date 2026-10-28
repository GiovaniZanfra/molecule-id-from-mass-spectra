# Estado da arte público — CASMI 2026 (fórum + notebooks, até 2026-10-28)

Fontes: 118 tópicos do fórum e os ~20 notebooks mais votados/mais bem pontuados.
Números entre colchetes = id do tópico; `autor/slug` = notebook no Kaggle.
Regras e respostas da organização: `notes/competition.md`.
As seções 1–7 são o levantamento de 15/10; a seção 0 traz o que mudou até 22/10.

## Atualização 2026-10-28
**Placar.** Melhor notebook público: **0,477** (`busyaprime/lb-0-477-two-0-472-fixes-merged-the-right-way`, lido 28/10);
`heliosli/happy-to-make-the-dataset-public-if-it-helps` 0,475 (precisa de dataset privado); ~10 em 0,472. Era 0,464 em 22/10.
O topo do LB de times não aparece no delta (a faixa 0,469–0,504 da seção 0 vale como última leitura).

**Linhagem 0,465 → 0,477** (`busyaprime/...`, `alexchilton/...0-75`):
- Rede espectro↔estrutura `casmi-spectrum-model-v1` (sanpier) fundida por RRF `w/(3+rank)`: 0,466 com w=0,25.
- Leader-safe: rank 1 com w=0,25 e ranks 2–25 com w=0,50 → 0,468 (`romanrozen`); w=0,75 → 0,472; w=1,00 → 0,470; offset RRF 1 → 0,466.
- "Fragment check" sem treino, regra física para o rank 1 (`romantamrazov/casmi-2026-spectral-parallax`, 0,472).
  Empilhar os dois só vale se o check roda na ordem V2 e os ranks 2–25 vêm da ordem 0,75: 0,477.
- Tail rescue (3 últimas posições para candidatas de fora): 0,468 → 0,468.
- Cada passo é ≤ 1 molécula (≈ 0,008; ~125 moléculas no público). A rede é peso de terceiros: sem código de treino, não é elegível a prêmio.

**Regras (organização, 26–28/10).**
- Uso automático e sem rótulo de similaridade entre espectros de moléculas diferentes do teste: permitido [751373, 752297].
- Não reproduzível sai do LB, não só do prêmio, e o LB é recalculado [748882, 752541].
- Peso sem código de treino: não elegível; retreinar com código próprio e dados permitidos vale [745072, 749397].
- Tabela de candidatas alterada do original exige o código dos passos; basta código + dados + versão/data, sem igualdade bit a bit [748882, 751373].
- SIRIUS/CSI:FingerID: não. Modelos só de bibliotecas públicas e modelos químicos de linguagem/grafo: sim. PubChemLite: sim [751373].
- Atlas ICEBERG do PubChem (previsões) e a lista de membros: liberados em Dataset privado com o código de recuperação [745134].
  Pendência: o ICEBERG do atlas vem de modelo treinado com NIST'23; confirmar com o humano antes de usar (regra 11).
- Org não divulga a classe de cada molécula; pode usar versões mais novas das fontes do train [750684].

**Classe 3.** Mais um negativo: biotransformação por regras sobre análogos recupera poucas c3 [753400, só título]. Nenhum ganho público de c3 desde 22/10 [751300].
Corpo dos tópicos novos de 28/10 veio vazio; só títulos/comentários foram lidos.

## 0. Atualização 2026-10-22
**Placar.** LB público: topo 0,504 · 0,499 · 0,495 · 0,489 · 0,484; ~15 times entre 0,469 e 0,475.
Melhor notebook público: **0,464** (`dmitriigluzdov/casmi-26-the-data-behind-spectra-to-structures`);
`flexonafft/casmi26-closing-the-evidence-gap` 0,462. Do topo ao público ≈ 0,04 ≈ 5 moléculas.

**Linhagem 0,446 → 0,464** (scores por versão em `dariushafshar/casmi26-the-public-chain-version-by-version`):
- PubChem promovido com `rel > 150` em vez de 250: +0,003 a +0,006 em 4 bases diferentes (0,451).
- ChEBI/LIPID MAPS CC BY (`sergiomalv`) no pool do motor 2 (≈ +0,005).
- **N1A**: sem espectro `[M+H]+`, os de `[M+NH4]+` / `[M-H2O+H]+` viram proxies `[M+H]+` para o GLACIER (0,452).
- PubChemLite CC0 no pool do motor 2 (0,455). Em duas bases deu −0,003 e +0,004: ruído.
- Peso da popularidade no canal PubChem 0,25 → 0,15 (0,456).
- **N1 challenger** (0,462–0,464): mantém os proxies GLACIER onde já há ICEBERG e reordena
  por ICEBERG + GLACIER-proxy juntos. Deu +0,008 (≈ 1 molécula) nas duas bases em que foi testado.
- Ablações do Dmitrii: tirar os candidatos que só existem no PubChem derruba **0,464 → 0,409**
  (o canal PubChem, ~105M estruturas a ±10 ppm, vale ~0,055 no público). Gerar a partir de 20
  análogos em vez de 6 piorou (0,456). 1 ou 5 FPNets fold-safe: igual.
- Quem mediu a cadeia inteira: 94% dos passos são menores que 1 erro-padrão (~0,04 com ~132
  moléculas); rodar o mesmo código de novo varia até 0,006. Os ganhos reais vieram de
  **evidência nova** (simuladores, segundo motor) e **fonte nova de candidatos**, não de pesos.

**Em teste nos notebooks públicos (sem score confiável):**
- `ahmedberatozer/casmi26-v23a-fpnet-s-views`: FPNet treinado com 25–50% de espectros `[M+H]+`
  simulados pelo GLACIER para estruturas só do COCONUT (holdout NP +0,016 a +0,031; sintético −0,006 a −0,016).
  O treino roda num kernel que não é público.
- `ahmedberatozer/casmi26-v25-unq`: rank 1 único entre moléculas (o teste não repete estrutura;
  se duas linhas têm o mesmo top-1, só a de maior evidência fica com ele e a outra troca 1↔2).
- `ahmedberatozer/casmi26-v18-c3r3`: canal de classe 3 (derivação de análogos mais larga) inserido no rank 3.
- Dmitrii: prior de documentação ampliado (depositor de espectro, COCONUT/LOTUS/NPAtlas,
  sinônimos PubChem, literatura LOTUS) e 5 FPNets fold-safe no canal PubChem: sem ganho visível no LB.
  Os scripts de treino desses FPNets são públicos (Apache-2.0), ao contrário do `casmi26-fpnet-full1`.
- `saddamaditya`: "dual gate", que também promove com `rel > 130 e pop ≥ 2,5`.

**Classe 3 continua sem ganho público.** Dois times com validação própria de classe 3: melhora
local, LB parado há duas semanas [751300]. O gerador do motor 1 (edição de análogos) produz a estrutura
certa para 100 de 130 moléculas fora dos bancos, mas só 7 ficam em 1º [750241]; abrir mais pais
piora (Dmitrii). Fragmentos de moléculas fora do PubChem também quase nunca estão catalogados
(300–600 Da: 5% contra 9%) → precisa de gerador, não de busca. Ninguém publicou ganho com
DiffMS, FRIGID ou MIST.

**Ambiente.** Cópias de notebooks rodando na imagem nova (Python 3.13) pontuaram até 0,014 abaixo
do original; fixar a imagem ("Pin to original environment") [749908].

## 1. Placar e ruído
- LB público (15/10): topo 0,495 · 0,491 · 0,486 · 0,484; ~10 times entre 0,46 e 0,47.
  Melhor notebook público: **0,446** (`seyitkaangunes/casmi26-e006-pubchem-promo-rel-150-lb-0-446`).
- O LB público tem **~130 moléculas** (estimativa dos participantes): 1 molécula de "fora do
  top 25" para "rank 1" ≈ **+0,008**. Ruído de reexecução/seed do ranker ±0,006; seed de modelo
  re-treinado move até ±0,02 em validação [742088, 743254, 748524]. Ganho < ~0,01 no LB é ruído.
- O private são as outras ~270 moléculas, com as 3 classes nos dois splits.

## 2. Composição estimada do teste (sondagem do LB; não oficial)
- Busca por cosseno pura: MRR ≈ 0,94 na classe 1 e 0 nas outras → LB 0,150 ⇒ **~16% classe 1**
  (`aryanawasthiteykik/casmi26-validation-that-behaves-like-the-test`).
- Estimativas de partilha: 16/45/39 (prvsiyan) e 28/20/52 (Bull, ajustado em vários envios) [741597, 748524].
  Consenso: **classe 3 é grande (~40–50%)**, e é aí que está a folga que resta.
- Teto de qualquer solução só de recuperação ≈ 0,16 + 0,45 ≈ 0,6.

## 3. A linhagem pública (0,44x) — como funciona
Cadeia: `prvsiyan` analog propagation (0,34) → `ahmedberatozer` motor v4 com FPNet (0,40) →
fusão de dois motores + canal PubChem (0,436) → `canqiang` E006 (0,440) → v7c MetFrag-lite →
`antriksharya` e2rank2 (0,442) → promoção PubChem `rel>150` (0,446). ~2h40 em T4.

1. **Motor 1** (v4n): pool ~711k = estruturas do train ∪ COCONUT; janela de massa ±8,5–10 ppm;
   **FPNet** (transformer espectro → 6.930 bits de fingerprint; score do candidato = `f·z`, um
   matmul); gerador de análogos (modificações em análogos espectrais); 4 LightGBM → top 60.
2. **Motor 2** (prvsiyan + megayak): 4 canais — busca em biblioteca (entropia/cosseno),
   *analog propagation* `Σ sim(q,a)^4 · Tanimoto(c,a)`, heurísticas de fragmentação, FPNet;
   31 features → HistGBM / dois rankers → top 40 [741745].
3. **Canal PubChem**: `f·z` + prior de popularidade `log(1+substâncias)+log(1+PubMed)`;
   promove para rank 1 só se `(S>6 e pop≥5) ou rel>150` (rel = ganho de `f·z` sobre o melhor do pool).
4. **Simuladores diretos** ICEBERG/GLACIER (pesos MassSpecGym) reordenam dentro do mesmo grupo
   de fórmula; GLACIER só em `[M+H]+`, ICEBERG como fallback; MetFrag-lite onde não há simulador.
5. **Fusão por posição** (RRF) por identidade da métrica: `1/(3+r1) + 0,6/(3+r2)`; o motor 2
   funciona como veto da promoção; deduplicação por InChIKey14; top-1 do motor 2 inserido no rank 2.

**Riscos dessa linhagem**
- **Conformidade**: depende de pesos de outros competidores (`casmi26-fpnet-full1`,
  `v4b-models`, `v3-models`…) sem código de treino publicado. Pela organização, isso **não
  pode** ser usado como está numa solução elegível (`notes/competition.md`).
- **Overfit ao LB**: cada passo de +0,002 a +0,008 foi escolhido só pelo LB de ~130 moléculas,
  sem CV [748068]. ICEBERG já foi "+0,013" num fork e "−0,015" em outro.

## 4. O que se sabe (medido por mais de uma pessoa)
**Classe 1 está resolvida.** Busca por cosseno/entropia dá MRR ≈ 0,91–0,94 com a estrutura na
biblioteca, mas vale no máximo ~0,16 do score [741404, 744417].

**O gargalo na classe 2 é separar isômeros, não achar candidatos.**
- A fórmula certa é encontrada em >95% dos casos [745468]. Quando a resposta está na lista mas
  não em 1º, o vencedor tem a mesma fórmula em ~95–98% dos casos [743254, 742055].
- Em simulação de classe 2: ranking vale **17×** mais MRR que recall; separar perfeitamente os
  isômeros de mesma fórmula fecharia 95% da folga (0,73 → 0,99) [742055].
- Ganho de ranking encolhe muito no LB: +0,14 em validação vira +0,03–0,04 no LB [748524, 749251].

**Fontes de candidatos**
- PubChem contém ~90–96% das respostas em validação disjunta por estrutura, mas deixa ~10k
  candidatos por molécula (~2,5k mesmo sabendo a fórmula) [744417, 748552].
- Expandir o pool com PubChem sem filtro **piora** (MRR classe 2 0,52→0,35; LB 0,335→0,205).
  Só funciona com promoção controlada [741659, 742055].
- ChEBI, LIPID MAPS, NPAtlas e LOTUS são ~95–100% subconjuntos do PubChem: quase nada novo [748552].
  COCONUT cobre 99,6% de `enveda-np-examples`, mas ~35% numa validação disjunta [744417].

**Sinais que funcionam no ranking**
- **Fingerprint previsto do espectro** (MLP 3×4096 → Morgan 2048 + MACCS, treinado em T4 em
  2,8h): acerto no rank 1 sobe de 21% para 39% em moléculas sem espectro público
  (`aryanawasthiteykik/casmi26-gpu-fingerprint-fragmentation-ranker`).
- **Fragmentação in silico**: cortes de 1–2 ligações e fração da intensidade explicada, usada
  como diferença para o melhor candidato da molécula. S2 0,50 → 0,63; LB +0,03. Tolerância
  usual: max(0,002 Da, 10 ppm). Um time viu 2 ligações piorar sem ponderar raridade [744417].
- **Análogos**: busca em ~980k espectros sem filtro de massa + Tanimoto com os melhores hits: S2 0,63 → 0,66.
- **LambdaRank** por molécula, 1 linha por InChIKey14.
- **Segundo motor independente** fundido por RRF: só +0,001 [749251].

**Armadilhas medidas**
- **Prior de popularidade**: +0,50 em validação com moléculas de biblioteca, só +0,075 no LB.
  Moléculas sem espectro público estão ~0% no topo por popularidade
  (`aryanawasthiteykik/casmi26-the-popularity-prior-trap`). Validar priors em estruturas do
  `enveda-180` **sem** espectro público.
- **Modelos treinados no MassSpecGym** (ICEBERG etc.): +0,02 em validação vira ~0 quando se
  avaliam só estruturas fora do MassSpecGym (memorização) [748524].
- **Fine-tuning do FPNet em timsTOF** melhorou a validação e piorou o LB [742055].
- **Candidatos derivados** (±CH₂, ±O, ±hexose em análogos) não renderam no LB [742055].
  Inserir candidatos gerados pode **piorar** o MRR: um falso positivo no rank 1 custa 0,5 [741659].
  Ninguém mostrou ganho público na classe 3 até agora.
- **Nada pré-calculado a partir das massas do `test.parquet` visível**: o teste oculto é outro, e
  a fonte fica vazia sem erro [744172]. Não fixar 400 moléculas no código: um assert que falha
  faz o Kaggle pontuar o `submission.csv` anterior [749412].
- Deduplicar pela chave da métrica: ~8% das moléculas tinham um tautômero da resposta com outra
  chave no PubChem [743254]. Evitar SMILES inválido ou de enchimento (relatos de score 0) [742088].

## 5. Dados — fatos úteis
- Teste visível = linhas do `enveda-180`, idênticas byte a byte ao train [741404].
- O teste tem ~10× mais picos crus (mediana 230 contra 27), mas 82% ficam abaixo de 1%; com
  esse corte, 18 contra 14 [741404].
- Teste visível: 79% `[M+H]+`, 16% `[M-H]-`, sem dímeros, m/z 245–460. Train: 46% timsTOF,
  13% dímeros, m/z absurdos (2 a 2,4M) [744011]. Precursor timsTOF: 99% dentro de 5 ppm [743254].
- Os adutos da Enveda estão limpos; nas bibliotecas há ~12k `[M+CH2O2-H]-` mal rotulados (formato) [742087].
- `precursor_error_ppm` do train é em **módulo** e está **errado para `[M+CH2O2-H]-`** (mediana ~2.000 ppm,
  embora o precursor esteja certo). Recalcule com `chem.ADDUCT_SHIFT` (medição nossa, 15/10).
- 22 pares de espectros idênticos com rótulos de isômeros diferentes (msdial × riken) [744446].
- Íons positivos do `enveda-180`: subtrair 0,000548 Da (`notes/competition.md`).
- `enveda-np-examples`: as 250 estruturas também estão em outras bibliotecas. Separar por
  `ingest_lib` vaza a estrutura; é preciso separar por `inchikey14` [741597].

## 6. Validação — o consenso
1. Separar por **`inchikey14`** em **todas** as bibliotecas.
2. Simular as classes alterando o que se remove [748524]:
   - **C1**: tira só os espectros da biblioteca de origem e mantém os de outras bibliotecas;
   - **C2**: tira todos os espectros da estrutura, que continua entre os candidatos;
   - **C3**: tira espectros e candidatos de estruturas parecidas (Tanimoto ≥ 0,7).
3. Conjuntos de consulta que se comportam diferente; reportar separados:
   - `enveda-np-examples` (250; timsTOF, produtos naturais, mas todos com espectro público → infla priors);
   - estruturas do `enveda-180` **sem espectro público** (~500; parecidas com a classe 2);
   - produtos naturais de outras bibliotecas que estão no COCONUT e fora do MassSpecGym;
   - estruturas fora do COCONUT.
4. Montar cada molécula sorteando espectros (1–16, mediana 3), só com os 10 adutos do teste,
   repetindo o sorteio 3–5 vezes.
5. Estimar o LB ponderando o MRR de cada classe pelas partilhas (Bull: estimou 0,334, LB deu 0,322).
6. Comparar variantes **pareadas** nas mesmas moléculas, com bootstrap por estrutura e média de
   seeds. Ajustar hiperparâmetros no mesmo painel em que se mede infla o ganho ~3× [743254].

## 7. Ideias para os primeiros jobs
- **infra**: adaptar o kit para recuperação. Métrica MRR@25 (wheel RDKit 2026.03.3 +
  RemoveStereochemistry), folds por `inchikey14`, simulação de classes, e o artefato de
  experimento passa a ser a lista ranqueada por molécula de consulta.
- **baseline**: busca por cosseno/entropia (classe 1) + pool train ∪ COCONUT. Todo o código nosso.
- **fpmodel**: modelo próprio espectro → fingerprint (MLP/transformer) treinado nos nossos
  folds; score `f·z`. É o sinal central e precisa ser reprodutível.
- **pubchem**: índice próprio por massa/fórmula + contagens de popularidade, como dataset
  Kaggle privado, com promoção controlada.
- **frag**: features de fragmentação in silico e um ranker LambdaRank.
- **forward**: ICEBERG/GLACIER/CFM-ID oficiais (permitidos) para separar isômeros, avaliados
  só em estruturas fora do MassSpecGym.
- **diverse/classe 3**: geração com controle de confiança — FRIGID (checkpoints oficiais
  permitidos), decoder estilo MS2Mol/tutorial oficial (`inversion/casmi-denovo-tutorial-notebook`:
  encoder de picos + decoder Transformer de SMILES) ou edição de análogos restrita à fórmula.
- **research**: embeddings DreaMS (backbone SSL permitido) para busca de análogos.
