# Competição — Enveda CASMI 2026: Molecule ID from Mass Spectra

https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra
Estado da arte público (fórum + notebooks): `notes/research.md`.
Texto completo das páginas: API do Kaggle → `competition_list_pages(...)`; fórum: `kaggle competitions topics list|show`.

## Tarefa
Dado o(s) espectro(s) MS/MS de uma molécula desconhecida, prever a estrutura 2D
(SMILES). Saída **por molécula** (`molecule_id`), não por espectro: agregar todos
os espectros da molécula (1–16, mediana 3) num único ranking de até 25 SMILES.

## Métrica — MRR@25
`mean(1/rank)` do primeiro acerto; 0 se nenhum dos 25 acertar.
Acerto = mesmo **InChIKey14** (primeiro bloco) após canonicalização de tautômero
do RDKit (2026.03.3). Estereoquímica e tautômero não importam → deduplicar
candidatos por InChIKey14 antes de rankear (duplicata desperdiça posição).
Notebook da métrica: https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank

## Submissão
`submission.csv` com `molecule_id,smiles`; SMILES separados por `;`, melhor primeiro,
≤25, sem nulos, sem `molecule_id` repetido.
**Code competition**: notebook CPU/GPU ≤ 9h, **sem internet**. Dados/modelos
externos livres e públicos são permitidos (subir como Kaggle Dataset).
5 submissões/dia, 2 finais. Time ≤ 5. Vencedor licencia em MIT.

## Prazos
Entrada/merge: 2026-12-07 · Final: 2026-12-14 (23:59 UTC). Prêmio total US$ 50k.

## Dados
- `train.parquet` — 2,54M espectros, ~275k estruturas. Label = `normalized_smiles`
  (+ `inchikey14`, `molecular_formula`, `ingest_lib`, `precursor_error_ppm`, ...).
- `test.parquet` — o arquivo local (1.213 espectros / 400 moléculas) é **amostra do
  train**; no rerun vira o teste oculto: ~1.500 espectros, ~400 moléculas, todas
  Bruker timsTOF, massa monoisotópica 157–1.159 Da (mediana 348).
- Picos: `ms2_mzs` + `ms2_normalized_intensities` (base peak = 1). Picos > precursor+2 Da
  já removidos no teste. Usar `collision_energy_ev` (lista; vários valores = merge).
- 10 adutos no teste: [M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+, [M+Na]+, [M+K]+,
  [M-H]-, [M-H2O-H]-, [M+CH2O2-H]-, [M+Cl]-.

### Classes de novidade do teste (distribuição oculta)
1. Estrutura com espectro público → busca por similaridade na biblioteca de treino.
2. Estrutura conhecida (PubChem / COCONUT) sem espectro público → recuperação em banco de candidatos.
3. Estrutura nova, fora do PubChem → geração de novo.

### Bibliotecas do train (`ingest_lib`)
`enveda-180` (1,15M espectros, timsTOF igual ao teste, mas química sintética/drug-like),
`pluskal_ms2` (Orbitrap, NCE), `riken` (metabólitos de plantas), `gnps`, `massbank`,
`mona`, `spectraverse`, `msdial`, `drug_plus`, `masaryk`, e
**`enveda-np-examples`** (1.151 espectros / 250 produtos naturais, mesmo instrumento e
pipeline do teste → melhor proxy de validação).

## Esclarecimentos da organização (fórum, até 2026-10-22)
Organização: David Healey (Enveda), Marie Killian e `inversion` (Kaggle).
Número entre colchetes = id do tópico (`kaggle competitions topics show <slug> <id>`).

### Teste oculto e LB
- O LB público é calculado no **split público do teste oculto**, não no `test.parquet`
  local (que é só amostra do enveda-180) [741404]. As 3 classes aparecem nos dois splits;
  a proporção não será divulgada, mas "há estruturas novas suficientes para valer a pena" [745758, 748882].
- Classe 3 = fora do PubChem **e** fora do COCONUT [742274].
- Nenhum `molecule_id` se repete entre train, teste local e teste oculto; teste pequeno → cuidado com ruído no LB [745715].
- O teste oculto **não tem** ion mobility / CCS; colunas iguais às do `test.parquet` [743368, 743569].
- **Correção de massa do elétron**: o teste oculto foi corrigido (íons positivos estavam
  ~0,5 mDa pesados por erro de calibração). train (`enveda-180`) e o Zenodo do Enveda-180
  **não** foram corrigidos → subtrair **0,000548 Da** dos m/z em modo positivo para alinhar [743395, 746954, 748882].
- O train foi atualizado em 2026-09-15 (adutos de perda de água) [741471]. Nosso arquivo
  já tem `[M-H2O+H]+` e `[M-2H2O+H]+` (conferido).

### Métrica
- O scorer ao vivo é idêntico ao notebook `metric/casmi-mean-reciprocal-rank` v13 [742265, 748882].
- Desde 2026-10-07 aplica `RemoveStereochemistry` antes da canonicalização de tautômero,
  na resposta e no palpite; os scores antigos foram recalculados [744556, 746878, 749430].
- Entradas vazias são descartadas antes de ranquear (`A;;B` → B fica na posição 2);
  um `;` no final não conta [742274].
- Acrescentar palpites válidos e distintos depois do 1º não reduz o score [748882].
  SMILES inválido ocupa posição (comportamento do v13).

### Dados e modelos externos — permitidos
- A lista de aprovados **não é exaustiva**: vale qualquer dado público que cumpra as regras [748552].
- `train.parquet`: tratar como open source; modelos treinados nele podem ser abertos e
  redistribuídos [742265, 743236, 745841]. `enveda-np-examples` também é open source [742193].
- **Licença não-comercial (NC) é permitida** e não tira, por si só, a elegibilidade a prêmio ou medalha [745615, 748065, 748363].
- Estruturas: PubChem (subconjuntos, inclusive em dataset Kaggle **privado** durante a
  competição; contagens CID-PMID/Patent/SID como feature) [741857, 748882, 749209],
  COCONUT (a licença dele basta) [743234], PubChemLite, ChEBI, LIPID MAPS, LOTUS, NPAtlas, HMDB [745072, 748882].
- Espectros: Enveda-180 do Zenodo (record 21346580), inclusive as ~816 estruturas que não
  estão no train [743569, 745148]; releases novos de GNPS/MassBank/MoNA, inclusive CC BY-SA,
  usados como distribuídos [745228, 749217]; **GNPS-NIST14-MATCHES é exceção permitida** [745832].
- Modelos: MassSpecGym e modelos treinados nele [742991]; GLACIER/ICEBERG (checkpoints
  MassSpecGym do `coleygroup/ms-pred`) [744338]; espectros do atlas ICEBERG do PubChem [745134];
  DreaMS, inclusive o backbone SSL [745604, 748882]; CFM-ID 4 com os pesos de fábrica e espectros
  CFM-ID gerados do COCONUT [741912, 743774]; checkpoints oficiais do FRIGID [746430];
  TabPFN v2 e 2.5+ [748882]; ChemBERTa [742011]; DiffMS (Zenodo 15122968) e DualLGD
  (Zenodo 20281201), mesmo pré-treinados com HMDB [750252].
- Desde 2026-10-17 a restrição a dado open source com licença **não comercial** caiu de vez (David Healey) [750330].
- Em aberto (sem resposta até 22/10): checkpoints MIST do `coleygroup` (Zenodo 23303289, MIT,
  MassSpecGym, 20/10), MS2DeepScore 2.0, FIORA, SIRIUS/CSI:FingerID [751373].
- Usar a fórmula derivada de cada candidato como entrada de um modelo é permitido [745832].
- Treinar fora do Kaggle (nuvem privada, Hugging Face) com os dados de treino é permitido [741876].

### Proibido
- **NIST**: dados, ou qualquer modelo cujos pesos foram treinados com dados derivados do NIST
  → desclassificação [744470, 749217]. NIST usado só em avaliação/probe não contamina os pesos [745604, 748799].
  Também proibidos: METLIN e bibliotecas proprietárias de fabricantes (Thermo, Bruker, SciEx) [742193, 742991].
- Checkpoints do MassSpecGym Hub só valem se o treino excluiu espectros derivados do NIST [749217].
- **Reprodutibilidade**: todo peso usado precisa ter o código de treino original **e** acesso
  aos dados permitidos. Pesos compartilhados por outros competidores **sem código de treino
  não podem ser usados como estão**; retreine você mesmo com código próprio [745841, 748065, 748882].
  Vale também para pesos de competidores treinados só no `train.parquet` [750612] e para
  fine-tunes de modelos aprovados sem receita publicada [749217]. Envios antigos não reproduzíveis
  não desclassificam os futuros, mas não podem ganhar; se o autor publicar o código de treino
  até o fim, os pesos passam a valer [748882].
  Muitos notebooks públicos acima de 0,40 dependem desses pesos.
- A organização pode verificar **qualquer** submissão; submissão fora das regras sai do
  leaderboard, não só do prêmio [745615, 748882].

### Ambiente
- P100 foi descontinuada para submissão [742265].
- Não há suporte individual para falha no rerun; ver https://www.kaggle.com/code-competition-debugging [742463].

## Implicações para o nosso kit (que é tabular)
- Não é classificação/regressão: é **recuperação/ranking de estruturas** (ou geração).
  `comp.json` (binary/auc) e o template OOF por linha não se aplicam como estão.
- Split precisa ser **agrupado por `inchikey14`** (mesma estrutura nunca em dois folds),
  idealmente também por scaffold, para simular as classes 2/3.
- Recursos citados pela organização: SIRIUS, MIST-CF (fórmula), matchms, DreaMS/GeMS
  (embeddings pré-treinados), FragHub, MassIVE, anotações GNPS propagadas.
