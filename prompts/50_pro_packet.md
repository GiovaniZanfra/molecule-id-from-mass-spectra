# 50 — Consultor Pro (fora do Cursor, turno único)

**Quando:** o time bateu num teto, ou você quer uma arquitetura/ideia que os workers não estão achando.
**Como:**
1. `python src/make_packet.py "tarefa concreta" --top 3` (pode pedir para um agente rodar). Gera packet.zip + PROMPT.md na pasta de artefatos (`packets/`).
2. Conversa NOVA no ChatGPT Pro / Claude no navegador: suba o packet.zip e cole o PROMPT.md.
3. Turno único. Se a resposta não servir, gere outro pacote com a tarefa ajustada em vez de insistir na mesma conversa.
4. Com o script que voltar: `python src/new_exp.py <familia> --author pro`, cole o código no arquivo criado, `--debug`, depois `kaggle_run.py run`.

Exemplos de tarefa: "escreva um NN tabular melhor que nn_j4_003", "proponha e implemente um modelo que aumente a diversidade do ensemble atual", "descubra por que fe_j2_* não melhora quando usa o dataset original".

O texto abaixo do marcador é o template que o make_packet.py preenche.

<!-- PROMPT -->
Você vai ajudar um time de agentes numa competição do Kaggle ({COMPETITION}, métrica {METRIC}). O pacote anexo tem as regras do time (AGENTS.md), a configuração (comp.json), a memória do time (LEARNINGS.md), a EDA, a auditoria, o ensemble atual, a biblioteca do contrato (src/common.py) e os scripts dos melhores modelos.

Melhores modelos atuais:
{TOP}

Arquivos no pacote:
{FILES}

Tarefa: {TASK}

Como responder:
- Entregue UM script Python completo no formato de experiments/_templates/gbdt.py: usa run_cv() (os folds do time), nada que use o target fora do fit_predict, termina em finish().
- Pense fora do que o time já tentou (veja LEARNINGS.md). Se repetir algo, explique por que agora seria diferente.
- Antes do código, no máximo 8 linhas: a ideia, por que deve ajudar o ensemble (diversidade conta) e o risco.
- Não invente arquivos que não estão no pacote; se precisar de algo que não existe, diga.
