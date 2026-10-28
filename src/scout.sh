#!/usr/bin/env bash
# Rotina diária do batedor: coleta o delta do Kaggle e roda o prompts/60_scout.md headless.
#
#   bash src/scout.sh                 # hoje
#   SCOUT_MODEL=opus bash src/scout.sh
#
# Agendado pelo Windows (ver README, "Batedor diário"): wsl.exe -e bash -lc '<repo>/src/scout.sh'
# Saída: artifacts/intel/<data>/ (bruto), artifacts/reports/intel/<data>.md (relatório),
# seção nova em notes/research.md e uma linha de custo em artifacts/reports/intel/usage.csv.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PATH="$HOME/.local/bin:$HOME/miniconda3/bin:$PATH"
DATA="$(date +%F)"
MODEL="${SCOUT_MODEL:-sonnet}"
OUT="$(python -c 'import sys; sys.path.insert(0, "src"); from common import ART; print(ART)')/reports/intel"
mkdir -p "$OUT/logs"
LOG="$OUT/logs/$DATA.log"

exec 9>"$OUT/.lock"
flock -n 9 || { echo "scout já está rodando" >>"$LOG"; exit 0; }
# um relatório por dia: rodar de novo no mesmo dia sobrescreveria o delta.md e duplicaria a seção
if [ -f "$OUT/$DATA.md" ] && [ -z "${SCOUT_FORCE:-}" ]; then
  echo "== $(date -Is) já rodou hoje ($OUT/$DATA.md); SCOUT_FORCE=1 para forçar" >>"$LOG"; exit 0
fi

{
  echo "== $(date -Is) coleta"
  python src/kaggle_intel.py
  rc=$?
  if [ "$rc" -eq 3 ]; then echo "sem novidade, LLM não roda"; exit 0; fi
  if [ "$rc" -ne 0 ]; then echo "coleta falhou (rc=$rc)"; exit "$rc"; fi

  echo "== $(date -Is) batedor ($MODEL)"
  PROMPT="$(sed -n '/<!-- PROMPT -->/,$p' prompts/60_scout.md | tail -n +2 | sed "s/{DATA}/$DATA/g")"
  claude -p "$PROMPT" \
    --model "$MODEL" \
    --max-budget-usd "${SCOUT_MAX_USD:-3}" \
    --output-format json \
    --permission-mode dontAsk \
    --allowedTools "Read" "Glob" "Grep" "Bash(python src/top.py:*)" \
      "Edit(./notes/research.md)" "Edit(./artifacts/reports/intel/**)" \
      "Write(./artifacts/reports/intel/**)" \
    --disallowedTools "WebFetch" "WebSearch" "Agent" \
    >"$OUT/logs/$DATA.json"
  echo "claude rc=$?"
  python - "$OUT/logs/$DATA.json" "$OUT/usage.csv" "$DATA" "$MODEL" <<'EOF'
import json, os, sys
src, csv, data, model = sys.argv[1:]
try:
    r = json.load(open(src))
except Exception as e:
    print("sem json do claude:", e); sys.exit(0)
u = r.get("usage", {})
row = [data, model, r.get("num_turns", ""), u.get("input_tokens", 0),
       u.get("cache_read_input_tokens", 0), u.get("cache_creation_input_tokens", 0),
       u.get("output_tokens", 0), r.get("total_cost_usd", "")]
new = not os.path.exists(csv)
with open(csv, "a") as f:
    if new:
        f.write("data,modelo,turnos,input,cache_read,cache_write,output,custo_usd\n")
    f.write(",".join(map(str, row)) + "\n")
print("uso:", row)
print(r.get("result", "")[:2000])
EOF
  echo "== $(date -Is) fim"
} >>"$LOG" 2>&1
