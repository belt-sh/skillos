#!/usr/bin/env bash
# Summarize inference.sh spend from output/infsh_tasks.jsonl.
#
# Usage:
#   scripts/infsh_cost.sh                       # sample N=20 newest tasks across all roles
#   scripts/infsh_cost.sh 100                   # sample 100 newest
#   scripts/infsh_cost.sh 20 executor           # only executor calls
#   SKILLOS_INFSH_TASKLOG=path scripts/...      # alt log location

set -euo pipefail

LOG="${SKILLOS_INFSH_TASKLOG:-./output/infsh_tasks.jsonl}"
SAMPLE="${1:-20}"
ROLE="${2:-}"

if [[ ! -f "$LOG" ]]; then
  echo "no task log at $LOG (set SKILLOS_INFSH_TASKLOG or run training first)" >&2
  exit 1
fi

if [[ -n "$ROLE" ]]; then
  ids=$(grep "\"role\": \"$ROLE\"" "$LOG" | tail -n "$SAMPLE" | sed -E 's/.*"task_id": "([^"]+)".*/\1/')
else
  ids=$(tail -n "$SAMPLE" "$LOG" | sed -E 's/.*"task_id": "([^"]+)".*/\1/')
fi

count=$(echo "$ids" | wc -l)
echo "Sampling $count task(s) from $LOG${ROLE:+ (role=$ROLE)}…" >&2

total=0
for id in $ids; do
  out=$(belt task cost "$id" --json 2>/dev/null || echo '{}')
  cost=$(echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('total_cost') or d.get('cost') or d.get('total') or 0)" 2>/dev/null || echo 0)
  printf '  %-40s %s\n' "$id" "\$$cost"
  total=$(python3 -c "print($total + ($cost or 0))")
done

avg=$(python3 -c "print($total / $count if $count else 0)")
echo
echo "Sample total: \$$total over $count task(s)"
echo "Avg per task: \$$avg"
