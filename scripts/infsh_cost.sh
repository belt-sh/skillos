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
  # belt task cost prints micro-cents in formatted output; "Charged:  $0.0002"
  out=$(belt task cost "$id" 2>/dev/null || echo "")
  dollars=$(echo "$out" | awk '/Charged:/ {gsub(/\$/,"",$2); print $2}')
  dollars=${dollars:-0}
  printf '  %-40s $%s\n' "$id" "$dollars"
  total=$(python3 -c "print($total + $dollars)")
done

avg=$(python3 -c "print($total / $count if $count else 0)")
echo
printf "Sample total: \$%.6f over %d task(s)\n" "$total" "$count"
printf "Avg per task: \$%.6f\n" "$avg"
