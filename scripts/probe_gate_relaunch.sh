#!/usr/bin/env bash
# Gate the natural sweep relaunch on executor health: re-probe every 5 min;
# relaunch scripts/run_natural_sweep.sh only after single probe + 10-burst
# all pass (OpenRouter "All providers ignored" outage, 2026-07-02).
set -u
cd "$(dirname "$0")/.."

while true; do
  RES=$(timeout 300 .venv/bin/python -c "
from concurrent.futures import ThreadPoolExecutor
from inferencesh import inference
from skillos.utils.infsh_auth import resolve_infsh_api_key
client = inference(api_key=resolve_infsh_api_key())
def one(i):
    try:
        r = client.tasks.run({'app':'openrouter/qwen3-8b','infra':'cloud','variant':'default','input':{'text':f'say {i}','max_tokens':8,'temperature':0.6}})
        return 'ok' if r and r.get('output') else 'empty'
    except Exception as e:
        return 'err'
res = [one(0)]
if res[0] == 'ok':
    with ThreadPoolExecutor(10) as p:
        res += list(p.map(one, range(10)))
print('PASS' if all(x=='ok' for x in res) and len(res)==11 else 'FAIL')
" 2>/dev/null | tail -1)
  echo "[$(date -u)] probe gate: $RES"
  if [ "$RES" = "PASS" ]; then
    echo "[$(date -u)] executor healthy — relaunching sweep"
    nohup bash scripts/run_natural_sweep.sh > /tmp/natural_sweep_orch.log 2>&1 &
    echo "RELAUNCHED"
    exit 0
  fi
  sleep 300
done
