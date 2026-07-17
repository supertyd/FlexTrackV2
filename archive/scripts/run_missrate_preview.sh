#!/bin/bash
# FAST preview of the missing-rate degradation curve: strided subset (~25 seqs) of VisEvent.
set -u
CFG=flextrackv2_b224_56_abl_rung0
STRIDE=13   # 320/13 ~= 25 sequences, spread across the set
cd /mnt/task_runtime/RGBE_workspace

for pair in "0.0 r000" "0.2 r020" "0.4 r040" "0.6 r060" "0.8 r080" "1.0 r100"; do
  set -- $pair; r=$1; tag=$2
  echo "=== rate $r (tag $tag) start $(date '+%H:%M') ==="
  python3 test_rgbe_missrate.py --yaml_name $CFG --rate $r --tag $tag --stride $STRIDE \
      --threads 8 --num_gpus 8 > /mnt/task_runtime/ablation_logs/mrprev_${tag}.log 2>&1
  echo "    done, seqs=$(ls ./workspace/results/VisEvent_mr/${tag}/${CFG}/ 2>/dev/null | wc -l)"
done

echo "=== computing preview curve (official protocol, subset) ==="
cd /mnt/task_runtime/VisEvent_SOT_Benchmark
python3 - <<'PY'
import eval_ours as eo, json, os, numpy as np
CFG='flextrackv2_b224_56_abl_rung0'
base='/mnt/task_runtime/RGBE_workspace/workspace/results/VisEvent_mr'
# restrict the official eval to the subset that actually has predictions
rates=[(0.0,'r000'),(0.2,'r020'),(0.4,'r040'),(0.6,'r060'),(0.8,'r080'),(1.0,'r100')]
ALL=list(eo.SEQS)
curve=[]
for r,tag in rates:
    d=f'{base}/{tag}/{CFG}'
    if not os.path.isdir(d): curve.append({'rate':r,'metrics':None}); continue
    have=set(f[:-4] for f in os.listdir(d) if f.endswith('.txt'))
    # eval only sequences we have (restrict from the FULL list each time)
    eo.SEQS=[s for s in ALL if s in have]
    m=eo.eval_dir(d)
    curve.append({'rate':r,'metrics':m,'nseq':len(have)})
    print(f'rate {r}: {m} (n={len(have)})')
json.dump(curve, open('/mnt/task_runtime/missrate_preview_rung0.json','w'), indent=2)
print('wrote /mnt/task_runtime/missrate_preview_rung0.json')
PY
echo "=== MISSRATE_PREVIEW_DONE ==="
