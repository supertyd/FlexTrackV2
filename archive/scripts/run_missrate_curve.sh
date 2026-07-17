#!/bin/bash
# Inference-time missing-rate degradation curve on VisEvent for one model.
# Runs 6 drop-rates (event modality), each full 320 seqs on 8 GPUs, then computes AUC/PR.
set -u
CFG=flextrackv2_b224_56_abl_rung0
cd /mnt/task_runtime/RGBE_workspace
rm -rf ./workspace/results/VisEvent_mr/smoke 2>/dev/null

for pair in "0.0 r000" "0.2 r020" "0.4 r040" "0.6 r060" "0.8 r080" "1.0 r100"; do
  set -- $pair; r=$1; tag=$2
  echo "=== rate $r (tag $tag) start $(date '+%H:%M') ==="
  python3 test_rgbe_missrate.py --yaml_name $CFG --rate $r --tag $tag --threads 8 --num_gpus 8 \
      > /mnt/task_runtime/ablation_logs/missrate_${tag}.log 2>&1
  n=$(ls ./workspace/results/VisEvent_mr/${tag}/${CFG}/ 2>/dev/null | wc -l)
  echo "    rate $r done, seqs=$n"
done

echo "=== computing curve (official VisEvent protocol) ==="
cd /mnt/task_runtime/VisEvent_SOT_Benchmark
python3 - <<'PY'
import eval_ours as eo, json, os
CFG='flextrackv2_b224_56_abl_rung0'
base='/mnt/task_runtime/RGBE_workspace/workspace/results/VisEvent_mr'
curve=[]
for r,tag in [(0.0,'r000'),(0.2,'r020'),(0.4,'r040'),(0.6,'r060'),(0.8,'r080'),(1.0,'r100')]:
    d=f'{base}/{tag}/{CFG}'
    m=eo.eval_dir(d) if os.path.isdir(d) else None
    curve.append({'rate':r,'metrics':m})
    print(f'rate {r}: {m}')
json.dump(curve, open('/mnt/task_runtime/missrate_curve_rung0.json','w'), indent=2)
print('wrote /mnt/task_runtime/missrate_curve_rung0.json')
PY
echo "=== MISSRATE_CURVE_DONE ==="
