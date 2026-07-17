#!/bin/bash
cd /mnt/task_runtime
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "[dtmiss] $(date +%H:%M:%S) 重跑 depthtrack_miss (参数对齐full: UPT0.77/UPH0.85/INTER55/MB862)"
/coreflow/venv/bin/python RGBT_workspace/test_depthtrack_mgpus.py --script_name flextrackv2 --dataset_name depthtrack_miss --yaml_name flextrackv2_b224_56 --mode parallel --threads 40 --num_gpus 8 --epoch 40 > /mnt/task_runtime/dtmiss_track.log 2>&1
echo "[dtmiss] $(date +%H:%M:%S) tracking done: $(ls workspace/results/depthtrack_miss/flextrackv2_b224_56/|grep -c '\.txt$')/50"
echo "[dtmiss] $(date +%H:%M:%S) VOT评测中..."
/coreflow/venv/bin/python -c "
import sys; sys.path.insert(0,'/mnt/task_runtime')
import grid_search_depthtrack_v2 as gs
ds=gs.DATASETS['depthtrack_miss']
gs.convert_to_vot_format(ds['raw_res_dir'], ds['vot_res_dir'])
gs.run(f\"cd {gs.WORKSPACE} && /coreflow/venv/bin/vot evaluate --workspace {gs.WORKSPACE} {ds['tracker']}\", gs.ENV_VOT, quiet=True)
gs.run(f\"cd {gs.WORKSPACE} && /coreflow/venv/bin/vot analysis --nocache --workspace {gs.WORKSPACE} {ds['tracker']}\", gs.ENV_VOT, quiet=True)
m=gs.parse_metrics(gs.latest_report(), ds['tracker']); t=ds['target']
print(f'RESULT depthtrack_miss(参数对齐full): Pr={m[\"Pr\"]:.2f} Re={m[\"Re\"]:.2f} F={m[\"F\"]:.2f} | FlexTrack Pr={t[\"Pr\"]} Re={t[\"Re\"]} F={t[\"F\"]}')
"
echo "[dtmiss] DTMISS_DONE"
