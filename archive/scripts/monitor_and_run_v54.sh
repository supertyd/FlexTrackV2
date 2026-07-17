#!/bin/bash
cd /mnt/task_runtime
export PYTHONPATH=/mnt/task_runtime:$PYTHONPATH
source /coreflow/venv/bin/activate

echo "=== Pipeline started: Monitoring download sessions ===" >> /mnt/task_runtime/pipeline.log

while tmux ls 2>/dev/null | grep -E 'hf_download|lasher_download' > /dev/null; do
    sleep 30
done

echo "=== Downloads complete! Verifying directories ===" >> /mnt/task_runtime/pipeline.log

# Verify that extraction succeeded or run final check
ls -la /mnt/task_wrapper/user_output/artifacts >> /mnt/task_runtime/pipeline.log 2>&1

echo "=== Starting V54 Training ===" >> /mnt/task_runtime/pipeline.log
bash /mnt/task_runtime/xtrain_54.sh >> /mnt/task_runtime/pipeline.log 2>&1

echo "=== V54 Training Finished! Starting V54 Evaluation ===" >> /mnt/task_runtime/pipeline.log
bash /mnt/task_runtime/xeval_54.sh >> /mnt/task_runtime/pipeline.log 2>&1

echo "=== FlexTrackV2 V54 Training and Testing Pipeline Finished! ===" >> /mnt/task_runtime/pipeline.log
