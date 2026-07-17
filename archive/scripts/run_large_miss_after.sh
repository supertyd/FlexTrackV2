#!/bin/bash
cd /mnt/task_runtime
# wait for the full-modality grid search process to exit (GPUs are busy)
while kill -0 803737 2>/dev/null; do
  sleep 60
done
echo "$(date) full search done, starting miss search" >> gs_largemiss_driver.log
/coreflow/venv/bin/python -u grid_search_large_miss.py >> gs_largemiss_driver.log 2>&1
echo "$(date) ALL DONE" >> gs_largemiss_driver.log
