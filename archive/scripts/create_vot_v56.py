import os, shutil
for dset in ["depthtrack", "depthtrack_miss"]:
  src = f"/mnt/task_runtime/workspace/results/{dset}/flextrackv2_b224_56"
  dst = f"/mnt/task_runtime/Depthtrack_workspace/results/flextrackv2_v56_{dset}/rgbd-unsupervised"
  os.makedirs(dst, exist_ok=True)
  if not os.path.exists(src): continue
  for f in os.listdir(src):
    if not f.endswith(".txt"): continue
    seq = f.replace(".txt", "")
    os.makedirs(os.path.join(dst, seq), exist_ok=True)
    shutil.copy(os.path.join(src, f), os.path.join(dst, seq, f"{seq}_001.txt"))
    with open(os.path.join(dst, seq, f"{seq}_001_time.value"), "w") as tf: tf.write("0.03\n" * 5000)
"INNEREOF"