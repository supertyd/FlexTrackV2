import sys
import yaml

yaml_path = "/mnt/task_runtime/experiments/flextrackv2/flextrackv2_b224_54.yaml"
dataset_name = sys.argv[1]
upt = float(sys.argv[2])
uph = float(sys.argv[3])
inter = int(sys.argv[4])

with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

db_key = dataset_name.upper()
cfg["TEST"]["UPT"][db_key] = upt
cfg["TEST"]["UPH"][db_key] = uph
cfg["TEST"]["INTER"][db_key] = inter

with open(yaml_path, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)
print(f"Updated {dataset_name} to UPT={upt}, UPH={uph}, INTER={inter}")
