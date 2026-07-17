import yaml
yaml_path = '/mnt/task_runtime/experiments/flextrackv2/flextrackv2_b224_56.yaml'
with open(yaml_path, 'r') as f:
    cfg = yaml.safe_load(f)
if 'AMP' in cfg['TRAIN']:
    del cfg['TRAIN']['AMP']
with open(yaml_path, 'w') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)
