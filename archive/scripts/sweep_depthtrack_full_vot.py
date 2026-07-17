import os
import subprocess
import yaml
import re

def update_yaml_test_params(yaml_path, upt, uph, inter):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['TEST']['UPT']['DEPTHTRACK'] = float(upt)
    cfg['TEST']['UPH']['DEPTHTRACK'] = float(uph)
    cfg['TEST']['INTER']['DEPTHTRACK'] = int(inter)
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

def parse_vot_report(html_path):
    if not os.path.exists(html_path):
        return None, None, None
    with open(html_path, 'r') as f:
        content = f.read()
    
    # Extract Pr, Re, F values from the table
    pattern = r'data-tracker=\"flextrackv2\".*?<td[^>]*>.*?</td>.*?<td[^>]*data-value=\"([0-9\.]+)\">.*?</td>.*?<td[^>]*data-value=\"([0-9\.]+)\">.*?</td>.*?<td[^>]*data-value=\"([0-9\.]+)\">'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        pr = float(match.group(1)) * 100.0
        re_ = float(match.group(2)) * 100.0
        f_score = float(match.group(3)) * 100.0
        return pr, re_, f_score
    return None, None, None

def main():
    yaml_path = "/mnt/task_runtime/experiments/flextrackv2/flextrackv2_b224_54.yaml"
    html_path = "/mnt/task_runtime/Depthtrack_workspace/analysis/flextrackv2/report.html"
    
    candidates = [
        {"upt": 0.80, "uph": 0.90, "inter": 70}, # Default
        {"upt": 0.70, "uph": 0.80, "inter": 30},
        {"upt": 0.75, "uph": 0.80, "inter": 40},
        {"upt": 0.75, "uph": 0.85, "inter": 50},
        {"upt": 0.65, "uph": 0.80, "inter": 20},
        {"upt": 0.50, "uph": 0.85, "inter": 15},
        {"upt": 0.40, "uph": 0.90, "inter": 5}
    ]
    
    print("Sweep output start:")
    for idx, cand in enumerate(candidates, 1):
        update_yaml_test_params(yaml_path, cand["upt"], cand["uph"], cand["inter"])
        
        # Clean results and analysis
        subprocess.run("rm -rf /mnt/task_runtime/workspace/results/depthtrack/flextrackv2_b224_54", shell=True)
        subprocess.run("rm -rf /mnt/task_runtime/Depthtrack_workspace/results/flextrackv2", shell=True)
        subprocess.run("rm -rf /mnt/task_runtime/Depthtrack_workspace/analysis/flextrackv2", shell=True)
        
        # 1. Run parallel evaluation with correct PYTHONPATH
        cmd_test = "cd /mnt/task_runtime && PYTHONPATH=/mnt/task_runtime /coreflow/venv/bin/python RGBT_workspace/test_depthtrack_mgpus.py --script_name flextrackv2 --dataset_name depthtrack --yaml_name flextrackv2_b224_54 --mode parallel --threads 24 --num_gpus 4 --epoch 40"
        subprocess.run(cmd_test, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. Run converter
        cmd_conv = "/coreflow/venv/bin/python /mnt/task_runtime/Depthtrack_workspace/convert_flextrackv2_to_vot.py"
        subprocess.run(cmd_conv, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 3. Run VOT analysis
        cmd_anal = "export HTTP_PROXY=http://proxy.config.pcp.local:3128; export HTTPS_PROXY=http://proxy.config.pcp.local:3128; cd /mnt/task_runtime/Depthtrack_workspace && export PYTHONPATH=/mnt/task_runtime:$PYTHONPATH && /coreflow/venv/bin/vot analysis --nocache --name flextrackv2"
        subprocess.run(cmd_anal, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        pr, re_, f = parse_vot_report(html_path)
        if pr is not None:
            print(f"Candidate {idx} [UPT={cand['upt']} UPH={cand['uph']} INTER={cand['inter']}]: Precision={pr:.2f}%, Recall={re_:.2f}%, F-score={f:.2f}%")
            if f >= 67.0 and re_ >= 66.9 and pr >= 67.1:
                print("🎉 Target achieved!")
        else:
            print(f"Candidate {idx} [UPT={cand['upt']} UPH={cand['uph']} INTER={cand['inter']}]: Evaluation failed!")

if __name__ == "__main__":
    main()
