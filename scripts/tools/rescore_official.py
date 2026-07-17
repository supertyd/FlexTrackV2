"""
Re-score ablation predictions with the OFFICIAL toolkits (not the homegrown
box-overlap AUC):
  - RGBT234        -> rgbt toolkit MPR / MSR (max over visible+infrared GT)
  - LasHeR         -> rgbt toolkit PR / SR
  - VisEvent       -> OPE AUC / PR with absent-frame exclusion (21 thresholds)
  - DepthTrack     -> already official VOT (taken from each config's B200 metrics.json)
Writes ablation_results_official/<config>/metrics.json
Usage: python3 rescore_official.py <config1> [<config2> ...]
"""
import sys, os, json, shutil
sys.path.insert(0, "/mnt/task_runtime/RGBT_toolkit_python/src")
sys.path.insert(0, "/mnt/task_runtime")
import numpy as np
from rgbt.dataset.lasher_dataset import LasHeR
from rgbt.dataset.rgbt234_dataset import RGBT234
from evaluate_lasher_visevent import evaluate_tracker_dataset

RES = "/mnt/task_runtime/workspace/results"
LASHER_GT = "/mnt/task_wrapper/user_output/artifacts/lasher/lasher/testingset"
RGBT234_GT = "/mnt/task_wrapper/user_output/artifacts/rgbt234/rgbt234"
VE_GT = "/mnt/task_wrapper/user_output/artifacts/visevent/visevent/test"

def ensure_seqtxt(gt, key='init.txt'):
    for s in os.listdir(gt):
        d = os.path.join(gt, s)
        if os.path.isdir(d):
            src, dst = os.path.join(d, key), os.path.join(gt, s + '.txt')
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)

def main():
    cfgs = sys.argv[1:]
    assert cfgs, "give config full-names"
    ensure_seqtxt(LASHER_GT); ensure_seqtxt(RGBT234_GT)

    # ---- RGBT234 (MPR/MSR) : register every config+variant, compute once ----
    rgbt = RGBT234(gt_path=RGBT234_GT, seq_name_path=os.path.join(RGBT234_GT, 'attr_txt/SequencesName.txt'))
    for c in cfgs:
        for suf, ds in [("", "RGBT234"), ("_miss", "RGBT234_miss")]:
            p = f"{RES}/{ds}/{c}"
            if os.path.isdir(p):
                rgbt(tracker_name=c+suf, result_path=p, bbox_type='ltwh')
    mpr, msr = rgbt.MPR(), rgbt.MSR()

    # ---- LasHeR (PR/SR) ----
    las = LasHeR(gt_path=LASHER_GT, seq_name_path=os.path.join(LASHER_GT, 'lashertest.txt'))
    for c in cfgs:
        for suf, ds in [("", "LasHeR"), ("_miss", "LasHeR_miss")]:
            p = f"{RES}/{ds}/{c}"
            if os.path.isdir(p):
                las(tracker_name=c+suf, result_path=p, bbox_type='ltwh')
    pr, sr = las.PR(), las.SR()

    # ---- VisEvent (OPE AUC/PR, absent-aware) ----
    ve_seqs = sorted(set(open(os.path.join(VE_GT, "testlist.txt")).read().splitlines()))

    for c in cfgs:
        m = {"config": c}
        def g(d, k): return round(d[k][0]*100, 2) if k in d else None
        m["RGBT234"]      = {"mpr": g(mpr, c),        "msr": g(msr, c)}
        m["RGBT234_miss"] = {"mpr": g(mpr, c+"_miss"),"msr": g(msr, c+"_miss")}
        m["LasHeR"]       = {"pr": g(pr, c),          "sr": g(sr, c)}
        m["LasHeR_miss"]  = {"pr": g(pr, c+"_miss"),  "sr": g(sr, c+"_miss")}
        for suf, ds, key in [("", "VisEvent", "VisEvent"), ("_miss", "VisEvent_miss", "VisEvent_miss")]:
            p = f"{RES}/{ds}/{c}"
            if os.path.isdir(p):
                auc, prv = evaluate_tracker_dataset(p, VE_GT, ve_seqs, "groundtruth.txt", is_visevent=True)
                m[key] = {"auc": round(auc, 2), "pr": round(prv, 2)}
            else:
                m[key] = None
        # carry over DepthTrack (official VOT) from the existing metrics.json if present
        old = f"/mnt/task_runtime/ablation_results/{c}/metrics.json"
        if os.path.exists(old):
            o = json.load(open(old))
            m["DepthTrack"] = o.get("DepthTrack"); m["DepthTrack_miss"] = o.get("DepthTrack_miss")
        outd = f"/mnt/task_runtime/ablation_results_official/{c}"
        os.makedirs(outd, exist_ok=True)
        json.dump(m, open(os.path.join(outd, "metrics.json"), "w"), indent=2)
        print(json.dumps(m))

if __name__ == "__main__":
    main()
