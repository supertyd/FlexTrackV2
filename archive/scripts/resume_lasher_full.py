import os, sys, csv, json, time
sys.path.insert(0, '/mnt/task_runtime')
import grid_search_track as gs

track = 'lasher_full'
cfg = gs.TRACKS[track]
key = cfg['key']
gpus = [0, 1, 2, 3, 4, 5, 6, 7]

# state recovered from gs_lasher_full_results.csv / driver log:
# trial1 baseline pr=76.377 auc=61.298 (best so far)
# trial2 UPT=0.4  pr=76.321 (worse)
# trial3 UPT=0.6  pr=76.211 (worse)
# trial4 UPH=0.85 pr=75.990 (worse)
# trial5 UPH=0.98 pr=76.317 (worse)
# => current stays at baseline for UPT and UPH blocks
current = {'UPT': 0.5, 'UPH': 0.95, 'INTER': 20, 'MB': 500}
best_metrics = {'pr': 76.37746104927132, 'auc': 61.29847409339176}
trial_idx = 5

csv_path = f'/mnt/task_runtime/gs_{track}_results.csv'
csv_f = open(csv_path, 'a', newline='')
writer = csv.writer(csv_f)

def candidates(pname, val):
    if pname in ('UPT', 'UPH'):
        return sorted(set([round(max(0.05, min(0.98, val - 0.1)), 3),
                            round(max(0.05, min(0.98, val + 0.1)), 3)]))
    if pname == 'INTER':
        return sorted(set([max(5, int(val * 0.5)), int(val * 1.5) + 1]))
    if pname == 'MB':
        return sorted(set([max(100, val - 200), val + 200]))
    return []

for pname in ['INTER', 'MB']:
    base_val = current[pname]
    for cand in candidates(pname, base_val):
        if cand == base_val:
            continue
        trial_idx += 1
        overrides = {pname: cand}
        trial_name = f"flextrackv2_b224_56_gs_{track}_{pname}_{trial_idx}"
        gs.write_trial_yaml(track, trial_name, key, overrides)
        gs.log(track, f"trial {trial_idx}: {pname}={cand} (others={current}) -> running tracking [RESUMED, 8 GPUs]")
        gs.run_tracking(track, cfg, trial_name, gpus)
        metrics = gs.SCORERS[track](cfg, trial_name)
        gs.log(track, f"trial {trial_idx}: {pname}={cand} -> metrics {metrics}")
        writer.writerow([trial_idx, pname, cand, json.dumps(metrics)]); csv_f.flush()

        primary = list(cfg['targets'].keys())[0]
        if metrics.get(primary, -1) > best_metrics.get(primary, -1):
            gs.log(track, f"  -> improvement on {primary}: {best_metrics.get(primary)} -> {metrics.get(primary)}, locking in {pname}={cand}")
            current[pname] = cand
            best_metrics = metrics
    gs.log(track, f"after param {pname}: current={current}, best_metrics={best_metrics}")

ok = gs.meets_targets(best_metrics, cfg['targets'])
gs.log(track, f"=== FINAL for {track}: params={current} metrics={best_metrics} targets={cfg['targets']} MET={ok} ===")
with open(f'/mnt/task_runtime/gs_{track}_FINAL.json', 'w') as f:
    json.dump({'params': current, 'metrics': best_metrics, 'targets': cfg['targets'], 'met': ok}, f, indent=2)
csv_f.close()
gs.log(track, f"=== {track.upper()}_GRIDSEARCH_DONE ===")
