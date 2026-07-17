"""
Scans ablation_results/*/metrics.json (populated by run_ablation_worker.sh /
compute_ablation_metrics.py on every worker machine, all pushed to the same
`ablations` branch) and regenerates the summary table appended to
ABLATIONS.md. Safe to re-run any time — it only rewrites the block between
the AUTO-GENERATED markers, so it doesn't conflict with the hand-written
parts of ABLATIONS.md.

Usage: python aggregate_ablation_results.py
"""
import glob
import json
import os

RESULTS_GLOB = "/mnt/task_runtime/ablation_results/*/metrics.json"
ABLATIONS_MD = "/mnt/task_runtime/ABLATIONS.md"
START_MARKER = "<!-- AUTO-GENERATED RESULTS START -->"
END_MARKER = "<!-- AUTO-GENERATED RESULTS END -->"


def fmt_box(d):
    if not d:
        return "—"
    return f"{d['auc']:.2f} / {d['pr']:.2f}"


def fmt_depth(d):
    if not d or "error" in d:
        return "—"
    return f"{d['precision']:.2f} / {d['recall']:.2f} / {d['fscore']:.2f}"


def build_table():
    rows = []
    for path in sorted(glob.glob(RESULTS_GLOB)):
        m = json.load(open(path))
        cfg = m.get("config", os.path.basename(os.path.dirname(path)))
        rows.append(
            "| {cfg} | {rgbt} | {rgbt_miss} | {vis} | {vis_miss} | {dt} | {dt_miss} |".format(
                cfg=cfg,
                rgbt=fmt_box(m.get("RGBT234")),
                rgbt_miss=fmt_box(m.get("RGBT234_miss")),
                vis=fmt_box(m.get("VisEvent")),
                vis_miss=fmt_box(m.get("VisEvent_miss")),
                dt=fmt_depth(m.get("DepthTrack")),
                dt_miss=fmt_depth(m.get("DepthTrack_miss")),
            )
        )
    header = (
        "| Config | RGBT234 (AUC/PR) | RGBT234_miss (AUC/PR) | "
        "VisEvent (AUC/PR) | VisEvent_miss (AUC/PR) | "
        "DepthTrack (Pr/Re/F) | DepthTrack_miss (Pr/Re/F) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    if not rows:
        return header + "| _(no results yet)_ | | | | | | |\n"
    return header + "\n".join(rows) + "\n"


def main():
    table = build_table()
    block = f"{START_MARKER}\n\n{table}\n{END_MARKER}"

    content = open(ABLATIONS_MD).read() if os.path.exists(ABLATIONS_MD) else ""
    if START_MARKER in content and END_MARKER in content:
        pre = content.split(START_MARKER)[0]
        post = content.split(END_MARKER)[1]
        content = pre + block + post
    else:
        content = content.rstrip() + "\n\n## Results (auto-generated)\n\n" + block + "\n"

    with open(ABLATIONS_MD, "w") as f:
        f.write(content)
    print(f"Updated {ABLATIONS_MD} with {len(glob.glob(RESULTS_GLOB))} config(s).")


if __name__ == "__main__":
    main()
