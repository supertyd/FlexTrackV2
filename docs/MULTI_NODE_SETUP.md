# Multi-Node Ablation Deployment — Setup Log

How the 9 V56 TPAMI ablations (see `ABLATIONS.md`) ended up running in
parallel across 9 machines, and every environment bug that had to be fixed
along the way. Kept as a reference so the same setup (or a future one like
it) doesn't have to rediscover these from scratch.

## Goal

Instead of running the 9 isolated ablation configs sequentially on one
8-GPU machine (~9x wall-clock), spread them across 9 machines: the original
node (`moe_big`, AWS `p4d.24xlarge` / A100) plus 8 new dedicated Bolt nodes
(one config each).

## Topology

| Config | Node type | Task ID |
|---|---|---|
| `moe_big` | AWS `p4d.24xlarge` (A100) | `92av62ravp` (pre-existing) |
| `moe_small`, `moe_middle`, `moe_hybrid`, `cma_fixed020`, `no_distill`, `pmax_015`, `pmax_025`, `pmax_050` | AWS `p6-b200.48xlarge` (B200) | 8 new nodes, submitted via `config-ablation-<name>.yaml` |

Code ships to each new node via `bolt task submit --git ...@ablations`
where possible; where Bolt's git access wasn't configured for a private
repo, via `bolt task scp` of a clean `git archive ablations` export instead
(see "Git access" below).

## Bugs found and fixed, in the order they surfaced

1. **`TensorboardWriter` import commented out.** `lib/train/admin/__init__.py`
   had it disabled from an old upstream commit; `lib/train/trainers/ltr_trainer.py`
   still imports it. Breaks training immediately. Fixed by restoring the
   import (the underlying `tensorboardX` dependency works fine here).

2. **`lib/train/admin/local.py` dataset paths one level too shallow.**
   The environment's actual HuggingFace-downloaded layout nests one extra
   directory (`lasher/lasher/trainingset`, `visevent/visevent/train`,
   `depthtrack/depthtrack/train/DepthTrackTraining`) compared to what the
   default config assumed.

3. **`download_missing.py` was never committed.** `provision_ablation_node.sh`
   calls it, but it only existed as an untracked local file on the original
   node, so fresh clones failed immediately at that step (with `set -e`
   aborting the whole script before `local.py` even got written). Committed
   it to the repo.

4. **`download_missing.py` had a proxy address that doesn't route from the
   new cluster.** It hardcoded a different proxy than the one
   `download_datasets_hf.py` successfully uses; swapped to match.

5. **Step-order bug in `provision_ablation_node.sh`.** It wrote the corrected
   `lib/train/admin/local.py` *before* calling
   `tracking/create_default_local_file.py`, but that script unconditionally
   *regenerates* `local.py` with generic `./data/<name>` placeholder paths —
   silently clobbering the real dataset paths. Every ablation training job
   crashed with `FileNotFoundError` on `.../data/visevent/train/trainlist.txt`.
   Fixed by writing `local.py` *last*.

6. **`p6-b200` (Blackwell/`sm_100`) GPUs cannot run the default environment's
   PyTorch at all.** This was the big one — not a version-pin fix:
   - The default venv has PyTorch 2.1.1+cu118 (Python 3.8). Every CUDA op
     fails: `RuntimeError: CUDA error: no kernel image is available for
     execution on the device`.
   - No PyTorch build for Python 3.8 ever added Blackwell support — PyTorch
     dropped Python 3.8 wheels around the 2.4.x series, *before* Blackwell
     (`sm_100`) support landed in 2.5+. Verified directly against the
     official `download.pytorch.org` wheel index: torch 2.7.1+cu126 ships
     `cp310`/`cp311`/`cp312` only, no `cp38`, at any CUDA version.
   - **Fix:** a separate `mci310` conda env (Python 3.10 + PyTorch 2.7.1,
     `conda-forge`'s `cuda126` build) built once per B200 node. Verified with
     a real 8-way NCCL `all_reduce` and a real `run_training.py` launch
     before rolling out to all 8 nodes. `run_ablation_worker.sh` picks up
     `MCI_PYTHON`/`MCI_TORCHRUN` env vars to select the interpreter, so
     non-B200 nodes are unaffected. Full package list and exact commands are
     in `ABLATIONS.md`'s "p6-b200 / Blackwell environment" section.
   - A few extra pip packages (`numba`, `setuptools<81` to unbreak
     build-isolated installs) were needed beyond `install.sh`'s list;
     `visdom` was skipped (build fails even with the setuptools pin, and
     it's not needed for training).

7. **LasHeR download truncated identically on every node that hit it.**
   `download_datasets_hf.py`'s LasHeR download (5 parts, ~50GB each) failed
   with a byte-for-byte identical truncation (2,486,314,258 bytes instead of
   53,687,091,200) on **5 independent nodes** — ruling out random network
   noise; almost certainly a proxy-side transfer duration/size cap on this
   one very large file. Confirmed the source file itself is not corrupted
   (real `Content-Length` on `huggingface.co` matches the expected size).
   `snapshot_download` raised on the checksum mismatch, but the per-repo
   `try/except` swallowed it and printed a misleading "All downloads and
   extractions completed!". Fixed two ways: `download_datasets_hf.py` now
   retries each dataset 3x and exits non-zero on final failure; and — since
   plain `curl -C -` resume doesn't work against HF's expiring signed
   redirect URLs — the recovery procedure uses `huggingface_hub.hf_hub_download`
   directly, which handles it correctly (documented in `ABLATIONS.md`).

8. **`cp -rn` (no-clobber) merge preserved stale pre-baked files.** The base
   Bolt image for 3 of the 8 nodes already had *some* files at
   `/mnt/task_runtime` (leftover from however the image was built). The
   no-clobber merge used to overlay the clean exported code skipped
   overwriting anything already present — including a stale, pre-fix copy of
   `lib/train/admin/__init__.py` — silently reintroducing bug #1 on those 3
   nodes. Fixed by forcing an overwrite (`cp -rf`) from the clean export.

9. **`scp -r` nesting gotcha.** On one node, a retried `scp -r <local dir>
   node:/mnt/task_runtime_new` landed one directory level deeper than
   intended because the destination directory already existed (scp nests the
   source *into* an existing target directory instead of populating it
   directly). Diagnosed via file count mismatch, fixed by clearing the
   destination first.

## Git access to a private repo from Bolt

`bolt task submit --git https://github.com/.../FlexTrack-V2.git@ablations`
initially failed with "Bolt does not have access to the provided git
repository" (private repo). Adding Bolt's own SSH key (`bolt config git-key
get`) as a read-only GitHub deploy key didn't help either, because outbound
SSH (port 22) is blocked on this network — `--git` over `ssh://` just hung.
Ended up not using `--git` for the bulk of the nodes at all: submitted bare
`sleep infinity` interactive tasks, then pushed a clean `git archive
ablations` export via `bolt task scp` and merged it in by hand. This also
sidestepped a real security concern the agent's own guardrails correctly
flagged: embedding a personal access token directly in `--git`'s URL would
have persisted that credential in Bolt's task configuration metadata
(visible to everyone with viewer access on the task), not just used it
ephemerally the way a local `git push` does.

## Secret handling

`HF_TOKEN` needed to reach each node's shell environment for the download
scripts, but never as literal text in a `bolt task ssh`-transmitted command
(that appears in the process table / task metadata). Settled on: write the
token to a local temp file, `bolt task scp` it to the node as a
restricted-permission file, and have a plain (secret-free) bootstrap script
`export HF_TOKEN="$(cat that_file)"; rm -f that_file` from inside itself —
so the only thing ever transmitted as command *text* is generic script logic,
never the secret value.

## Outcome

All 9 configs are training. Steady-state throughput on the B200 nodes
reached ~29 FPS per rank after the first epoch's cold-cache disk reads
(HuggingFace-downloaded datasets, freshly extracted, so the OS page cache
starts empty) — the first epoch alone can look deceptively stalled (sustained
98% CPU, 0% GPU, no loss lines yet) for several minutes before it's actually
just I/O-bound, not stuck.

## Evaluation-phase bugs (found once training finished, in the order hit)

Training succeeding didn't mean evaluation would — a second wave of bugs
only showed up once the 8 B200 nodes reached the test/eval stage, hours
after training completed:

10. **`.gitignore` excluded the entire `RGBT_workspace/` and
    `Depthtrack_workspace/` directories**, so `test_rgbt_mgpus.py`,
    `test_depthtrack_mgpus.py`, and the VOT workspace config
    (`config.yaml`, `dataset.json`) never shipped to any B200 node —
    evaluation failed immediately with `FileNotFoundError`. Fixed by
    un-ignoring just those small files (`/RGBT_workspace/*` +
    `!/RGBT_workspace/test_*.py`, same pattern for Depthtrack_workspace);
    the bulk `sequences/`/`results/`/`logs/` stay excluded.

11. **PyTorch 2.6+ changed `torch.load`'s default `weights_only` to `True`.**
    The B200 nodes' `mci310` env has PyTorch 2.7.1, so loading a checkpoint
    saved with an older PyTorch (containing a plain `AverageMeter` object)
    raised `WeightsUnpickler error`. Fixed by passing `weights_only=False`
    explicitly at all three `torch.load` call sites (test-time tracker init,
    training resume, training pretrained-init).

12. **`multiprocessing.Pool.starmap()` aborts the whole batch on the first
    worker exception**, silently discarding every sequence still
    queued/in-flight in the other workers — not just the one that failed.
    One bad "sequence" (RGBT234 ships a stray `attr_txt/` annotation
    directory alongside the real sequence directories, with no
    visible/infrared subfolders) or one genuinely too-small ground-truth box
    elsewhere could take an arbitrary chunk of otherwise-good sequences down
    with it, with zero visible error in the per-config log — a full-looking
    eval run had silently dropped 5-40% of its sequences. Fixed by wrapping
    `run_sequence()`'s body in try/except in both `test_rgbt_mgpus.py` and
    `test_depthtrack_mgpus.py`, and filtering `attr_txt` out of the RGBT234
    sequence listing.

13. **`vot-toolkit==0.5.3` wasn't installed in the B200 nodes' `mci310` env
    at all**, and once installed, its own code doesn't run on Python 3.10:
    - `vot/analysis/_processor.py` did `from collections import Iterable`
      (moved to `collections.abc` since Python 3.3, alias removed in 3.10).
    - `vot/analysis/tpr.py`'s `_Thresholds` analysis (needed to compute the
      actual Precision/Recall/F-score numbers) did `.astype(np.int)` — the
      deprecated NumPy alias for the built-in `int`, removed in NumPy 1.24+.
      This one is sneaky: `vot analysis` still exits 0 and prints "Analysis
      successful", but the specific result silently comes back `None`
      (`compute_ablation_metrics.py` then writes `null` for that dataset
      instead of raising, so nothing in the pipeline complains).
    Fixed both by patching the two files directly (`sed -i` for the one-line
    `collections`/`.astype` changes — no upstream release exists for either
    fix in this vot-toolkit version) on all 8 nodes' `mci310` install.

14. **`vot evaluate`'s `Workspace.download_dataset()` only skips its network
    fetch of the `votrgbd2021` reference dataset if `<sequences_dir
    >/list.txt` already exists** — and on the B200 nodes, the real DepthTrack
    sequences live under the HuggingFace download path
    (`/mnt/task_wrapper/user_output/artifacts/Depthtrack_workspace/
    Depthtrack_workspace/sequences`), not under `Depthtrack_workspace/`
    itself, so it decided the local dataset was incomplete and tried to
    download `votrgbd2021` from `data.votchallenge.net` — confirmed
    unreachable (HTTP 503) through this network's proxy regardless of
    `http_proxy`/`https_proxy` being set correctly. Fixed by symlinking
    `Depthtrack_workspace/sequences` to the real download path (now done
    automatically by `run_ablation_worker.sh`/`rerun_ablation_eval.sh`).

15. **Scaling eval to use all 8 GPUs (instead of just GPU 0) initially made
    things *worse*, not better.** PyTorch/numpy's BLAS/OpenMP layer defaults
    to using every core on the box per process; 32 parallel worker processes
    each grabbing all cores produced observed load averages of 235 (90-core
    node) and 1597 (178-core node) — an order of magnitude oversubscribed,
    leaving GPUs mostly idle while CPUs thrashed on scheduling overhead.
    Fixed by pinning `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/
    `OPENBLAS_NUM_THREADS`/`NUMEXPR_NUM_THREADS=1` so the process pool (not
    runaway per-process threading) is the only source of parallelism.

16. **Killing eval processes by script-name pattern
    (`pkill -f test_rgbt_mgpus.py`) doesn't work on their actual worker
    processes.** Once `multiprocessing` re-execs a worker via `spawn`, its
    `ps` command line becomes generic (`python3 -c "from multiprocessing
    .spawn import spawn_main; ..."`), no longer containing the original
    script name — so restarts kept leaving old workers running alongside new
    ones, compounding the CPU oversubscription above. Fixed by also matching
    `multiprocessing.spawn`/`multiprocessing.resource_tracker` when killing.
