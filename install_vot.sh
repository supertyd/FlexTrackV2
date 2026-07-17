#!/usr/bin/env bash
set -euo pipefail
# VOT toolkit (only needed for DepthTrack / VOT-RGBD evaluation)
python -m pip install "git+https://github.com/votchallenge/vot-toolkit-python.git@v0.5.3"
