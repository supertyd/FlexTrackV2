#!/bin/bash
# Waits for flextrackv2_l224_56 to reach its epoch-40 checkpoint on the remote
# training node (qad3e9trm9), pulls it to this node (VOT22RGBD's 20GB
# workspace only lives here), then runs the live VOT22RGBD MultiStart/EAO
# eval via run_vot22rgbd_local.sh. Meant to run unattended in the background.
#
# Usage: ./wait_and_run_vot22rgbd_l224_56.sh

set -u
cd /mnt/task_runtime
CFG=flextrackv2_l224_56
REMOTE_TASK=qad3e9trm9
CKPT_REL="checkpoints/train/flextrackv2/${CFG}/FlexTrackV2_ep0040.pth.tar"
SSH_BASE="ssh -A -o StrictHostKeyChecking=no -i /root/.turibolt/bolt_ssh_key -F /root/.turibolt/ssh_config.v4 -o 'ProxyCommand=bolt_tunnel --proxy proxy-aws-10-bolt.corp.apple.com:443 --dest %h:%p --task_id ${REMOTE_TASK}' -p 31259 root@bolt-qad3e9trm9-u2mx7i7cwc.bolt-pods.turi-bolt.svc.cluster.local"

echo "########## [$(hostname)] Waiting for remote checkpoint ${CKPT_REL} on ${REMOTE_TASK} at $(date) ##########"
while true; do
  if eval "$SSH_BASE \"test -f /mnt/task_runtime/${CKPT_REL}\""; then
    break
  fi
  sleep 120
done

echo "########## [$(hostname)] Remote checkpoint ready, pulling it at $(date) ##########"
mkdir -p "checkpoints/train/flextrackv2/${CFG}"
bolt task scp "${REMOTE_TASK}:/mnt/task_runtime/${CKPT_REL}" "${CKPT_REL}"

echo "########## [$(hostname)] Running VOT22RGBD eval at $(date) ##########"
./run_vot22rgbd_local.sh "$CFG"

echo "########## [$(hostname)] VOT22RGBD DONE at $(date) ##########"
