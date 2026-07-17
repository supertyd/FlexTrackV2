from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.flextrackv2.config import cfg, update_config_from_file


def parameters(yaml_name: str, debug=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/flextrackv2/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    params.yaml_name = yaml_name
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.debug = debug

    # Network checkpoint path.
    #
    # FlexTrack-V2 is a single, self-describing model: the config carries the
    # exact weights it runs with via TEST.CHECKPOINT (relative to the project
    # root, or absolute). No hidden per-config redirect -- what the yaml says is
    # what loads. The legacy per-config path is kept only as a fallback so the
    # archived ablation configs (checkpoints/train/flextrackv2/<cfg>/...) still
    # resolve without special-casing.
    candidates = []
    if getattr(cfg.TEST, "CHECKPOINT", ""):
        ckpt = cfg.TEST.CHECKPOINT
        candidates.append(ckpt if os.path.isabs(ckpt) else os.path.join(prj_dir, ckpt))
    candidates.append(os.path.join(prj_dir, "checkpoints/FlexTrackV2.pth.tar"))
    epoch_val = cfg.TEST.EPOCH
    candidates.append(os.path.join(
        prj_dir, "checkpoints/train/flextrackv2/%s/FlexTrackV2_ep%04d.pth.tar" % (yaml_name, epoch_val)))
    candidates.append(os.path.join(
        prj_dir, "output/checkpoints/train/flextrackv2/%s/FlexTrackV2_ep%04d.pth.tar" % (yaml_name, epoch_val)))

    params.checkpoint = next((p for p in candidates if os.path.exists(p)), candidates[0])
    print("checkpoint: ", params.checkpoint)

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
