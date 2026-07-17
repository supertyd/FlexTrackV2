import sys
import os
import torch
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import vot
from lib.test.tracker.flextrackv2 import FlexTrackV2Tracker
import lib.test.parameter.flextrackv2 as rgbt_prompt_params
from lib.train.dataset.depth_utils import get_x_frame

class FlexTrackV2Vot(object):
    def __init__(self, yaml_name=None, epoch=None):
        yaml_name = yaml_name or os.environ.get('FLEXTRACKV2_YAML', 'flextrackv2_b224_54')
        epoch = epoch or int(os.environ.get('FLEXTRACKV2_EPOCH', 40))
        # 1. Load parameters
        params = rgbt_prompt_params.parameters(yaml_name, epoch)
        # 2. Build tracker
        self.tracker = FlexTrackV2Tracker(params, 'depthtrack')
        self.cfg = params.cfg
        self.dtype = getattr(self.cfg.DATA, 'XTYPE', 'rgbcolormap') # default to rgbcolormap for DepthTrack/RGBD
        
    def initialize(self, color_path, depth_path, rect):
        # rect is [x1, y1, w, h]
        image = get_x_frame(color_path, depth_path, dtype=self.dtype, depth_clip=True)
        init_info = {'init_bbox': rect}
        self.tracker.initialize(image, init_info)
        
    def track(self, color_path, depth_path):
        image = get_x_frame(color_path, depth_path, dtype=self.dtype, depth_clip=True)
        outputs = self.tracker.track(image, [1.0, 1.0])
        pred_bbox = outputs['target_bbox']
        return pred_bbox

# Main execution loop
if __name__ == '__main__':
    # Initialize tracker
    tracker = FlexTrackV2Vot()
    
    # Initialize VOT handle
    handle = vot.VOT("rectangle", channels="rgbd")
    selection = handle.region()
    
    # Get first frame paths
    first_frame = handle.frame()
    if isinstance(first_frame, (list, tuple)):
        color_path, depth_path = first_frame
    else:
        color_path = first_frame
        depth_path = None
        
    # Initialize tracker
    rect = [selection.x, selection.y, selection.width, selection.height]
    tracker.initialize(color_path, depth_path, rect)
    
    while True:
        frame_paths = handle.frame()
        if not frame_paths:
            break
            
        if isinstance(frame_paths, (list, tuple)):
            color_path, depth_path = frame_paths
        else:
            color_path = frame_paths
            depth_path = None
            
        bbox = tracker.track(color_path, depth_path)
        
        # Report prediction to VOT
        handle.report(vot.Rectangle(bbox[0], bbox[1], bbox[2], bbox[3]))
