import sys
import os
import torch
import yaml
import numpy as np
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as f
from PIL import Image
from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints,
    coords_to_dict,
)

# Add this directory to sys.path so imports in inference.py work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


lines_coords = [
    [[0.0, 54.16, 0.0], [16.5, 54.16, 0.0]],
    [[16.5, 13.84, 0.0], [16.5, 54.16, 0.0]],
    [[16.5, 13.84, 0.0], [0.0, 13.84, 0.0]],
    [[88.5, 54.16, 0.0], [105.0, 54.16, 0.0]],
    [[88.5, 13.84, 0.0], [88.5, 54.16, 0.0]],
    [[88.5, 13.84, 0.0], [105.0, 13.84, 0.0]],
    [[0.0, 37.66, -2.44], [0.0, 30.34, -2.44]],
    [[0.0, 37.66, 0.0], [0.0, 37.66, -2.44]],
    [[0.0, 30.34, 0.0], [0.0, 30.34, -2.44]],
    [[105.0, 37.66, -2.44], [105.0, 30.34, -2.44]],
    [[105.0, 30.34, 0.0], [105.0, 30.34, -2.44]],
    [[105.0, 37.66, 0.0], [105.0, 37.66, -2.44]],
    [[52.5, 0.0, 0.0], [52.5, 68, 0.0]],
    [[0.0, 68.0, 0.0], [105.0, 68.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 68.0, 0.0]],
    [[105.0, 0.0, 0.0], [105.0, 68.0, 0.0]],
    [[0.0, 0.0, 0.0], [105.0, 0.0, 0.0]],
    [[0.0, 43.16, 0.0], [5.5, 43.16, 0.0]],
    [[5.5, 43.16, 0.0], [5.5, 24.84, 0.0]],
    [[5.5, 24.84, 0.0], [0.0, 24.84, 0.0]],
    [[99.5, 43.16, 0.0], [105.0, 43.16, 0.0]],
    [[99.5, 43.16, 0.0], [99.5, 24.84, 0.0]],
    [[99.5, 24.84, 0.0], [105.0, 24.84, 0.0]],
]


def projection_from_cam_params(final_params_dict):
    cam_params = final_params_dict["cam_params"]
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([[x_focal_length, 0, principal_point[0]], [0, y_focal_length, principal_point[1]], [0, 0, 1]])
    P = Q @ (rotation @ It)

    return P


class PnLCalibWrapper:
    def __init__(self, weights_kp, weights_line, config_kp, config_line, device='cuda:0'):
        self.device = device
        self.kp_threshold = 0.3434
        self.line_threshold = 0.7867
        self.pnl_refine = True

        # Load configs
        with open(config_kp, 'r') as f:
            self.cfg_kp = yaml.safe_load(f)
        with open(config_line, 'r') as f:
            self.cfg_line = yaml.safe_load(f)

        # Load models
        self.model_kp = get_cls_net(self.cfg_kp)
        self.model_kp.load_state_dict(torch.load(weights_kp, map_location=device))
        self.model_kp.to(device)
        self.model_kp.eval()

        self.model_line = get_cls_net_l(self.cfg_line)
        self.model_line.load_state_dict(torch.load(weights_line, map_location=device))
        self.model_line.to(device)
        self.model_line.eval()

        self.cam = None
        self.transform2 = T.Resize((540, 960))

    def inference(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        frame_tensor = f.to_tensor(frame_pil).float().unsqueeze(0)
        _, _, h_original, w_original = frame_tensor.size()

        if frame_tensor.size()[-1] != 960:
            frame_tensor = self.transform2(frame_tensor)

        frame_tensor = frame_tensor.to(self.device)
        b, c, h, w = frame_tensor.size()

        with torch.no_grad():
            heatmaps = self.model_kp(frame_tensor)
            heatmaps_l = self.model_line(frame_tensor)

        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        kp_dict = coords_to_dict(kp_coords, threshold=self.kp_threshold)
        lines_dict = coords_to_dict(line_coords, threshold=self.line_threshold)
        kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)

        self.cam.update(kp_dict, lines_dict)
        final_params_dict = self.cam.heuristic_voting(refine_lines=self.pnl_refine)

        return final_params_dict

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        if self.cam is None:
            self.cam = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)

        final_params_dict = self.inference(frame)

        if final_params_dict is not None:
            P = projection_from_cam_params(final_params_dict)
            return P
        return None
