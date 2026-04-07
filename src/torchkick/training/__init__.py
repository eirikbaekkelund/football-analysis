"""
Training scripts for football analysis models.

Submodules:
    train_yolo_detection: YOLO player/ball detection training
    train_detection: RT-DETR-X player/ball/referee detection
    train_keypoints: ViTPose-L pitch landmark detection
    train_reid: DINOv2+LoRA+ArcFace player re-identification
    train_distill: Knowledge distillation ViT-L/14 → ViT-S/8

Example:
    >>> from torchkick.training import train_detection, train_yolo
    >>> weights = train_detection(data_config=[...], epochs=50)
    >>> ball_weights = train_yolo(data_zip="data/train.zip", epochs=30)
"""

from torchkick.training.train_pitch_heatmap import train_pitch_heatmap
from torchkick.training.train_yolo_detection import (
    convert_to_yolo_format,
    convert_dir_to_yolo_format,
    convert_soccernet_calibration_to_yolo_pose,
    build_soccernet_keypoint_dataset,
    train_yolo,
    get_jersey_color_class,
)
from torchkick.training.train_detection import train_detection
from torchkick.training.train_keypoints import train_keypoints
from torchkick.training.train_reid import train_reid, train_reid_from_video
from torchkick.training.train_distill import train_distill
from torchkick.training.train_body_pose import train_body_pose
from torchkick.training.train_pose_lifter import train_pose_lifter, PoseLiftMLP

__all__ = [
    # DINOv2 pitch heatmap detector
    "train_pitch_heatmap",
    # YOLO player/ball detection
    "convert_to_yolo_format",
    "convert_dir_to_yolo_format",
    "convert_soccernet_calibration_to_yolo_pose",
    "build_soccernet_keypoint_dataset",
    "train_yolo",
    "get_jersey_color_class",
    # RT-DETR-X player/ball/referee detection
    "train_detection",
    # ViTPose-L pitch keypoints
    "train_keypoints",
    # DINOv2+LoRA+ArcFace ReID
    "train_reid",
    "train_reid_from_video",
    # Knowledge distillation ViT-L/14 → ViT-S/8
    "train_distill",
    # ViTPose fine-tuning on FIFA body pose data
    "train_body_pose",
    # 2D→3D pose lifting MLP
    "train_pose_lifter",
    "PoseLiftMLP",
]
