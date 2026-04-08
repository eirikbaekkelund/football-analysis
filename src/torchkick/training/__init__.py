"""
Training scripts for football analysis models.

Submodules:
    train_pitch_heatmap: DINOv2 pitch keypoint heatmap training
    train_yolo_detection: YOLO player detection training
    train_reid: DINOv2+LoRA+ArcFace player re-identification training
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
from torchkick.training.train_reid import train_reid, train_reid_from_video

__all__ = [
    # DINOv2 pitch heatmap detector
    "train_pitch_heatmap",
    # YOLO player detection
    "convert_to_yolo_format",
    "convert_dir_to_yolo_format",
    "convert_soccernet_calibration_to_yolo_pose",
    "build_soccernet_keypoint_dataset",
    "train_yolo",
    "get_jersey_color_class",
    # DINOv2+LoRA+ArcFace ReID
    "train_reid",
    "train_reid_from_video",
]
