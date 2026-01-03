"""
Training scripts for football analysis models.

This module provides training pipelines for the various detection
and tracking models used in torchkick.

Submodules:
    train_yolo: YOLO player detection training
    train_fcnn: Faster R-CNN player detection training
    train_rtdetr: RT-DETR player detection training
    train_lines: Pitch line detection training

Example:
    Training via CLI (recommended for documentation):
    
    ```bash
    # YOLO player detection
    torchkick train yolo --data soccernet/tracking/train.zip --epochs 50
    
    # Faster R-CNN
    torchkick train fcnn --data soccernet/tracking/train.zip --epochs 10
    
    # RT-DETR (transformer-based)
    torchkick train rtdetr --data soccernet/tracking/train.zip --epochs 20
    
    # Pitch line detection
    torchkick train lines --data path/to/annotations --epochs 100
    ```
    
    Or via Python API:
    
    >>> from torchkick.training import train_yolo, train_fcnn
    >>> 
    >>> train_yolo.run_training(
    ...     data_zip="soccernet/tracking/train.zip",
    ...     epochs=50,
    ...     batch_size=256,
    ... )
"""

from torchkick.training.train_yolo import (
    convert_to_yolo_format,
    train_yolo,
    get_jersey_color_class,
)
from torchkick.training.train_fcnn import (
    get_player_detector_model,
    train_fcnn,
)
from torchkick.training.train_rtdetr import (
    get_rtdetr_model,
    train_rtdetr,
)
from torchkick.training.train_lines import (
    LineDetectionDataset,
    train_lines,
)

__all__ = [
    # YOLO
    "convert_to_yolo_format",
    "train_yolo",
    "get_jersey_color_class",
    # FCNN
    "get_player_detector_model",
    "train_fcnn",
    # RT-DETR
    "get_rtdetr_model",
    "train_rtdetr",
    # Pitch lines
    "LineDetectionDataset",
    "train_lines",
]
