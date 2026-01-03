"""
SoccerNet dataset utilities for football/soccer video analysis.

This package provides PyTorch Dataset classes and download utilities
for the SoccerNet tracking and calibration datasets.

Datasets:
    - PlayerTrackingDataset: Player bounding boxes with track IDs
    - LineDetectionDataset: Pitch line annotations (heatmap/segmentation/keypoint output)
    - LineKeypointDataset: Fixed-size keypoint tensors for regression

Download utilities:
    - download_tracking_data: Download player tracking annotations
    - download_pitch_calibration: Download pitch line calibration data

Example:
    >>> from torchkick.soccernet import (
    ...     PlayerTrackingDataset,
    ...     LineKeypointDataset,
    ...     download_tracking_data,
    ... )
    >>> 
    >>> # Download data
    >>> download_tracking_data("./data/tracking", splits=["train"])
    >>> 
    >>> # Load tracking dataset
    >>> tracking_ds = PlayerTrackingDataset("./data/tracking/train.zip")
    >>> 
    >>> # Load calibration dataset
    >>> calibration_ds = LineKeypointDataset("./data/calibration/train.zip")

Note:
    Requires the `soccernet` optional dependency for downloads:
    pip install torchkick[soccernet]
"""

from torchkick.soccernet.calibration_data import (
    LINE_CLASSES,
    CIRCLE_CLASSES,
    LineDetectionDataset,
    LineKeypointDataset,
    visualize_sample as visualize_calibration_sample,
)
from torchkick.soccernet.tracking_data import (
    PlayerTrackingDataset,
    tracking_collate_fn,
    visualize_sample as visualize_tracking_sample,
)
from torchkick.soccernet.download import (
    download_tracking_data,
    download_pitch_calibration,
)

__all__ = [
    # Constants
    "LINE_CLASSES",
    "CIRCLE_CLASSES",
    # Calibration datasets
    "LineDetectionDataset",
    "LineKeypointDataset",
    "visualize_calibration_sample",
    # Tracking datasets
    "PlayerTrackingDataset",
    "tracking_collate_fn",
    "visualize_tracking_sample",
    # Download utilities
    "download_tracking_data",
    "download_pitch_calibration",
]
