"""
SoccerNet dataset utilities for downloading and loading data.

This module provides download functions for SoccerNet tracking and
calibration datasets using the official SoccerNet SDK.

Example:
    >>> from torchkick.soccernet import download_tracking_data, download_pitch_calibration
    >>> 
    >>> # Download tracking data to local directory
    >>> download_tracking_data("./data/tracking")
    >>> 
    >>> # Download calibration data
    >>> download_pitch_calibration("./data/calibration")

Note:
    Requires the `soccernet` optional dependency:
    pip install torchkick[soccernet]
"""

from __future__ import annotations

import os
from typing import List, Literal


def download_tracking_data(
    local_dir: str,
    splits: List[Literal["train", "test", "challenge"]] | None = None,
    include_2023: bool = True,
) -> None:
    """
    Download SoccerNet tracking dataset.

    Downloads the player tracking annotations with bounding boxes, track IDs,
    and team labels for training detection and tracking models.

    Args:
        local_dir: Local directory to save downloaded files.
        splits: Dataset splits to download. Default is all splits.
        include_2023: Whether to also download the 2023 tracking challenge data.

    Raises:
        ImportError: If SoccerNet package is not installed.

    Example:
        >>> download_tracking_data("./soccernet/tracking", splits=["train", "test"])
    """
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError as e:
        raise ImportError(
            "SoccerNet package is required for dataset downloads. " "Install with: pip install torchkick[soccernet]"
        ) from e

    os.makedirs(local_dir, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=local_dir)

    if splits is None:
        splits = ["train", "test", "challenge"]

    downloader.downloadDataTask(task="tracking", split=splits)

    if include_2023:
        downloader.downloadDataTask(task="tracking-2023", split=splits)


def download_pitch_calibration(
    local_dir: str,
    splits: List[Literal["train", "test", "challenge"]] | None = None,
) -> None:
    """
    Download SoccerNet pitch calibration dataset.

    Downloads the camera calibration data with pitch line annotations
    for training homography estimation models.

    Args:
        local_dir: Local directory to save downloaded files.
        splits: Dataset splits to download. Default is all splits.

    Raises:
        ImportError: If SoccerNet package is not installed.

    Example:
        >>> download_pitch_calibration("./soccernet/calibration")
    """
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError as e:
        raise ImportError(
            "SoccerNet package is required for dataset downloads. " "Install with: pip install torchkick[soccernet]"
        ) from e

    os.makedirs(local_dir, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=local_dir)

    if splits is None:
        splits = ["train", "test", "challenge"]

    downloader.downloadDataTask(task="calibration", split=splits)


__all__ = [
    "download_tracking_data",
    "download_pitch_calibration",
]
