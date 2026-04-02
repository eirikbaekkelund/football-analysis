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
from typing import List, Literal, Optional


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


def _resolve_roboflow_api_key(api_key: Optional[str]) -> str:
    """Return api_key, loading .env then falling back to ROBOFLOW_API_KEY env var."""
    if api_key:
        return api_key
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise ValueError(
            "ROBOFLOW_API_KEY not found. Either:\n"
            "  1. Add ROBOFLOW_API_KEY=<key> to your .env file, or\n"
            "  2. Pass --api-key <key> on the command line.\n"
            "Get a free API key at https://roboflow.com."
        )
    return key


def download_roboflow_field_keypoints(
    output_dir: str,
    api_key: Optional[str] = None,
    version: int = 15,
) -> str:
    """
    Download football-field-detection-f07vi dataset (32-keypoint pitch landmarks).

    317 images, YOLO-pose format. Use with ``train_yolo_keypoints()``.
    Source: roboflow.com/roboflow-jvuqo/football-field-detection-f07vi

    Args:
        output_dir: Directory to save the dataset.
        api_key: Roboflow API key. Falls back to ``ROBOFLOW_API_KEY`` in ``.env``.
        version: Dataset version (default 15).
    """
    return download_roboflow_dataset(
        workspace="roboflow-jvuqo",
        project="football-field-detection-f07vi",
        version=version,
        output_dir=output_dir,
        api_key=api_key,
    )


def download_roboflow_players(
    output_dir: str,
    api_key: Optional[str] = None,
    version: int = 20,
) -> str:
    """
    Download football-players-detection-3zvbc dataset (4-class player detection).

    Player, goalkeeper, referee, ball — YOLOv11 format.
    Source: roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc

    Args:
        output_dir: Directory to save the dataset.
        api_key: Roboflow API key. Falls back to ``ROBOFLOW_API_KEY`` in ``.env``.
        version: Dataset version (default 20).
    """
    return download_roboflow_dataset(
        workspace="roboflow-jvuqo",
        project="football-players-detection-3zvbc",
        version=version,
        output_dir=output_dir,
        fmt="yolov11",
        api_key=api_key,
    )


def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int,
    output_dir: str,
    fmt: str = "yolov8",
    api_key: Optional[str] = None,
) -> str:
    """
    Download any Roboflow Universe dataset.

    Args:
        workspace: Roboflow workspace slug (visible in the dataset URL:
            ``roboflow.com/<workspace>/<project>``).
        project: Roboflow project slug.
        version: Dataset version number.
        output_dir: Directory to save the downloaded dataset.
        fmt: Export format (default ``"yolov8"``; use ``"coco"`` for COCO JSON).
        api_key: Roboflow API key. Falls back to ``ROBOFLOW_API_KEY`` in ``.env``.

    Returns:
        Path to the downloaded dataset directory.

    Example:
        >>> path = download_roboflow_dataset(
        ...     workspace="my-workspace",
        ...     project="football-players",
        ...     version=2,
        ...     output_dir="data/players/",
        ... )
    """
    # Do NOT pre-create output_dir — the Roboflow SDK skips downloading if
    # the target directory already exists.
    key = _resolve_roboflow_api_key(api_key)

    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("roboflow package required. Install: pip install torchkick[roboflow]")

    rf = Roboflow(api_key=key)
    dataset = rf.workspace(workspace).project(project).version(version).download(fmt, location=output_dir)
    return dataset.location


__all__ = [
    "download_tracking_data",
    "download_pitch_calibration",
    "download_roboflow_dataset",
    "download_roboflow_field_keypoints",
    "download_roboflow_players",
]
