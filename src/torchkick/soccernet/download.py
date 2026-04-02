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


def download_roboflow_field_keypoints(
    output_dir: str,
    api_key: Optional[str] = None,
    version: int = 14,
) -> str:
    """
    Download football-field-detection-f07vi dataset (32-keypoint field landmarks).

    317 annotated images with 32 pitch keypoints in YOLO-pose format.
    Compatible with ``train_yolo_keypoints()`` — supports more landmark types
    than the SoccerNet 29-keypoint set.

    If ``api_key`` is provided, downloads via the Roboflow Python client.
    Otherwise, attempts a direct HTTPS zip download (no account required for
    public datasets).

    Args:
        output_dir: Directory to save the downloaded dataset.
        api_key: Optional Roboflow API key for authenticated downloads.
        version: Dataset version number (default 14).

    Returns:
        Path to the downloaded dataset directory.

    Example:
        >>> path = download_roboflow_field_keypoints("data/roboflow_field/")
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    if api_key is not None:
        try:
            from roboflow import Roboflow

            rf = Roboflow(api_key=api_key)
            project = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi")
            dataset = project.version(version).download("yolov8", location=output_dir)
            return dataset.location
        except ImportError:
            raise ImportError("roboflow package required for API download. Install: pip install torchkick[roboflow]")
    else:
        import zipfile

        import requests

        url = f"https://universe.roboflow.com/ds/XxFTJxfTJ7?key=roboflow-jvuqo-football-field-v{version}"
        zip_path = os.path.join(output_dir, "field_keypoints.zip")
        print(f"Downloading field keypoints dataset to {output_dir}...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        os.remove(zip_path)
        return output_dir


def download_roboflow_players(
    output_dir: str,
    api_key: Optional[str] = None,
    version: int = 2,
) -> str:
    """
    Download football-players-detection-3zvbc dataset (4-class player detection).

    372 annotated images with player, goalkeeper, referee, and ball classes
    in YOLO format. Useful for fine-tuning player detectors.

    If ``api_key`` is provided, downloads via the Roboflow Python client.
    Otherwise, attempts a direct HTTPS zip download.

    Args:
        output_dir: Directory to save the downloaded dataset.
        api_key: Optional Roboflow API key for authenticated downloads.
        version: Dataset version number (default 2).

    Returns:
        Path to the downloaded dataset directory.

    Example:
        >>> path = download_roboflow_players("data/roboflow_players/")
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    if api_key is not None:
        try:
            from roboflow import Roboflow

            rf = Roboflow(api_key=api_key)
            project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
            dataset = project.version(version).download("yolov8", location=output_dir)
            return dataset.location
        except ImportError:
            raise ImportError("roboflow package required for API download. Install: pip install torchkick[roboflow]")
    else:
        import zipfile

        import requests

        url = f"https://universe.roboflow.com/ds/XxFTJxfTJ8?key=roboflow-jvuqo-football-players-v{version}"
        zip_path = os.path.join(output_dir, "players.zip")
        print(f"Downloading player detection dataset to {output_dir}...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        os.remove(zip_path)
        return output_dir


__all__ = [
    "download_tracking_data",
    "download_pitch_calibration",
    "download_roboflow_field_keypoints",
    "download_roboflow_players",
]
