"""
Dataset management and conversion utilities.

This module provides tools for managing annotation datasets, converting
between formats, merging datasets, and implementing active learning
strategies for efficient labeling.

Example:
    >>> from torchkick.annotation import DatasetManager
    >>> 
    >>> manager = DatasetManager("./datasets")
    >>> manager.merge_datasets(["dataset1", "dataset2"], "merged")
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    """
    Information about a managed dataset.

    Tracks dataset metadata, statistics, and configuration for
    versioning and reproducibility.
    """

    name: str = Field(..., description="Dataset name")
    version: str = Field(default="1.0.0", description="Dataset version")
    format: str = Field(default="yolo", description="Dataset format (yolo, coco, voc)")
    num_images: int = Field(default=0, description="Total number of images")
    num_annotations: int = Field(default=0, description="Total number of annotations")
    class_distribution: Dict[str, int] = Field(default_factory=dict, description="Annotations per class")
    train_split: float = Field(default=0.8, description="Training split ratio")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    source_videos: List[str] = Field(default_factory=list, description="Source video IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DatasetManager:
    """
    Manage annotation datasets for training.

    Provides dataset creation, merging, splitting, and versioning
    capabilities for maintaining clean training datasets.

    Args:
        base_dir: Base directory for all datasets.

    Example:
        >>> manager = DatasetManager("./datasets")
        >>> info = manager.create_dataset("players_v2", format="yolo")
        >>> manager.add_images(info.name, image_paths, label_paths)
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_dataset(
        self,
        name: str,
        format: str = "yolo",
        train_split: float = 0.8,
    ) -> DatasetInfo:
        """
        Create a new empty dataset.

        Args:
            name: Dataset name (used as directory name).
            format: Dataset format (yolo, coco, voc).
            train_split: Training data fraction.

        Returns:
            DatasetInfo for the created dataset.

        Raises:
            ValueError: If dataset already exists.
        """
        dataset_dir = self.base_dir / name
        if dataset_dir.exists():
            raise ValueError(f"Dataset already exists: {name}")

        dataset_dir.mkdir(parents=True)

        if format == "yolo":
            (dataset_dir / "images" / "train").mkdir(parents=True)
            (dataset_dir / "images" / "val").mkdir(parents=True)
            (dataset_dir / "labels" / "train").mkdir(parents=True)
            (dataset_dir / "labels" / "val").mkdir(parents=True)
        elif format == "coco":
            (dataset_dir / "images").mkdir(parents=True)
            (dataset_dir / "annotations").mkdir(parents=True)

        info = DatasetInfo(
            name=name,
            format=format,
            train_split=train_split,
        )

        self._save_info(info)
        return info

    def get_dataset(self, name: str) -> DatasetInfo:
        """
        Get dataset information.

        Args:
            name: Dataset name.

        Returns:
            DatasetInfo object.

        Raises:
            FileNotFoundError: If dataset doesn't exist.
        """
        info_path = self.base_dir / name / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Dataset not found: {name}")

        with open(info_path) as f:
            data = json.load(f)

        return DatasetInfo(**data)

    def list_datasets(self) -> List[DatasetInfo]:
        """
        List all managed datasets.

        Returns:
            List of DatasetInfo objects.
        """
        datasets = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and (path / "dataset_info.json").exists():
                datasets.append(self.get_dataset(path.name))
        return sorted(datasets, key=lambda d: d.name)

    def merge_datasets(
        self,
        source_names: List[str],
        target_name: str,
        deduplicate: bool = True,
    ) -> DatasetInfo:
        """
        Merge multiple datasets into one.

        Args:
            source_names: Names of datasets to merge.
            target_name: Name for merged dataset.
            deduplicate: Remove duplicate images by filename.

        Returns:
            DatasetInfo for merged dataset.
        """
        sources = [self.get_dataset(name) for name in source_names]

        # Check format compatibility
        formats = set(s.format for s in sources)
        if len(formats) > 1:
            raise ValueError(f"Cannot merge datasets with different formats: {formats}")

        format = sources[0].format
        target = self.create_dataset(target_name, format=format)
        target_dir = self.base_dir / target_name

        seen_files = set()
        total_images = 0
        total_annotations = 0
        class_dist: Dict[str, int] = {}

        for source in sources:
            source_dir = self.base_dir / source.name

            if format == "yolo":
                for split in ["train", "val"]:
                    images_dir = source_dir / "images" / split
                    labels_dir = source_dir / "labels" / split

                    for img_path in images_dir.glob("*"):
                        if deduplicate and img_path.name in seen_files:
                            continue
                        seen_files.add(img_path.name)

                        # Copy image
                        shutil.copy(
                            img_path,
                            target_dir / "images" / split / img_path.name,
                        )
                        total_images += 1

                        # Copy label
                        label_path = labels_dir / img_path.with_suffix(".txt").name
                        if label_path.exists():
                            shutil.copy(
                                label_path,
                                target_dir / "labels" / split / label_path.name,
                            )

                            # Count annotations
                            with open(label_path) as f:
                                for line in f:
                                    if line.strip():
                                        class_id = line.strip().split()[0]
                                        class_dist[class_id] = class_dist.get(class_id, 0) + 1
                                        total_annotations += 1

        # Update target info
        target.num_images = total_images
        target.num_annotations = total_annotations
        target.class_distribution = class_dist
        target.source_videos = []
        for source in sources:
            target.source_videos.extend(source.source_videos)
        target.metadata["merged_from"] = source_names

        self._save_info(target)
        return target

    def split_dataset(
        self,
        name: str,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """
        Re-split dataset into train/val sets.

        Args:
            name: Dataset name.
            train_ratio: Fraction for training set.
            shuffle: Randomly shuffle before splitting.
            seed: Random seed for reproducibility.
        """
        info = self.get_dataset(name)
        dataset_dir = self.base_dir / name

        if info.format != "yolo":
            raise NotImplementedError(f"Splitting not implemented for {info.format}")

        # Collect all images
        images = []
        for split in ["train", "val"]:
            images_dir = dataset_dir / "images" / split
            images.extend(list(images_dir.glob("*")))

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Move to correct splits
        for img_path in train_images:
            target = dataset_dir / "images" / "train" / img_path.name
            if img_path != target:
                shutil.move(str(img_path), str(target))

                label_path = (
                    img_path.parent.parent / "labels" / img_path.parent.name / img_path.with_suffix(".txt").name
                )
                if label_path.exists():
                    shutil.move(
                        str(label_path),
                        str(dataset_dir / "labels" / "train" / label_path.name),
                    )

        for img_path in val_images:
            target = dataset_dir / "images" / "val" / img_path.name
            if img_path != target:
                shutil.move(str(img_path), str(target))

                label_path = (
                    img_path.parent.parent / "labels" / img_path.parent.name / img_path.with_suffix(".txt").name
                )
                if label_path.exists():
                    shutil.move(
                        str(label_path),
                        str(dataset_dir / "labels" / "val" / label_path.name),
                    )

        info.train_split = train_ratio
        info.updated_at = datetime.now().isoformat()
        self._save_info(info)

    def compute_statistics(self, name: str) -> Dict[str, Any]:
        """
        Compute dataset statistics.

        Args:
            name: Dataset name.

        Returns:
            Dict with image count, annotation count, class distribution, etc.
        """
        info = self.get_dataset(name)
        dataset_dir = self.base_dir / name

        stats = {
            "name": name,
            "format": info.format,
            "images": {"train": 0, "val": 0, "total": 0},
            "annotations": {"train": 0, "val": 0, "total": 0},
            "classes": {},
        }

        if info.format == "yolo":
            for split in ["train", "val"]:
                images_dir = dataset_dir / "images" / split
                labels_dir = dataset_dir / "labels" / split

                images = list(images_dir.glob("*"))
                stats["images"][split] = len(images)
                stats["images"]["total"] += len(images)

                for label_file in labels_dir.glob("*.txt"):
                    with open(label_file) as f:
                        for line in f:
                            if line.strip():
                                class_id = line.strip().split()[0]
                                stats["classes"][class_id] = stats["classes"].get(class_id, 0) + 1
                                stats["annotations"][split] += 1
                                stats["annotations"]["total"] += 1

        return stats

    def _save_info(self, info: DatasetInfo) -> None:
        """Save dataset info to JSON."""
        info_path = self.base_dir / info.name / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(info.model_dump(), f, indent=2)


class ActiveLearningSelector:
    """
    Select frames for annotation using active learning strategies.

    Implements uncertainty sampling and diversity selection to
    prioritize frames that will most improve model performance.

    Args:
        strategy: Selection strategy ('uncertainty', 'diversity', 'random').

    Example:
        >>> selector = ActiveLearningSelector(strategy='uncertainty')
        >>> frames_to_label = selector.select(
        ...     all_frames, predictions, num_samples=100
        ... )
    """

    def __init__(self, strategy: str = "uncertainty") -> None:
        if strategy not in ("uncertainty", "diversity", "random"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy

    def select(
        self,
        frame_paths: List[str],
        predictions: Optional[List[Dict[str, Any]]] = None,
        num_samples: int = 100,
        seed: int = 42,
    ) -> List[str]:
        """
        Select frames for annotation.

        Args:
            frame_paths: All available frame paths.
            predictions: Model predictions for uncertainty estimation.
            num_samples: Number of frames to select.
            seed: Random seed.

        Returns:
            List of selected frame paths.
        """
        rng = np.random.default_rng(seed)

        if self.strategy == "random":
            indices = rng.choice(len(frame_paths), size=min(num_samples, len(frame_paths)), replace=False)
            return [frame_paths[i] for i in indices]

        if predictions is None:
            # Fall back to random if no predictions
            return self.select(frame_paths, None, num_samples, seed)

        if self.strategy == "uncertainty":
            return self._select_by_uncertainty(frame_paths, predictions, num_samples)

        if self.strategy == "diversity":
            return self._select_by_diversity(frame_paths, predictions, num_samples)

        return frame_paths[:num_samples]

    def _select_by_uncertainty(
        self,
        frame_paths: List[str],
        predictions: List[Dict[str, Any]],
        num_samples: int,
    ) -> List[str]:
        """Select frames with highest prediction uncertainty."""
        uncertainties = []

        for pred in predictions:
            if not pred.get("boxes"):
                # No detections = high uncertainty
                uncertainties.append(1.0)
            else:
                # Average confidence as inverse uncertainty
                confidences = [box.get("confidence", 0.5) for box in pred["boxes"]]
                avg_conf = np.mean(confidences) if confidences else 0.5
                uncertainties.append(1.0 - avg_conf)

        # Sort by uncertainty (descending)
        indices = np.argsort(uncertainties)[::-1][:num_samples]
        return [frame_paths[i] for i in indices]

    def _select_by_diversity(
        self,
        frame_paths: List[str],
        predictions: List[Dict[str, Any]],
        num_samples: int,
    ) -> List[str]:
        """Select diverse frames based on detection patterns."""
        # Simple diversity: cluster by number of detections
        detection_counts = [len(pred.get("boxes", [])) for pred in predictions]

        # Bin by detection count
        bins: Dict[int, List[int]] = {}
        for i, count in enumerate(detection_counts):
            bin_key = min(count // 5, 10)  # Group by 5s, max 50+
            bins.setdefault(bin_key, []).append(i)

        # Sample from each bin
        selected = []
        samples_per_bin = max(1, num_samples // len(bins))

        rng = np.random.default_rng(42)
        for bin_indices in bins.values():
            n = min(samples_per_bin, len(bin_indices))
            sampled = rng.choice(bin_indices, size=n, replace=False)
            selected.extend(sampled)

        # Fill remaining with random
        while len(selected) < num_samples and len(selected) < len(frame_paths):
            idx = rng.choice(len(frame_paths))
            if idx not in selected:
                selected.append(idx)

        return [frame_paths[i] for i in selected[:num_samples]]


__all__ = [
    "DatasetInfo",
    "DatasetManager",
    "ActiveLearningSelector",
]
